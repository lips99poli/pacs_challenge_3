#ifndef JACOBI_HPP
#define JACOBI_HPP

#include <vector>
#include <Eigen/Dense>
#include <functional>
#include "mpi.h"

/* da fare: 
- griglia della funzione così non la devo chiamare ed eseguire ogni volta!!
- anzichè row_above e row_under ingrandisci local_grid
- cambiare boundary condition così: [mappa identificatore della zona -> funzione constraint di quella zona] + [mappa con set di indici -> identificatore per quella zona]
    poi quando le devo applicare vado a cercare l'indice, cerco il suo identificatore di zona e poi entro con quell'identificatore nella mappa e prendo la funzione da applicargli
- concetpt is_scalar per T, non vogliamo numeri complessi
- aggiungere al constructor inizializzazione dei 4 vertici
- aggiungere soluzione alternativa di comunicazione con sendrecv
*/

template <typename T>
struct BoundaryCondition{
    unsigned int grid_row;
    unsigned int grid_col;

    std::function<T(unsigned int i, unsigned int j)> nord;
    std::function<T(unsigned int i, unsigned int j)> est;
    std::function<T(unsigned int i, unsigned int j)> sud;
    std::function<T(unsigned int i, unsigned int j)> ovest;
};

template <typename T>
class Jacobi {

    // Solver parameters: maximum number of iteration, tolerance and number of points
    unsigned int maxIt;
    T tol;
    unsigned int N;

    // Edge of the grid
    T x0 = 0.;
    T y0 = 0.;
    T xN = 1.;
    T yN = 1.;

    // Spacing and its square
    T h;
    T h_2;

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> grid; // U
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> function_grid; //f(U)

    // MPI Variables are members since it is a parallel solver
    int rank;
    int size;



    // Boundary Condition
    BoundaryCondition<T> bc;

    // Object function of the problem
    std::function<T(T,T)> f;

    // Save the error
    T error;

    public:
    Jacobi(unsigned int maxIt=1e3, T tol=1e-6, unsigned int N=1e3) : maxIt(maxIt), tol(tol), N(N), grid(N,N), function_grid(N-1,N-1){
        grid.setZero();
        h = (xN-x0)/N; //se lo lasci così lo spacing è omogeneo lungo x e y
        h_2 = h*h;

        // Initialize MPI variables
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        };

    void setFunction(std::function<T(T,T)>& f){
        this->f = f;

        // Parallel Initialization of the function grid
        unsigned int local_rows = (N-2)%size > rank ? (N-2)/size + 1 : (N-2)/size;
        unsigned int cols = N-2;
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> local_function_grid(local_rows, cols);
        local_function_grid.setZero();

        // Each rank has to know what is its first row w.r.t. original grid
        unsigned int first_row = (N-2)/size * rank + std::min(rank, (N-2)%size); // or unsigned int first_row = ((N-2)%size > rank) ? (N-2)/size * rank + rank  : (N-2)/size * rank + (N-2)%size;        

        // Fill local matrix, recall that w.r.t. the full grid we avoid the boundaries so sum 1 in the indexes
        for(Eigen::Index i=0; i<local_rows; ++i){
            for(Eigen::Index j=0; j<cols; ++j){
                local_function_grid(i,j) = f((first_row + i + 1)*h, (j + 1)*h);
            }
        }

        // Define the things i need to Allgather the results
        std::vector<int> displs(size);
        std::vector<int> sendcounts(size);
        int sum = 0;
        for (int i = 0; i < size; ++i) {
            sendcounts[i] = (N-2)%size > i ? (N-2)/size + 1 : (N-2)/size;
            displs[i] = sum;
            sum += sendcounts[i];
        }

        // Copy the local grid to the global grid
        MPI_Allgatherv(local_function_grid.data(), local_rows*(N-2), MPI_DOUBLE, function_grid.data(), sendcounts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);

    }

    void setBoundaryCondition(BoundaryCondition<T>& bc){
        this->bc = bc;
    };

    bool solve(){

        // Create local grid: rows block partitioned
        // rank 0 and rank size-1 have 1 more row to contain respectively row under and row above,
        // All other ranks have 2 more rows to contain both row_under and row_above
        // These rows will be updated every time before the iteration start, with the infromation coming from adjacent ranks
        unsigned int cols = grid.cols();
        unsigned int local_rows = (grid.rows()%size > rank) ? grid.rows()/size + 1 + 2 : grid.rows()/size + 2;
        if (rank==0 || rank==size-1) --local_rows;

        unsigned int first_row = N/size * rank + std::min(rank, N%size);

        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> local_grid(local_rows, cols);
        local_grid.setZero();

        // Each rank has to apply boundary condition on its local grid
        apply_boundary_condtion(local_grid, rank, size);

        // We need a container with the updates
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> new_local_grid = local_grid;

        // The goal is performance: in the first iteration i don't need any comuncation, which is needed at the beginning of every iteration, AFTER THE CONVERGENCE CHECK!
        // In order to avoid useless communication i want to do the for loop starting with the communication and then perform updates and finally check convergence. But since the 
        // the first communication is useless I perform the first iteration outside and then enter the loop
        // The matrix is initialized with all zeros so I can do the first iteration outside the loop, and without sending anything
        // This is actually not true, imagine the extreme case where each process has one line: the second rank needs the line of the first because is a boundary so i need to send it
        // Even better, everything goes in the loop

        // Jacobi iteration
        bool go_on = true;
        for (unsigned int it = 0; go_on; ++it) {
            communicate_rows(local_grid, rank, size);
            
            // if (rank==0) I shouldn't touch the first row and the first and last columns
            // if(rank==size-1) I shouldn't touch the last row and the first and last columns
            // else I shouldn't touch the first and last rows and columns
            // Eigen::Index startRow = (rank == 0) ? 1 : 0;
            // Eigen::Index endRow = (rank == size - 1) ? local_rows - 1 : local_rows;
            // Guarda che la prima riga e l'ultima o sono bordi o sono inviate dagli altri rank quindi non le cambio mai!

            // First and rows are never changed in any rank because either they are boundaries (rank 0, size-1) or they are row_above/row_under, so they are updated by other ranks
            // Recall that the function_grid doesn't have the 4 edges of the square so when calling it with indexing of the real grid i should subtract one. Moreover i should know what row each rank is starting from

            for (Eigen::Index i=1; i<local_rows-1; ++i) {
                for (Eigen::Index j=1; j<cols-1; ++j) {
                    // Update
                    new_local_grid(i, j) = 0.25 * (
                        local_grid(i-1, j) + local_grid(i+1, j) + local_grid(i, j-1) + local_grid(i, j+1) + 
                        h_2 * function_grid(first_row + i -1, j-1)
                    );
                }
            }
            go_on = !check_convergence(local_grid,new_local_grid) && it < maxIt;
            local_grid.swap(new_local_grid);
        }

        // Define the things i need to gather the results
        std::vector<int> displs(size);
        std::vector<int> sendcounts(size);
        int sum = 0;
        for (int i = 0; i < size; ++i) {
            sendcounts[i] = (grid.rows()%size > i) ? grid.rows()/size + 1 : grid.rows()/size;
            displs[i] = sum;
            sum += sendcounts[i];
        }

        // Copy the local grid to the global grid
        unsigned offset = (rank==0)? 0 : cols;
        unsigned count = (rank==0 || rank==size-1)? local_rows-1 : local_rows-2;
        MPI_Allgatherv(local_grid.data() + offset, count, MPI_DOUBLE, grid.data(), sendcounts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    }

    private:

    void apply_boundary_condtion(Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic, Eigen::RowMajor>& local_grid, int r, int s) const {
        unsigned int first_row = N/size * rank + std::min(rank, N%size);
        if(r==0){// Nord
            for(unsigned int j=0; j<grid.cols(); j++){
                local_grid(0,j) = bc.nord(0,j);
            }
        }else if(r==s-1){// Sud
            for(unsigned int j=0; j<j<grid.cols(); j++){
                local_grid(local_grid.rows()-1,j) = bc.sud(grid.rows(),j);
            }
        }else{ // Est Ovest
            // Recall that in local_grid there are also row_above and row_under, so ignore them with proper indexing
            for(unsigned int i=1; i<local_grid.rows()-1; i++){
                local_grid(i,0) = bc.ovest(i,0);
                local_grid(i,grid.cols()-1) = bc.est(first_row + i,grid.cols()-1);
            }
        }
    };

    void comunicate_rows(Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic, Eigen::RowMajor>& local_grid){
        unsigned int local_rows = local_grid.rows();
        unsigned int cols = local_grid.cols();

        int prev_rank = rank - 1;
        int next_rank = rank + 1;
        MPI_Request requests[4] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL};
        if (rank > 0) {  // Communicate with previous rank
            MPI_Isend(local_grid.row(1).data(), cols, MPI_DOUBLE, prev_rank, 0, MPI_COMM_WORLD, &requests[0]);
            MPI_Irecv(local_grid.row(0).data(), cols, MPI_DOUBLE, prev_rank, 1, MPI_COMM_WORLD, &requests[1]);
        }
        if (rank < size - 1) {  // Communicate with next rank
            MPI_Isend(local_grid.row(local_rows - 2).data(), cols, MPI_DOUBLE, next_rank, 1, MPI_COMM_WORLD, &requests[2]);
            MPI_Irecv(local_grid.row(local_rows - 1).data(), cols, MPI_DOUBLE, next_rank, 0, MPI_COMM_WORLD, &requests[3]);
        }
        // Determine the correct start point for the MPI_Request array
        MPI_Request* request_start = (rank == 0) ? &requests[2] : requests;
        MPI_Waitall((rank > 0 && rank < size - 1) ? 4 : 2, request_start, MPI_STATUSES_IGNORE);

        /* Alternative with SendRecv:
        // Utilizzo MPI_Sendrecv per comunicare con il rango precedente
        if (rank > 0) {  // Comunica con il rank precedente
            MPI_Sendrecv(local_grid.row(1).data(), cols, MPI_DOUBLE, prev_rank, 0,
                        local_grid.row(0).data(), cols, MPI_DOUBLE, prev_rank, 1,
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // Utilizzo MPI_Sendrecv per comunicare con il rango successivo
        if (rank < size - 1) {  // Comunica con il rank successivo
            MPI_Sendrecv(local_grid.row(local_rows - 2).data(), cols, MPI_DOUBLE, next_rank, 1,
                        local_grid.row(local_rows - 1).data(), cols, MPI_DOUBLE, next_rank, 0,
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        */

        // Synchronize all ranks before proceeding to the next step
        MPI_Barrier(MPI_COMM_WORLD);
    };

    // Alternative with SendRecv 

    bool check_convergence(const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic, Eigen::RowMajor>& local_grid, const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic, Eigen::RowMajor>& new_local_grid)const {
        error = 0.;
        
        // Compute the error, i don't need to check the first and last row because they are either boundary conditions or they are updated by other ranks
        for(Eigen::Index i=1; i<local_grid.rows()-1; ++i){
            for(Eigen::Index j=1; j<local_grid.cols()-1; ++j){
                T diff = local_grid(i,j) - new_local_grid(i,j);
                error += diff*diff; 
            }
        }
        MPI_Allreduce(MPI_IN_PLACE, &error, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        error = std::sqrt(h*error);
        return error < tol;
    }
};

#endif