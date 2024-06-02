#ifndef JACOBI_HPP
#define JACOBI_HPP

#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <functional>
#include "mpi.h"
#include <concepts>

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
    unsigned int grid_row=100;
    unsigned int grid_col=100;

    std::function<T(unsigned int i, unsigned int j)> nord=nullptr;
    std::function<T(unsigned int i, unsigned int j)> est=nullptr;
    std::function<T(unsigned int i, unsigned int j)> sud=nullptr;
    std::function<T(unsigned int i, unsigned int j)> ovest=nullptr;

    BoundaryCondition(unsigned int grid_row, unsigned int grid_col, std::function<T(unsigned int i, unsigned int j)> nord, std::function<T(unsigned int i, unsigned int j)> est, std::function<T(unsigned int i, unsigned int j)> sud, std::function<T(unsigned int i, unsigned int j)> ovest) : grid_row(grid_row), grid_col(grid_col), nord(nord), est(est), sud(sud), ovest(ovest){};
    BoundaryCondition() = default;
};

template <typename T>
class Jacobi {

    // Solver parameters: maximum number of iteration, tolerance and number of points
    unsigned int maxIt;
    T tol;
    unsigned int N; 

    // Edge of the grid (hp that it is a square grid)
    T x0 = 0.;
    T xN = 1.;

    // Spacing and its square
    T h;
    T h_2;

    // Common members
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> grid; // U
    // Local members
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> local_function_grid; //f(U)
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> local_grid;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> new_local_grid;
    unsigned int first_row;

    public:
    // MPI Variables are members since it is a parallel solver
    int rank;
    int size;

    private:
    // Boundary Condition
    BoundaryCondition<T> bc;

    // Object function of the problem
    std::function<T(T,T)> f;

    // Save the error
    T error = 0.;

    // Save the number of iteration
    unsigned int it = 0;

    public:
    // The user gives me the number of intervals he wants, so the number of points is N+1
    Jacobi(unsigned int maxIt=1e4, T tol=1e-20, unsigned int N=11) : maxIt(maxIt), tol(tol), N(N), grid(N,N){
        // Initialize MPI variables
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        grid.setZero();
        h = (xN-x0)/(N-1); //se lo lasci così lo spacing è omogeneo lungo x e y
        h_2 = h*h;

        // Create local grid: rows block partitioned
        // rank 0 and rank size-1 have 1 more row to contain respectively row under and row above,
        // All other ranks have 2 more rows to contain both row_under and row_above
        // These rows will be updated every time before the iteration start, with the infromation coming from adjacent ranks
        unsigned int local_rows = (N%size > rank) ? N/size + 1 + 2 : N/size + 2;
        if (rank==0 || rank==size-1) --local_rows;

        first_row = N/size * rank + std::min(rank, static_cast<int> (N%size));
        
        local_grid.resize(local_rows, N);
        local_grid.setZero();

        // Create local function grid: for each rank i need the same rows that I have in the local grid, excluding the boundaries and row_above/row_under
        unsigned int function_local_rows = local_rows - 2;
        local_function_grid.resize(function_local_rows, N-2);
        }


    void setFunction(std::function<T(T,T)> f){
        this->f = f;

        // Initialization of the local function grid
        // Each rank has to know what is its first row w.r.t. original grid
        unsigned int cols = N-2;
        unsigned int local_rows = local_function_grid.rows();
        unsigned int first_row = (N-2)/size * rank + std::min(rank, static_cast<int> (N-2)%size); // or unsigned int first_row = ((N-2)%size > rank) ? (N-2)/size * rank + rank  : (N-2)/size * rank + (N-2)%size;        

        // Fill local matrix, recall that w.r.t. the full grid we avoid the boundaries so sum 1 in the indexes
        for(Eigen::Index i=0; i<local_rows; ++i){
            for(Eigen::Index j=0; j<cols; ++j){
                local_function_grid(i,j) = f((first_row + i + 1)*h, (j + 1)*h);
            }
        }

        /* Check of the comunication
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> function_grid(N-2, N-2);
        std::vector<int> displs(size);
        std::vector<int> sendcounts(size);
        int sum = 0;
        for (int i = 0; i < size; ++i) {
            sendcounts[i] = (N-2)%size > i ? (N-2)/size + 1 : (N-2)/size;
            sendcounts[i] *= N-2;
            displs[i] = sum;
            sum += sendcounts[i];
        }
        // Copy the local grid to the global grid
        MPI_Allgatherv(local_function_grid.data(), local_rows*(N-2), MPI_DOUBLE, function_grid.data(), sendcounts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        if (rank==0){
            std::cout << function_grid << std::endl;
        }
        */
    }

    void setBoundaryCondition(BoundaryCondition<T>& bc){
        this->bc = bc;

        // Each rank has to apply boundary condition on its local grid
        apply_boundary_condtion();

        // We need a container with the updates
        new_local_grid = local_grid;
    };


    void solve(){
        unsigned int cols = local_grid.cols();
        unsigned int local_rows = local_grid.rows();
    
        // The goal is performance: in the first iteration i don't need any comuncation, which is needed at the beginning of every iteration, AFTER THE CONVERGENCE CHECK!
        // In order to avoid useless communication i want to do the for loop starting with the communication and then perform updates and finally check convergence. But since the 
        // the first communication is useless I perform the first iteration outside and then enter the loop
        // The matrix is initialized with all zeros so I can do the first iteration outside the loop, and without sending anything
        // This is actually not true, imagine the extreme case where each process has one line: the second rank needs the line of the first because is a boundary so i need to send it
        // Even better, everything goes in the loop

        // Jacobi iteration
        bool go_on = true;
        it = 0;
        for (; go_on; ++it) {
            communicate_rows(local_grid);
            
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
                        h_2 * local_function_grid(i-1, j-1)
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
            sendcounts[i] = (N%size > i) ? (N/size + 1)*cols : (N/size)*cols;
            displs[i] = sum;
            sum += sendcounts[i];
        }

        // The first line shouldn't be sent, a part from rank 0
        unsigned offset = (rank==0)? 0 : cols;

        // Copy the local grid to the global grid
        MPI_Allgatherv(local_grid.data() + offset, sendcounts[rank], MPI_DOUBLE, grid.data(), sendcounts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    }

    void print() const {
        if(rank==0){
            std::cout << "The solution is:" << grid << std::endl;
            std::cout << "The error is: " << error << std::endl;
            std::cout << "The number of iteration performed is: " << it << std::endl;
        }
    }

    private:

    void apply_boundary_condtion() {
        // Recall that in local_grid there are also row_above and row_under, so ignore them with proper indexing
        for(unsigned int i=1; i<local_grid.rows()-1; ++i){ // Est Ovest
            local_grid(i,0) = 0.;
            local_grid(i,local_grid.cols()-1) = 0.;
        }
        // Moreover we have upper and lower boundaries
        if(rank==0){// Nord
            for(unsigned int j=0; j<local_grid.cols(); ++j){
                local_grid(0,j) = 0.;
            }
        }else if(rank==size-1){// Sud
            for(unsigned int j=0; j<local_grid.cols(); ++j){
                local_grid(local_grid.rows()-1,j) = 0.;
            }
        }
    };

    void communicate_rows(Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic, Eigen::RowMajor>& local_grid){
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

    bool check_convergence(const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic, Eigen::RowMajor>& local_grid, const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic, Eigen::RowMajor>& new_local_grid) {
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