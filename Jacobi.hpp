#ifndef JACOBI_HPP
#define JACOBI_HPP

#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <functional>
#include "mpi.h"
#include <concepts>
#include <set>
#include <map>

/* da fare: 
- griglia della funzione così non la devo chiamare ed eseguire ogni volta!!
- anzichè row_above e row_under ingrandisci local_grid
- cambiare boundary condition così: [mappa identificatore della zona -> funzione constraint di quella zona] + [mappa con set di indici -> identificatore per quella zona]
    poi quando le devo applicare vado a cercare l'indice, cerco il suo identificatore di zona e poi entro con quell'identificatore nella mappa e prendo la funzione da applicargli
- concetpt is_scalar per T, non vogliamo numeri complessi
- aggiungere al constructor inizializzazione dei 4 vertici
- aggiungere soluzione alternativa di comunicazione con sendrecv
*/


// The template parameter is useful to define the precision of the solver
template <typename T>
struct BoundaryCondition{
    // When defining the the boundary condition the user is obliged to provide a function for each boundary, then if he doesn't give a second argument the default is a square (since the implementation is only for square grids)
    // Nord, Sud, Est, Ovest with the entire edge of the square.
    // If he wants to customize it he has to send also a map with the indexes grouped by the boundary name

    // I need also the dimension of the grid
    unsigned int N;

    std::map<std::string,std::function<T(T,T)>> boundary_functions;
    std::map<std::string,std::set<std::pair<unsigned int, unsigned int>>> boundary_indexes;

    BoundaryCondition() = default;

    BoundaryCondition(unsigned int N, std::map<std::string,std::function<T(T,T)>> boundary_functions, std::map<std::string,std::set<std::pair<unsigned int, unsigned int>>> boundary_indexes = {}): 
                        N(N), boundary_functions(boundary_functions), boundary_indexes(boundary_indexes){
                            // if the user doesn't provide boundaries_indexes definition, then the default is the entire edge of the square
                            if(boundary_indexes.empty()){
                                for(std::size_t i=0; i<N; ++i){
                                    boundary_indexes["Nord"].insert({0,i});
                                    boundary_indexes["Sud"].insert({N-1,i});
                                    boundary_indexes["Est"].insert({i,N-1});
                                    boundary_indexes["Ovest"].insert({i,0});
                                }
                            }
                        };
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

    // MPI Variables are members since it is a parallel solver
    int rank;
    int size;

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
    Jacobi(unsigned int maxIt=1e4, T tol=1e-20, unsigned int N=11);

    void setFunction(std::function<T(T,T)> f);

    void setBoundaryCondition(BoundaryCondition<T>& bc);

    void solve();

    void print() const {
        if(rank==0){
            std::cout << "Solution:\n" << grid << std::endl;
            std::cout << "Relative error: " << error << std::endl;
            std::cout << "Number of iteration performed: " << it << std::endl;
        }
    }

    private:

    void apply_boundary_condtion();

    void communicate_rows(Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic, Eigen::RowMajor>& local_grid);

    bool check_convergence(const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>& new_local_grid);

};

// Implementation of the Jacobi class

template <typename T>
Jacobi<T>::Jacobi(unsigned int maxIt, T tol, unsigned int N) : maxIt(maxIt), tol(tol), N(N), grid(N,N){
    // Initialize MPI variables
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    grid.setZero();
    h = (xN-x0)/N; //se lo lasci così lo spacing è omogeneo lungo x e y
    h_2 = h*h;

    // Create local grid: rows block partitioned
    // rank 0 and rank size-1 have 1 more row to contain respectively row above and row under,
    // All other ranks have 2 more rows to contain both row_under and row_above
    // These rows will be updated every time before the iteration start, with the infromation coming from adjacent ranks
    unsigned int local_rows = (N%size > rank) ? N/size + 1 + 2 : N/size + 2;
    if (rank==0 || rank==size-1) --local_rows;

    first_row = N-(N/size * rank + std::min(rank, static_cast<int> (N%size)));
    
    local_grid.resize(local_rows, N);
    local_grid.setZero();

    // Create local function grid: for each rank i need the same rows that I have in the local grid, excluding the boundaries and row_above/row_under
    unsigned int function_local_rows = local_rows - 2;
    local_function_grid.resize(function_local_rows, N-2);
}


template <typename T>
void Jacobi<T>::setFunction(std::function<T(T,T)> f){
    this->f = f;

    // Initialization of the local function grid
    // Each rank has to know what is its first row w.r.t. original grid
    unsigned int cols = N-2;
    unsigned int local_rows = local_function_grid.rows();
    unsigned int first_row = (N-2)/size * rank + std::min(rank, static_cast<int> (N-2)%size); // or unsigned int first_row = ((N-2)%size > rank) ? (N-2)/size * rank + rank  : (N-2)/size * rank + (N-2)%size;        

    // Fill local matrix, recall that w.r.t. the full grid we avoid the boundaries so sum 1 in the indexes
    for(Eigen::Index i=0; i<local_rows; ++i){
        for(Eigen::Index j=0; j<cols; ++j){
            local_function_grid(i,j) = f((first_row - i - 1)*h, (j + 1)*h);
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


template <typename T>
void Jacobi<T>::setBoundaryCondition(BoundaryCondition<T>& bc){
    this->bc = bc;

    // Each rank has to apply boundary condition on its local grid
    apply_boundary_condtion();

    // We need a container with the updates
    new_local_grid = local_grid;
}


template <typename T>
void Jacobi<T>::apply_boundary_condtion(){ //not efficient but general, it has to be run only at the definition of the problem, no need of more run.
    // Recall that in local_grid there are also row_above and row_under, so ignore them with proper indexing
    for(unsigned int i_loc=0; i_loc<local_grid.rows(); ++i_loc){
        for (unsigned int j=0; j<local_grid.cols(); ++j){
            unsigned int i = first_row - i_loc - 1;
            // if i am not on the edges i skip
            if(i!=0 && j!=0 && i!=N-1 && j!=N-1) continue;
            
             // find the index corresponding boundary 
            for(const auto& boundary : bc.boundary_indexes){
                if(boundary.second.find({i, j}) != boundary.second.end()){
                    local_grid(i_loc,j) = bc.boundary_functions[boundary.first](i*h,j*h);
                }
            }
        }
    }
}


// Solve the problem
template <typename T>
void Jacobi<T>::solve(){
    unsigned int cols = local_grid.cols();
    unsigned int local_rows = local_grid.rows();

    // Why I don't do first iteration outside the loop?
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

        // First and rows are never changed in any rank because either they are boundaries (rank 0, size-1) or they are row_above/row_under, so they are updated by other ranks
        // Recall that the function_grid doesn't have the 4 edges of the square so when calling it with indexing of the real grid i should subtract one. Moreover i should know what row each rank is starting from

        #pragma omp parallel for collapse(2)
        for (Eigen::Index i=1; i<local_rows-1; ++i) {
            for (Eigen::Index j=1; j<cols-1; ++j) {
                // Update
                new_local_grid(i, j) = 0.25 * (
                    local_grid(i-1, j) + local_grid(i+1, j) + local_grid(i, j-1) + local_grid(i, j+1) + 
                    h_2 * local_function_grid(i-1, j-1)
                );
            }
        }
        go_on = !check_convergence(new_local_grid) && it < maxIt;
        local_grid.swap(new_local_grid);
    }

    // Define the things i need to gather the results
    std::vector<int> displs(size);
    std::vector<int> sendcounts(size);
    int sum = N*N;
    for (int i = 0; i < size; ++i) {
        sendcounts[i] = (N%size > i) ? (N/size + 1)*cols : (N/size)*cols;
        sum -= sendcounts[i];
        displs[i] = sum;
    }

    // The first line shouldn't be sent, a part from rank 0
    unsigned offset = (rank==size-1)? 0 : cols;

    // Copy the local grid to the global grid
    MPI_Allgatherv(local_grid.data() + offset, sendcounts[rank], MPI_DOUBLE, grid.data(), sendcounts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
}

template <typename T>
void Jacobi<T>::communicate_rows(Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic, Eigen::RowMajor>& local_grid){
    unsigned int local_rows = local_grid.rows();
    unsigned int cols = local_grid.cols();

    int prev_rank = rank - 1;
    int next_rank = rank + 1;
    MPI_Request requests[4] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL};
    if (rank > 0) {  // Communicate with previous rank
        MPI_Isend(local_grid.row(local_rows-2).data(), cols, MPI_DOUBLE, prev_rank, 0, MPI_COMM_WORLD, &requests[0]);
        MPI_Irecv(local_grid.row(local_rows-1).data(), cols, MPI_DOUBLE, prev_rank, 1, MPI_COMM_WORLD, &requests[1]);
    }
    if (rank < size - 1) {  // Communicate with next rank
        MPI_Isend(local_grid.row(1).data(), cols, MPI_DOUBLE, next_rank, 1, MPI_COMM_WORLD, &requests[2]);
        MPI_Irecv(local_grid.row(0).data(), cols, MPI_DOUBLE, next_rank, 0, MPI_COMM_WORLD, &requests[3]);
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

template <typename T>
bool Jacobi<T>::check_convergence(const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>& new_local_grid){
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




#endif