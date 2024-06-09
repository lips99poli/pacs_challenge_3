#include "Jacobi.hpp"

double f(double x, double y){
    return 8*M_PI*M_PI*sin(2*M_PI*x)*sin(2*M_PI*y);
}

double u(double x, double y){
    return sin(2*M_PI*x)*sin(2*M_PI*y);
}


int main(int argc, char** argv){
    MPI_Init(&argc, &argv);
    int rank,size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);


    unsigned int N = 11;
    auto bc_fun = [](double x, double y){return 0.;};
    std::map<std::string, std::function<double(double, double)>> bc_funs;
    bc_funs["Nord"] = bc_fun;
    bc_funs["Sud"] = bc_fun;
    bc_funs["Est"] = bc_fun;
    bc_funs["Ovest"] = bc_fun;
    BoundaryCondition<double> bc(N, bc_funs);
    Jacobi<double> j(1e4,1e-10,N);
    j.setBoundaryCondition(bc);
    j.setFunction(f);
    j.solve();
    j.print();

    // rank 0 will construct and print the solution
    std::cout<<std::endl;
    if(rank == 0){
        Eigen::MatrixXd solution(N, N);
        std::cout << "Solution: " << std::endl;
        for(unsigned int i = 0; i < N; i++){
            for(unsigned int j = 0; j < N; j++){
                solution(i, j) = u(i/(N-1.), j/(N-1.));
            }
        }
        std::cout << solution << std::endl;
    }

    MPI_Finalize();
    return 0;
}