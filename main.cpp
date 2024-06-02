#include "Jacobi.hpp"

double f(double x, double y){
    return 8*M_PI*M_PI*sin(2*M_PI*x)*sin(2*M_PI*y);
}

double u(double x, double y){
    return sin(2*M_PI*x)*sin(2*M_PI*y);
}


int main(int argc, char** argv){
    MPI_Init(&argc, &argv);
    unsigned int N = 20;
    auto bc_fun = [](unsigned int i, unsigned int j){return 0.;};
    BoundaryCondition<double> bc(N, N, bc_fun, bc_fun, bc_fun, bc_fun);
    Jacobi<double> j(1e4,1e-10,N);
    j.setBoundaryCondition(bc);
    j.setFunction(f);
    j.solve();
    j.print();

    // rank 0 will construct and print the solution
    std::cout<<std::endl;
    if(j.rank == 0){
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