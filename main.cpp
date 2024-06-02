#include "Jacobi.hpp"

double f(double x, double y){
    return 8*M_PI*M_PI*sin(2*M_PI*x)*sin(2*M_PI*y);
}


int main(int argc, char** argv){
    MPI_Init(&argc, &argv);
    unsigned int N = 10;
    auto bc_fun = [](unsigned int i, unsigned int j){return 0.;};
    BoundaryCondition<double> bc(N, N, bc_fun, bc_fun, bc_fun, bc_fun);
    Jacobi<double> j;
    j.setBoundaryCondition(bc);
    j.setFunction(f);
    j.solve();
    j.print();

    MPI_Finalize();
    return 0;
}