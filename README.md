# LaplaceParallelSolver
A high-performance parallel solver for the Laplace equation using Jacobi iteration. This project utilizes C++, MPI for distributed computing, OpenMP for shared memory parallelism.


## Prerequisites
- C++ Compiler
- MPI library (e.g., OpenMPI)
- OpenMP

## Instantiation and Testing

1. Clone the repository:
   ```sh
   git clone git@github.com:lips99poli/pacs_challenge_3.git
   cd pacs_challenge_3
   ```


2. Compile the code:
   ```sh
    make
   ```


To run the solver basic test, use the following command:
```sh
mpirun -np <number_of_processes> ./main
```
