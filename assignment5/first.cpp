/*
   159.735 Parallel Programming - Assignment 5
   MPI Parallel Sum of Integers (Two-Process: Master + One Slave)

   Computes the sum 1 + 2 + ... + N using exactly 2 processes:
   master handles the first half, slave handles the second half.

   Compile:  mpic++ first.cpp -o first
   Run:      mpirun -np 2 ./first 1000000
*/

#include <mpi.h>
#include <iostream>
#include <cstdlib>

const int MASTER   = 0;
const int SLAVE    = 1;
const int TAG_WORK = 1;
const int TAG_RESULT = 2;

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    long long N = 1000000LL;
    if (argc > 1) N = atoll(argv[1]);

    if (rank == MASTER) {
        double t_start = MPI_Wtime();

        // Master handles lower half [1 .. N/2]
        long long half = N / 2;

        // Tell slave its range: [half+1 .. N]
        long long slave_start = half + 1;
        long long slave_end   = N;
        long long range[2] = {slave_start, slave_end};
        MPI_Send(range, 2, MPI_LONG_LONG, SLAVE, TAG_WORK, MPI_COMM_WORLD);

        // Compute master's partial sum
        long long master_sum = 0;
        for (long long i = 1; i <= half; ++i)
            master_sum += i;

        // Receive slave's partial sum
        long long slave_sum = 0;
        MPI_Recv(&slave_sum, 1, MPI_LONG_LONG, SLAVE, TAG_RESULT,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        long long total    = master_sum + slave_sum;
        long long expected = N * (N + 1) / 2;
        double t_end = MPI_Wtime();

        std::cout << "# Two-process MPI sum" << std::endl;
        std::cout << "# N            : " << N        << std::endl;
        std::cout << "# Computed sum : " << total    << std::endl;
        std::cout << "# Expected sum : " << expected << std::endl;
        std::cout << "# Correct      : " << (total == expected ? "YES" : "NO") << std::endl;
        std::cout << "# Time (s)     : " << (t_end - t_start) << std::endl;

    } else {
        // Slave: receive range, compute partial sum, send back
        long long range[2];
        MPI_Recv(range, 2, MPI_LONG_LONG, MASTER, TAG_WORK,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        long long partial = 0;
        for (long long i = range[0]; i <= range[1]; ++i)
            partial += i;

        MPI_Send(&partial, 1, MPI_LONG_LONG, MASTER, TAG_RESULT, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
