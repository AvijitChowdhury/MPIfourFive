/*
   159.735 Parallel Programming - Assignment 5
   MPI Parallel Sum of Integers (Master/Slave with N processes)

   Computes the sum 1 + 2 + ... + N by dividing the range equally
   among all available MPI processes (one master, rest are slaves).

   Compile:  mpic++ second.cpp -o second
   Run:      mpirun -np 20 ./second 1000000
*/

#include <mpi.h>
#include <iostream>
#include <cstdlib>
#include <cmath>

// ----------------------------------------------------------------
// Process ranks
// ----------------------------------------------------------------
const int MASTER = 0;

// MPI message tags
const int TAG_WORK   = 1;   // master -> slave: work assignment
const int TAG_RESULT = 2;   // slave  -> master: partial sum

// ---------------------------------------------------------------
// Structure to send work to a slave
// ---------------------------------------------------------------
struct WorkPacket {
    long long start;   // first integer to sum (inclusive)
    long long end;     // last  integer to sum (inclusive)
};

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // Upper limit N (default 1,000,000; can pass as command-line arg)
    long long N = 1000000LL;
    if (argc > 1) N = atoll(argv[1]);

    // ----------------------------------------------------------------
    // MASTER process: distribute work, collect results
    // ----------------------------------------------------------------
    if (rank == MASTER) {
        double t_start = MPI_Wtime();

        std::cout << "# Processes : " << nprocs   << std::endl;
        std::cout << "# Summing   : 1 .. " << N   << std::endl;

        int num_slaves = nprocs - 1;

        if (num_slaves == 0) {
            // Only one process: do everything here
            long long total = 0;
            for (long long i = 1; i <= N; ++i) total += i;
            double t_end = MPI_Wtime();
            std::cout << "# Sum (serial) : " << total << std::endl;
            std::cout << "# Expected     : " << N*(N+1)/2 << std::endl;
            std::cout << "# Time (s)     : " << (t_end - t_start) << std::endl;
            MPI_Finalize();
            return 0;
        }

        // Divide the range [1..N] into num_slaves equal chunks
        long long chunk = N / num_slaves;
        long long remainder = N % num_slaves;

        // Send work to each slave
        long long cur_start = 1;
        for (int s = 1; s <= num_slaves; ++s) {
            long long cur_end = cur_start + chunk - 1;
            if (s <= remainder) cur_end++;   // distribute leftover

            WorkPacket pkt;
            pkt.start = cur_start;
            pkt.end   = cur_end;
            MPI_Send(&pkt, sizeof(WorkPacket), MPI_BYTE, s, TAG_WORK, MPI_COMM_WORLD);

            cur_start = cur_end + 1;
        }

        // Collect partial sums from all slaves
        long long total = 0;
        for (int s = 1; s <= num_slaves; ++s) {
            long long partial;
            MPI_Recv(&partial, 1, MPI_LONG_LONG, s, TAG_RESULT, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
            total += partial;
        }

        double t_end = MPI_Wtime();

        // Analytic answer for verification: N*(N+1)/2
        long long expected = N * (N + 1) / 2;
        std::cout << "# Computed sum : " << total    << std::endl;
        std::cout << "# Expected sum : " << expected << std::endl;
        std::cout << "# Correct      : " << (total == expected ? "YES" : "NO") << std::endl;
        std::cout << "# Time (s)     : " << (t_end - t_start) << std::endl;

    } else {
        // ----------------------------------------------------------------
        // SLAVE process: receive range, compute partial sum, send back
        // ----------------------------------------------------------------
        WorkPacket pkt;
        MPI_Recv(&pkt, sizeof(WorkPacket), MPI_BYTE, MASTER, TAG_WORK,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        long long partial = 0;
        for (long long i = pkt.start; i <= pkt.end; ++i)
            partial += i;

        MPI_Send(&partial, 1, MPI_LONG_LONG, MASTER, TAG_RESULT, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
