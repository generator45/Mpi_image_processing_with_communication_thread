#include "timer.h"
#include <mpi.h>

double record_time(int stop) {
    static double start_time = 0.0;
    
    if (!stop) {
        // Start timing
        start_time = MPI_Wtime();
        return 0.0;
    } else {
        // Stop timing and return elapsed duration
        return MPI_Wtime() - start_time;
    }
}
