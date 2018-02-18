#include "benchmarks.h"

double matrix_matrix_add_int8_mpi(size, threads)
{
    return 1.0;
}

double matrix_matrix_add_int16_mpi(size, threads)
{
    return 2.0;
}

double matrix_matrix_add_int32_mpi(size, threads)
{

    int rank, num_cores;
    MPI_Comm_size(MPI_COMM_WORLD, &num_cores);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    printf("Hello from core %d\n", rank);
    
    if(rank==0)
    {
        return 10.0;
    }
    else
    {
        return 0.0;
    }


}

double matrix_matrix_add_spfp_mpi(size, threads)
{
    return 4.0;
}

double matrix_matrix_add_dpfp_mpi(size, threads)
{
    return 5.0;
}
