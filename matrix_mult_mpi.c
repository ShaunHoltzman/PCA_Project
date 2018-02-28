#include "benchmarks.h"

double matrix_matrix_mult_int8_mpi(size, threads)
{
    srand(SEED);
    int rank, num_cores;
    MPI_Comm_size(MPI_COMM_WORLD, &num_cores);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    int8_t *A;
    int8_t *B;
    int16_t *C;
    
    int i,j,k;
    
    TIME_STRUCT start,end;
    double runtime=0.0;
    
    if(rank==0)
    {
        A=malloc(sizeof(int8_t)*(size*size));
        B=malloc(sizeof(int8_t)*(size*size));
        C=malloc(sizeof(int16_t)*(size*size));
        
        for(i=0; i<size*size; i++)
        {
            A[i]=(int8_t)rand();
            B[i]=(int8_t)rand();
            C[i]=0;
        }
    }
    else
    {
        A=malloc(sizeof(int8_t)*(size/threads)*size);
        B=malloc(sizeof(int8_t)*(size*size));
        C=malloc(sizeof(int16_t)*(size*size));
    }
    
    if(rank==0)
    {
        TIME_GET(&start);
    }
    
    MPI_Scatter(A, (size/threads)*size, MPI_CHAR, A, (size/threads)*size, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(B, size*size, MPI_CHAR, 0, MPI_COMM_WORLD);
    
    for(i=0; i<size/threads; i++)
    {
        for(k=0; k<size; k++)
        {
            for(j=0; j<size; j++)
            {
                C[i*size+j]+=A[i*size+k]*B[k*size+j];
            }
        }
    }
    
    MPI_Gather(C, (size/threads)*size, MPI_SHORT, C, (size/threads)*size, MPI_SHORT, 0, MPI_COMM_WORLD);
    
    if(rank==0)
    {
        TIME_GET(&end);
        runtime=TIME_RUNTIME(start,end);
    }
    
    free(A);
    free(B);
    free(C);
    
    return runtime;

}

double matrix_matrix_mult_int16_mpi(size, threads)
{
    srand(SEED);
    int rank, num_cores;
    MPI_Comm_size(MPI_COMM_WORLD, &num_cores);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    int8_t *A;
    int8_t *B;
    int16_t *C;
    
    int i,j,k;
    
    TIME_STRUCT start,end;
    double runtime=0.0;
    
    if(rank==0)
    {
        A=malloc(sizeof(int16_t)*(size*size));
        B=malloc(sizeof(int16_t)*(size*size));
        C=malloc(sizeof(int32_t)*(size*size));
        
        for(i=0; i<size*size; i++)
        {
            A[i]=(int16_t)rand();
            B[i]=(int16_t)rand();
            C[i]=0;
        }
    }
    else
    {
        A=malloc(sizeof(int16_t)*(size/threads)*size);
        B=malloc(sizeof(int16_t)*(size*size));
        C=malloc(sizeof(int32_t)*(size*size));
    }
    
    if(rank==0)
    {
        TIME_GET(&start);
    }
    
    MPI_Scatter(A, (size/threads)*size, MPI_SHORT, A, (size/threads)*size, MPI_SHORT, 0, MPI_COMM_WORLD);
    MPI_Bcast(B, size*size, MPI_SHORT, 0, MPI_COMM_WORLD);
    
    for(i=0; i<size/threads; i++)
    {
        for(k=0; k<size; k++)
        {
            for(j=0; j<size; j++)
            {
                C[i*size+j]+=A[i*size+k]*B[k*size+j];
            }
        }
    }
    
    MPI_Gather(C, (size/threads)*size, MPI_INT, C, (size/threads)*size, MPI_INT, 0, MPI_COMM_WORLD);
    
    if(rank==0)
    {
        TIME_GET(&end);
        runtime=TIME_RUNTIME(start,end);
    }
    
    free(A);
    free(B);
    free(C);
    
    return runtime;
    
}

double matrix_matrix_mult_int32_mpi(size, threads)
{
    srand(SEED);
    int rank, num_cores;
    MPI_Comm_size(MPI_COMM_WORLD, &num_cores);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    int32_t *A;
    int32_t *B;
    int64_t *C;
    
    int i,j,k;
    
    TIME_STRUCT start,end;
    double runtime=0.0;
    
    if(rank==0)
    {
        A=malloc(sizeof(int32_t)*(size*size));
        B=malloc(sizeof(int32_t)*(size*size));
        C=malloc(sizeof(int64_t)*(size*size));
        
        for(i=0; i<size*size; i++)
        {
            A[i]=(int32_t)rand();
            B[i]=(int32_t)rand();
            C[i]=0;
        }
    }
    else
    {
        A=malloc(sizeof(int32_t)*(size/threads)*size);
        B=malloc(sizeof(int32_t)*(size*size));
        C=malloc(sizeof(int64_t)*(size*size));
    }
    
    if(rank==0)
    {
        TIME_GET(&start);
    }
    
    MPI_Scatter(A, (size/threads)*size, MPI_INT, A, (size/threads)*size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(B, size*size, MPI_INT, 0, MPI_COMM_WORLD);
    
    for(i=0; i<size/threads; i++)
    {
        for(k=0; k<size; k++)
        {
            for(j=0; j<size; j++)
            {
                C[i*size+j]+=A[i*size+k]*B[k*size+j];
            }
        }
    }
    
    MPI_Gather(C, (size/threads)*size, MPI_LONG, C, (size/threads)*size, MPI_LONG, 0, MPI_COMM_WORLD);
    
    if(rank==0)
    {
        TIME_GET(&end);
        runtime=TIME_RUNTIME(start,end);
    }
    
    free(A);
    free(B);
    free(C);
    
    return runtime;
    
}

double matrix_matrix_mult_spfp_mpi(size, threads)
{
    srand(SEED);
    int rank, num_cores;
    MPI_Comm_size(MPI_COMM_WORLD, &num_cores);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    int8_t *A;
    int8_t *B;
    int16_t *C;
    
    int i,j,k;
    
    TIME_STRUCT start,end;
    double runtime=0.0;
    
    if(rank==0)
    {
        A=malloc(sizeof(float)*(size*size));
        B=malloc(sizeof(float)*(size*size));
        C=malloc(sizeof(float)*(size*size));
        
        for(i=0; i<size*size; i++)
        {
            A[i]=(float)rand();
            B[i]=(float)rand();
            C[i]=0;
        }
    }
    else
    {
        A=malloc(sizeof(float)*(size/threads)*size);
        B=malloc(sizeof(float)*(size*size));
        C=malloc(sizeof(float)*(size*size));
    }
    
    if(rank==0)
    {
        TIME_GET(&start);
    }
    
    MPI_Scatter(A, (size/threads)*size, MPI_FLOAT, A, (size/threads)*size, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(B, size*size, MPI_FLOAT, 0, MPI_COMM_WORLD);
    
    for(i=0; i<size/threads; i++)
    {
        for(k=0; k<size; k++)
        {
            for(j=0; j<size; j++)
            {
                C[i*size+j]+=A[i*size+k]*B[k*size+j];
            }
        }
    }
    
    MPI_Gather(C, (size/threads)*size, MPI_FLOAT, C, (size/threads)*size, MPI_FLOAT, 0, MPI_COMM_WORLD);
    
    if(rank==0)
    {
        TIME_GET(&end);
        runtime=TIME_RUNTIME(start,end);
    }
    
    free(A);
    free(B);
    free(C);
    
    return runtime;
}

double matrix_matrix_mult_dpfp_mpi(size, threads)
{
    srand(SEED);
    int rank, num_cores;
    MPI_Comm_size(MPI_COMM_WORLD, &num_cores);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    int8_t *A;
    int8_t *B;
    int16_t *C;
    
    int i,j,k;
    
    TIME_STRUCT start,end;
    double runtime=0.0;
    
    if(rank==0)
    {
        A=malloc(sizeof(double)*(size*size));
        B=malloc(sizeof(double)*(size*size));
        C=malloc(sizeof(double)*(size*size));
        
        for(i=0; i<size*size; i++)
        {
            A[i]=(double)rand();
            B[i]=(double)rand();
            C[i]=0;
        }
    }
    else
    {
        A=malloc(sizeof(double)*(size/threads)*size);
        B=malloc(sizeof(double)*(size*size));
        C=malloc(sizeof(double)*(size*size));
    }
    
    if(rank==0)
    {
        TIME_GET(&start);
    }
    
    MPI_Scatter(A, (size/threads)*size, MPI_DOUBLE, A, (size/threads)*size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(B, size*size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    for(i=0; i<size/threads; i++)
    {
        for(k=0; k<size; k++)
        {
            for(j=0; j<size; j++)
            {
                C[i*size+j]+=A[i*size+k]*B[k*size+j];
            }
        }
    }
    
    MPI_Gather(C, (size/threads)*size, MPI_DOUBLE, C, (size/threads)*size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    if(rank==0)
    {
        TIME_GET(&end);
        runtime=TIME_RUNTIME(start,end);
    }
    
    free(A);
    free(B);
    free(C);
    
    return runtime;
}
