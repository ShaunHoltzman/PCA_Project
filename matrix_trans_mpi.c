#include "benchmarks.h"

double matrix_matrix_trans_int8_mpi(int size, int threads)
{
    int i,j;
    int rank, num_cores;
    srand(SEED);
    MPI_Comm_size(MPI_COMM_WORLD, &num_cores);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    int8_t *A;
    int8_t *B;
    
    TIME_STRUCT start,end;
    TIME_STRUCT send_start,send_end;
    TIME_STRUCT recv_start,recv_end;
    double runtime=0.0;
    double sendtime=0.0;
    double recvtime=0.0;
    double calctime=0.0;
    
    if(rank==0)
    {
        A=malloc(sizeof(int8_t)*(size*size));
        B=malloc(sizeof(int8_t)*(size*size));
        
        for(i=0; i<(size*size); i++)
        {
            A[i]=(int8_t)rand();
            B[i]=0;
        }
    }
    else
    {
        A=malloc(sizeof(int8_t)*(size*size));
        B=malloc(sizeof(int8_t)*( (size/threads)*size) );
    }
    
    if(rank==0)
    {
        TIME_GET(&send_start);
    }
    
    MPI_Bcast(A, size*size, MPI_CHAR, 0, MPI_COMM_WORLD);
    
    if(rank==0)
    {
        TIME_GET(&send_end);
        TIME_GET(&start);
    }
    
    for(i=0; i<size; i++)
    {
        for(j=0; j<size/threads; j++)
        {
            B[j*size+i]=A[i*size+j+(rank*(size/threads))];
        }
    }
    
    if(rank==0)
    {
        TIME_GET(&end);
        TIME_GET(&recv_start);
    }
    
    if(rank !=0)
    {
        MPI_Gather(B, (size/threads)*size, MPI_CHAR, NULL, (size/threads)*size, MPI_CHAR, 0, MPI_COMM_WORLD);
    }
    else
    {
        MPI_Gather(MPI_IN_PLACE, (size/threads)*size, MPI_CHAR, B, (size/threads)*size, MPI_CHAR, 0, MPI_COMM_WORLD);
    }
    
    if(rank==0)
    {
        TIME_GET(&recv_end);
        sendtime=TIME_RUNTIME(send_start,send_end);
        recvtime=TIME_RUNTIME(recv_start,recv_end);
        calctime=TIME_RUNTIME(start,end);
        printf("Send time = %f\n", sendtime);
        printf("Calc time = %f\n", calctime);
        printf("Gather time = %f\n", recvtime);
        runtime=sendtime+calctime+recvtime;
    }
    
    free(B);
    free(A);
    
    return runtime;
}

double matrix_matrix_trans_int16_mpi(int size, int threads)
{
    int i,j;
    int rank, num_cores;
    srand(SEED);
    MPI_Comm_size(MPI_COMM_WORLD, &num_cores);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    int16_t *A;
    int16_t *B;
    
    TIME_STRUCT start,end;
    TIME_STRUCT send_start,send_end;
    TIME_STRUCT recv_start,recv_end;
    double runtime=0.0;
    double sendtime=0.0;
    double recvtime=0.0;
    double calctime=0.0;
    
    if(rank==0)
    {
        A=malloc(sizeof(int16_t)*(size*size));
        B=malloc(sizeof(int16_t)*(size*size));
        
        for(i=0; i<(size*size); i++)
        {
            A[i]=(int16_t)rand();
            B[i]=0;
        }
    }
    else
    {
        A=malloc(sizeof(int16_t)*(size*size));
        B=malloc(sizeof(int16_t)*( (size/threads)*size) );
    }
    
    if(rank==0)
    {
        TIME_GET(&send_start);
    }
    
    MPI_Bcast(A, size*size, MPI_SHORT, 0, MPI_COMM_WORLD);
    
    if(rank==0)
    {
        TIME_GET(&send_end);
        TIME_GET(&start);
    }
    
    for(i=0; i<size; i++)
    {
        for(j=0; j<size/threads; j++)
        {
            B[j*size+i]=A[i*size+j+(rank*(size/threads))];
        }
    }
    
    if(rank==0)
    {
        TIME_GET(&end);
        TIME_GET(&recv_start);
    }
    
    if(rank !=0)
    {
        MPI_Gather(B, (size/threads)*size, MPI_SHORT, NULL, (size/threads)*size, MPI_SHORT, 0, MPI_COMM_WORLD);
    }
    else
    {
        MPI_Gather(MPI_IN_PLACE, (size/threads)*size, MPI_SHORT, B, (size/threads)*size, MPI_SHORT, 0, MPI_COMM_WORLD);
    }
    
    if(rank==0)
    {
        TIME_GET(&recv_end);
        sendtime=TIME_RUNTIME(send_start,send_end);
        recvtime=TIME_RUNTIME(recv_start,recv_end);
        calctime=TIME_RUNTIME(start,end);
        printf("Send time = %f\n", sendtime);
        printf("Calc time = %f\n", calctime);
        printf("Gather time = %f\n", recvtime);
        runtime=sendtime+calctime+recvtime;
    }
    
    free(B);
    free(A);
    
    return runtime;
}

double matrix_matrix_trans_int32_mpi(int size, int threads)
{
    int i,j;
    int rank, num_cores;
    srand(SEED);
    MPI_Comm_size(MPI_COMM_WORLD, &num_cores);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    int32_t *A;
    int32_t *B;
    
    TIME_STRUCT start,end;
    TIME_STRUCT send_start,send_end;
    TIME_STRUCT recv_start,recv_end;
    double runtime=0.0;
    double sendtime=0.0;
    double recvtime=0.0;
    double calctime=0.0;
    
    if(rank==0)
    {
        A=malloc(sizeof(int32_t)*(size*size));
        B=malloc(sizeof(int32_t)*(size*size));
        
        for(i=0; i<(size*size); i++)
        {
            A[i]=(int32_t)rand();
            B[i]=0;
        }
    }
    else
    {
        A=malloc(sizeof(int32_t)*(size*size));
        B=malloc(sizeof(int32_t)*( (size/threads)*size) );
    }
    
    if(rank==0)
    {
        TIME_GET(&send_start);
    }
    
    MPI_Bcast(A, size*size, MPI_INT, 0, MPI_COMM_WORLD);
    
    if(rank==0)
    {
        TIME_GET(&send_end);
        TIME_GET(&start);
    }
    
    for(i=0; i<size; i++)
    {
        for(j=0; j<size/threads; j++)
        {
            B[j*size+i]=A[i*size+j+(rank*(size/threads))];
        }
    }
    
    if(rank==0)
    {
        TIME_GET(&end);
        TIME_GET(&recv_start);
    }
    
    if(rank !=0)
    {
        MPI_Gather(B, (size/threads)*size, MPI_INT, NULL, (size/threads)*size, MPI_INT, 0, MPI_COMM_WORLD);
    }
    else
    {
        MPI_Gather(MPI_IN_PLACE, (size/threads)*size, MPI_INT, B, (size/threads)*size, MPI_INT, 0, MPI_COMM_WORLD);
    }
    
    if(rank==0)
    {
        TIME_GET(&recv_end);
        sendtime=TIME_RUNTIME(send_start,send_end);
        recvtime=TIME_RUNTIME(recv_start,recv_end);
        calctime=TIME_RUNTIME(start,end);
        printf("Send time = %f\n", sendtime);
        printf("Calc time = %f\n", calctime);
        printf("Gather time = %f\n", recvtime);
        runtime=sendtime+calctime+recvtime;
    }
    
    free(B);
    free(A);
    
    return runtime;
}

double matrix_matrix_trans_spfp_mpi(int size, int threads)
{
    int i,j;
    int rank, num_cores;
    srand(SEED);
    MPI_Comm_size(MPI_COMM_WORLD, &num_cores);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    float *A;
    float *B;
    
    TIME_STRUCT start,end;
    TIME_STRUCT send_start,send_end;
    TIME_STRUCT recv_start,recv_end;
    double runtime=0.0;
    double sendtime=0.0;
    double recvtime=0.0;
    double calctime=0.0;
    
    if(rank==0)
    {
        A=malloc(sizeof(float)*(size*size));
        B=malloc(sizeof(float)*(size*size));
        
        for(i=0; i<(size*size); i++)
        {
            A[i]=(float)rand();
            B[i]=0;
        }
    }
    else
    {
        A=malloc(sizeof(float)*(size*size));
        B=malloc(sizeof(float)*( (size/threads)*size) );
    }
    
    if(rank==0)
    {
        TIME_GET(&send_start);
    }
    
    MPI_Bcast(A, size*size, MPI_FLOAT, 0, MPI_COMM_WORLD);
    
    if(rank==0)
    {
        TIME_GET(&send_end);
        TIME_GET(&start);
    }
    
    for(i=0; i<size; i++)
    {
        for(j=0; j<size/threads; j++)
        {
            B[j*size+i]=A[i*size+j+(rank*(size/threads))];
        }
    }
    
    if(rank==0)
    {
        TIME_GET(&end);
        TIME_GET(&recv_start);
    }
    
    if(rank !=0)
    {
        MPI_Gather(B, (size/threads)*size, MPI_FLOAT, NULL, (size/threads)*size, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }
    else
    {
        MPI_Gather(MPI_IN_PLACE, (size/threads)*size, MPI_FLOAT, B, (size/threads)*size, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }
    
    if(rank==0)
    {
        TIME_GET(&recv_end);
        sendtime=TIME_RUNTIME(send_start,send_end);
        recvtime=TIME_RUNTIME(recv_start,recv_end);
        calctime=TIME_RUNTIME(start,end);
        printf("Send time = %f\n", sendtime);
        printf("Calc time = %f\n", calctime);
        printf("Gather time = %f\n", recvtime);
        runtime=sendtime+calctime+recvtime;
    }
    
    free(B);
    free(A);
    
    return runtime;
}

double matrix_matrix_trans_dpfp_mpi(int size, int threads)
{
    int i,j;
    int rank, num_cores;
    srand(SEED);
    MPI_Comm_size(MPI_COMM_WORLD, &num_cores);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    double *A;
    double *B;
    
    TIME_STRUCT start,end;
    TIME_STRUCT send_start,send_end;
    TIME_STRUCT recv_start,recv_end;
    double runtime=0.0;
    double sendtime=0.0;
    double recvtime=0.0;
    double calctime=0.0;
    
    if(rank==0)
    {
        A=malloc(sizeof(double)*(size*size));
        B=malloc(sizeof(double)*(size*size));
        
        for(i=0; i<(size*size); i++)
        {
            A[i]=(double)rand();
            B[i]=0;
        }
    }
    else
    {
        A=malloc(sizeof(double)*(size*size));
        B=malloc(sizeof(double)*( (size/threads)*size) );
    }
    
    if(rank==0)
    {
        TIME_GET(&send_start);
    }
    
    MPI_Bcast(A, size*size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    if(rank==0)
    {
        TIME_GET(&send_end);
        TIME_GET(&start);
    }
    
    for(i=0; i<size; i++)
    {
        for(j=0; j<size/threads; j++)
        {
            B[j*size+i]=A[i*size+j+(rank*(size/threads))];
        }
    }
    
    if(rank==0)
    {
        TIME_GET(&end);
        TIME_GET(&recv_start);
    }
    
    if(rank !=0)
    {
        MPI_Gather(B, (size/threads)*size, MPI_DOUBLE, NULL, (size/threads)*size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    else
    {
        MPI_Gather(MPI_IN_PLACE, (size/threads)*size, MPI_DOUBLE, B, (size/threads)*size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    
    if(rank==0)
    {
        TIME_GET(&recv_end);
        sendtime=TIME_RUNTIME(send_start,send_end);
        recvtime=TIME_RUNTIME(recv_start,recv_end);
        calctime=TIME_RUNTIME(start,end);
        printf("Send time = %f\n", sendtime);
        printf("Calc time = %f\n", calctime);
        printf("Gather time = %f\n", recvtime);
        runtime=sendtime+calctime+recvtime;
    }
    
    free(B);
    free(A);
    
    return runtime;
}
