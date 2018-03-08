#include "benchmarks.h"

double matrix_matrix_sub_int8_mpi(int size, int threads)
{
    int i;
    int rank, num_cores;
    srand(SEED);
    MPI_Comm_size(MPI_COMM_WORLD, &num_cores);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    int size2D = (size*size);
    
    int8_t *A;
    int8_t *B;
    int8_t *C;
    
    TIME_STRUCT start,end;
    TIME_STRUCT send_start,send_end;
    TIME_STRUCT recv_start,recv_end;
    double runtime=0.0;
    double sendtime=0.0;
    double recvtime=0.0;
    double calctime=0.0;
    
    if(rank==0)
    {
        A=malloc(sizeof(int8_t)*size2D);
        B=malloc(sizeof(int8_t)*size2D);
        C=malloc(sizeof(int8_t)*size2D);

        for(i=0; i<size2D; i++)
        {
            A[i]=(int8_t)rand();
            B[i]=(int8_t)rand();
            C[i]=0;
        }
    }
    else
    {
        A=malloc(sizeof(int8_t)*(size2D/threads));
        B=malloc(sizeof(int8_t)*(size2D/threads));
        C=malloc(sizeof(int8_t)*(size2D/threads));
    }
    
    if(rank==0)
    {
        TIME_GET(&send_start);
    }
    
    if(rank==0)
    {
        MPI_Scatter(A, (size2D/threads), MPI_CHAR, MPI_IN_PLACE, (size2D/threads), MPI_CHAR, 0, MPI_COMM_WORLD);
    	MPI_Scatter(B, (size2D/threads), MPI_CHAR, MPI_IN_PLACE, (size2D/threads), MPI_CHAR, 0, MPI_COMM_WORLD);
    }
    else
    {
    	MPI_Scatter(A, (size2D/threads), MPI_CHAR, A, (size2D/threads), MPI_CHAR, 0, MPI_COMM_WORLD);
    	MPI_Scatter(B, (size2D/threads), MPI_CHAR, B, (size2D/threads), MPI_CHAR, 0, MPI_COMM_WORLD);
    }
    
    if(rank==0)
    {
        TIME_GET(&send_end);
        TIME_GET(&start);
    }
    
    for(i=0; i<(size2D/threads); i++)
    {
        C[i]=A[i]-B[i];
    }
    
    if(rank==0)
    {
        TIME_GET(&end);
        TIME_GET(&recv_start);
    }
    
    if(rank != 0)
    {
    	MPI_Gather(C, (size2D/threads), MPI_CHAR, NULL, (size2D/threads), MPI_CHAR, 0, MPI_COMM_WORLD);
    }
    else
    {
    	MPI_Gather(MPI_IN_PLACE, (size2D/threads), MPI_CHAR, C, (size2D/threads), MPI_CHAR, 0, MPI_COMM_WORLD);
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
    
    free(A);
    free(B);
    free(C);
    
    return runtime;
}

double matrix_matrix_sub_int16_mpi(int size, int threads)
{
    int i;
    int rank, num_cores;
    srand(SEED);
    MPI_Comm_size(MPI_COMM_WORLD, &num_cores);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    int size2D = (size*size);
    
    int16_t *A;
    int16_t *B;
    int16_t *C;
    
    TIME_STRUCT start,end;
    TIME_STRUCT send_start,send_end;
    TIME_STRUCT recv_start,recv_end;
    double runtime=0.0;
    double sendtime=0.0;
    double recvtime=0.0;
    double calctime=0.0;
    
    if(rank==0)
    {
        A=malloc(sizeof(int16_t)*size2D);
        B=malloc(sizeof(int16_t)*size2D);
        C=malloc(sizeof(int16_t)*size2D);

        for(i=0; i<size2D; i++)
        {
            A[i]=(int16_t)rand();
            B[i]=(int16_t)rand();
            C[i]=0;
        }
    }
    else
    {
        A=malloc(sizeof(int16_t)*(size2D/threads));
        B=malloc(sizeof(int16_t)*(size2D/threads));
        C=malloc(sizeof(int16_t)*(size2D/threads));
    }
    
    if(rank==0)
    {
        TIME_GET(&send_start);
    }
    
    if(rank==0)
    {
    	MPI_Scatter(A, (size2D/threads), MPI_SHORT, MPI_IN_PLACE, (size2D/threads), MPI_SHORT, 0, MPI_COMM_WORLD);
    	MPI_Scatter(B, (size2D/threads), MPI_SHORT, MPI_IN_PLACE, (size2D/threads), MPI_SHORT, 0, MPI_COMM_WORLD);
    }
    else
    {
    	MPI_Scatter(A, (size2D/threads), MPI_SHORT, A, (size2D/threads), MPI_SHORT, 0, MPI_COMM_WORLD);
    	MPI_Scatter(B, (size2D/threads), MPI_SHORT, B, (size2D/threads), MPI_SHORT, 0, MPI_COMM_WORLD);
    }
    
    if(rank==0)
    {
        TIME_GET(&send_end);
        TIME_GET(&start);
    }
    
    for(i=0; i<(size2D/threads); i++)
    {
        C[i]=A[i]-B[i];
    }
    
    if(rank==0)
    {
        TIME_GET(&end);
        TIME_GET(&recv_start);
    }
    
    if(rank!=0)
    {
    	MPI_Gather(C, (size2D/threads), MPI_SHORT, NULL, (size2D/threads), MPI_SHORT, 0, MPI_COMM_WORLD);
    }
    else
    {
    	MPI_Gather(MPI_IN_PLACE, (size2D/threads), MPI_SHORT, C, (size2D/threads), MPI_SHORT, 0, MPI_COMM_WORLD);
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
    
    free(A);
    free(B);
    free(C);
    
    return runtime;
}

double matrix_matrix_sub_int32_mpi(int size, int threads)
{

    int i;
    int rank, num_cores;
    srand(SEED);
    MPI_Comm_size(MPI_COMM_WORLD, &num_cores);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    int size2D = (size*size);
    
    int32_t *A;
    int32_t *B;
    int32_t *C;
    
    TIME_STRUCT start,end;
    TIME_STRUCT send_start,send_end;
    TIME_STRUCT recv_start,recv_end;
    double runtime=0.0;
    double sendtime=0.0;
    double recvtime=0.0;
    double calctime=0.0;
    
    if(rank==0)
    {
        A=malloc(sizeof(int32_t)*size2D);
        B=malloc(sizeof(int32_t)*size2D);
        C=malloc(sizeof(int32_t)*size2D);

        for(i=0; i<size2D; i++)
        {
            A[i]=(int32_t)rand();
            B[i]=(int32_t)rand();
            C[i]=0;
        }
    }
    else
    {
        A=malloc(sizeof(int32_t)*(size2D/threads));
        B=malloc(sizeof(int32_t)*(size2D/threads));
        C=malloc(sizeof(int32_t)*(size2D/threads));
    }
    
    if(rank==0)
    {
        TIME_GET(&send_start);
    }
    
    if(rank==0)
    {
    	MPI_Scatter(A, (size2D/threads), MPI_INT, MPI_IN_PLACE, (size2D/threads), MPI_INT, 0, MPI_COMM_WORLD);
    	MPI_Scatter(B, (size2D/threads), MPI_INT, MPI_IN_PLACE, (size2D/threads), MPI_INT, 0, MPI_COMM_WORLD);
    }
    else
    {
    	MPI_Scatter(A, (size2D/threads), MPI_INT, A, (size2D/threads), MPI_INT, 0, MPI_COMM_WORLD);
    	MPI_Scatter(B, (size2D/threads), MPI_INT, B, (size2D/threads), MPI_INT, 0, MPI_COMM_WORLD);
    }
    
    if(rank==0)
    {
        TIME_GET(&send_end);
        TIME_GET(&start);
    }
    
    for(i=0; i<(size2D/threads); i++)
    {
        C[i]=A[i]-B[i];
    }
    
    if(rank==0)
    {
        TIME_GET(&end);
        TIME_GET(&recv_start);
    }
    
    if(rank!=0)
    {
    	MPI_Gather(C, (size2D/threads), MPI_INT, NULL, (size2D/threads), MPI_INT, 0, MPI_COMM_WORLD);
    }
    else
    {
    	MPI_Gather(MPI_IN_PLACE, (size2D/threads), MPI_INT, C, (size2D/threads), MPI_INT, 0, MPI_COMM_WORLD);
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
    
    free(A);
    free(B);
    free(C);
    
    return runtime;


}

double matrix_matrix_sub_spfp_mpi(int size, int threads)
{
    int i;
    int rank, num_cores;
    srand(SEED);
    MPI_Comm_size(MPI_COMM_WORLD, &num_cores);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    int size2D = (size*size);
    
    float *A;
    float *B;
    float *C;
    
    TIME_STRUCT start,end;
    TIME_STRUCT send_start,send_end;
    TIME_STRUCT recv_start,recv_end;
    double runtime=0.0;
    double sendtime=0.0;
    double recvtime=0.0;
    double calctime=0.0;
    
    if(rank==0)
    {
        A=malloc(sizeof(float)*size2D);
        B=malloc(sizeof(float)*size2D);
        C=malloc(sizeof(float)*size2D);

        for(i=0; i<size2D; i++)
        {
            A[i]=(float)rand();
            B[i]=(float)rand();
            C[i]=0;
        }
    }
    else
    {
        A=malloc(sizeof(float)*(size2D/threads));
        B=malloc(sizeof(float)*(size2D/threads));
        C=malloc(sizeof(float)*(size2D/threads));
    }
    
    if(rank==0)
    {
        TIME_GET(&send_start);
    }
    
    if(rank==0)
    {
    	MPI_Scatter(A, (size2D/threads), MPI_FLOAT, MPI_IN_PLACE, (size2D/threads), MPI_FLOAT, 0, MPI_COMM_WORLD);
    	MPI_Scatter(B, (size2D/threads), MPI_FLOAT, MPI_IN_PLACE, (size2D/threads), MPI_FLOAT, 0, MPI_COMM_WORLD);
    }
    else
    {
    	MPI_Scatter(A, (size2D/threads), MPI_FLOAT, A, (size2D/threads), MPI_FLOAT, 0, MPI_COMM_WORLD);
    	MPI_Scatter(B, (size2D/threads), MPI_FLOAT, B, (size2D/threads), MPI_FLOAT, 0, MPI_COMM_WORLD);
    }
    
    if(rank==0)
    {
        TIME_GET(&send_end);
        TIME_GET(&start);
    }
    
    for(i=0; i<(size2D/threads); i++)
    {
        C[i]=A[i]-B[i];
    }
    
    if(rank==0)
    {
        TIME_GET(&end);
        TIME_GET(&recv_start);
    }
    
    if(rank!=0)
    {
    	MPI_Gather(C, (size2D/threads), MPI_FLOAT, NULL, (size2D/threads), MPI_FLOAT, 0, MPI_COMM_WORLD);
    }
    else
    {
    	MPI_Gather(MPI_IN_PLACE, (size2D/threads), MPI_FLOAT, C, (size2D/threads), MPI_FLOAT, 0, MPI_COMM_WORLD);
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
    
    free(A);
    free(B);
    free(C);
    
    return runtime;

}

double matrix_matrix_sub_dpfp_mpi(int size, int threads)
{
    int i;
    int rank, num_cores;
    srand(SEED);
    MPI_Comm_size(MPI_COMM_WORLD, &num_cores);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    int size2D = (size*size);
    
    double *A;
    double *B;
    double *C;
    
    TIME_STRUCT start,end;
    TIME_STRUCT send_start,send_end;
    TIME_STRUCT recv_start,recv_end;
    double runtime=0.0;
    double sendtime=0.0;
    double recvtime=0.0;
    double calctime=0.0;
    
    if(rank==0)
    {
        A=malloc(sizeof(double)*size2D);
        B=malloc(sizeof(double)*size2D);
        C=malloc(sizeof(double)*size2D);

        for(i=0; i<size2D; i++)
        {
            A[i]=(double)rand();
            B[i]=(double)rand();
            C[i]=0;
        }
    }
    else
    {
        A=malloc(sizeof(double)*(size2D/threads));
        B=malloc(sizeof(double)*(size2D/threads));
        C=malloc(sizeof(double)*(size2D/threads));
    }
    
    if(rank==0)
    {
        TIME_GET(&send_start);
    }
    
    
    if(rank==0)
    {
    	MPI_Scatter(A, (size2D/threads), MPI_DOUBLE, MPI_IN_PLACE, (size2D/threads), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    	MPI_Scatter(B, (size2D/threads), MPI_DOUBLE, MPI_IN_PLACE, (size2D/threads), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    else
    {
    	MPI_Scatter(A, (size2D/threads), MPI_DOUBLE, A, (size2D/threads), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    	MPI_Scatter(B, (size2D/threads), MPI_DOUBLE, B, (size2D/threads), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    
    if(rank==0)
    {
        TIME_GET(&send_end);
        TIME_GET(&start);
    }
    
    for(i=0; i<(size2D/threads); i++)
    {
        C[i]=A[i]-B[i];
    }
    
    if(rank==0)
    {
        TIME_GET(&end);
        TIME_GET(&recv_start);
    }
    
    if(rank!=0)
    {
    	MPI_Gather(C, (size2D/threads), MPI_DOUBLE, NULL, (size2D/threads), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    else
    {
    	MPI_Gather(MPI_IN_PLACE, (size2D/threads), MPI_DOUBLE, C, (size2D/threads), MPI_DOUBLE, 0, MPI_COMM_WORLD);
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
    
    free(A);
    free(B);
    free(C);
    
    return runtime;

}
