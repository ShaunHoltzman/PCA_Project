#include "benchmarks.h"

double keplers_spfp_mpi(int size, int threads)
{
    srand(SEED);
    int rank, num_cores;
    MPI_Comm_size(MPI_COMM_WORLD, &num_cores);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    int i,j;
    
    float *e;
    float *M;
    float *E;
    float *Eold;
    TIME_STRUCT start,end;
    TIME_STRUCT send_start,send_end;
    TIME_STRUCT recv_start,recv_end;
    double runtime=0.0;
    double sendtime=0.0;
    double recvtime=0.0;
    double calctime=0.0;
    
    float threshold=1E-8;
    
    if(rank==0)
    {
        e=malloc(sizeof(float)*size);
        M=malloc(sizeof(float)*size);
        E=malloc(sizeof(float)*size);
        Eold=malloc(sizeof(float)*size);
        
        for(i=0;i<size;i++)
        {
            e[i]=rand()/(float)RAND_MAX;
            M[i]=2*M_PI*rand()/(float)RAND_MAX;
            E[i]=M[i];
            Eold[i]=0;
        }
    }
    else
    {
        e=malloc(sizeof(float)*(size/threads));
        M=malloc(sizeof(float)*(size/threads));
        E=malloc(sizeof(float)*(size/threads));
        Eold=malloc(sizeof(float)*(size/threads));
    }
    
    if(rank==0)
    {
        TIME_GET(&send_start);
    }
    
    if(rank==0)
    {
        MPI_Scatter(e, size/threads, MPI_FLOAT, MPI_IN_PLACE, size/threads, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Scatter(M, size/threads, MPI_FLOAT, MPI_IN_PLACE, size/threads, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Scatter(E, size/threads, MPI_FLOAT, MPI_IN_PLACE, size/threads, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }
    else
    {
        MPI_Scatter(e, size/threads, MPI_FLOAT, e, size/threads, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Scatter(M, size/threads, MPI_FLOAT, M, size/threads, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Scatter(E, size/threads, MPI_FLOAT, E, size/threads, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }
    
    if(rank==0)
    {
        TIME_GET(&send_end);
        TIME_GET(&start);
    }
    
    for(i=0;i<size/threads;i++)
    {
        while(fabs(E[i]-Eold[i])>threshold)
        {
            Eold[i]=E[i];
            E[i]=E[i]-(E[i]-e[i]*sin(E[i])-M[i])/(1-e[i]*cos(E[i]));
        }
    }
    
    if(rank==0)
    {
        TIME_GET(&end);
        TIME_GET(&recv_start);
    }
    
    if(rank!=0)
    {
        MPI_Gather(E, size/threads, MPI_FLOAT, NULL, size/threads, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }
    else
    {
        MPI_Gather(MPI_IN_PLACE, size/threads, MPI_FLOAT, E, size/threads, MPI_FLOAT, 0, MPI_COMM_WORLD);
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
    
    free(Eold);
    free(E);
    free(e);
    
    return runtime;
}

double keplers_dpfp_mpi(int size, int threads)
{
    srand(SEED);
    int rank, num_cores;
    MPI_Comm_size(MPI_COMM_WORLD, &num_cores);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    int i,j;
    
    double *e;
    double *M;
    double *E;
    double *Eold;
    TIME_STRUCT start,end;
    TIME_STRUCT send_start,send_end;
    TIME_STRUCT recv_start,recv_end;
    double runtime=0.0;
    double sendtime=0.0;
    double recvtime=0.0;
    double calctime=0.0;
    
    double threshold=1E-8;
    
    if(rank==0)
    {
        e=malloc(sizeof(double)*size);
        M=malloc(sizeof(double)*size);
        E=malloc(sizeof(double)*size);
        Eold=malloc(sizeof(double)*size);
        
        for(i=0;i<size;i++)
        {
            e[i]=rand()/(double)RAND_MAX;
            M[i]=2*M_PI*rand()/(double)RAND_MAX;
            E[i]=M[i];
            Eold[i]=0;
        }
    }
    else
    {
        e=malloc(sizeof(double)*(size/threads));
        M=malloc(sizeof(double)*(size/threads));
        E=malloc(sizeof(double)*(size/threads));
        Eold=malloc(sizeof(double)*(size/threads));
    }
    
    if(rank==0)
    {
        TIME_GET(&send_start);
    }
    
    if(rank==0)
    {
        MPI_Scatter(e, size/threads, MPI_DOUBLE, MPI_IN_PLACE, size/threads, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatter(M, size/threads, MPI_DOUBLE, MPI_IN_PLACE, size/threads, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatter(E, size/threads, MPI_DOUBLE, MPI_IN_PLACE, size/threads, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    else
    {
        MPI_Scatter(e, size/threads, MPI_DOUBLE, e, size/threads, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatter(M, size/threads, MPI_DOUBLE, M, size/threads, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatter(E, size/threads, MPI_DOUBLE, E, size/threads, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    
    if(rank==0)
    {
        TIME_GET(&send_end);
        TIME_GET(&start);
    }
    
    for(i=0;i<size/threads;i++)
    {
        while(fabs(E[i]-Eold[i])>threshold)
        {
            Eold[i]=E[i];
            E[i]=E[i]-(E[i]-e[i]*sin(E[i])-M[i])/(1-e[i]*cos(E[i]));
        }
    }
    
    if(rank==0)
    {
        TIME_GET(&end);
        TIME_GET(&recv_start);
    }
    
    if(rank!=0)
    {
        MPI_Gather(E, size/threads, MPI_DOUBLE, NULL, size/threads, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    else
    {
        MPI_Gather(MPI_IN_PLACE, size/threads, MPI_DOUBLE, E, size/threads, MPI_DOUBLE, 0, MPI_COMM_WORLD);
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
    
    free(Eold);
    free(E);
    free(e);
    
    return runtime;
    
}
