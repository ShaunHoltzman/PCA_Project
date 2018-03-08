#include "benchmarks.h"

double cw_spfp_mpi(int size, int threads)
{
    srand(SEED);
    int rank, num_cores;
    MPI_Comm_size(MPI_COMM_WORLD, &num_cores);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    int i;
    
    float *x0;
	float *y0;
	float *z0;
	float *xhat0;
	float *yhat0;
	float *zhat0;
	float *t;
	float *x;
	float *y;
	float *z;
	float *xhat;
	float *yhat;
	float *zhat;
    
    float n=0.0011;
	float nr=1/n;
	float n3=3*n;
	float n6=6*n;
	float nt=0;
	float sinnt=0;
	float cosnt=0;
    
    TIME_STRUCT start,end;
    TIME_STRUCT send_start,send_end;
    TIME_STRUCT recv_start,recv_end;
    float runtime=0.0;
    float sendtime=0.0;
    float recvtime=0.0;
    float calctime=0.0;
    
    if(rank==0)
    {
        x0=malloc(sizeof(float)*size);
        y0=malloc(sizeof(float)*size);
        z0=malloc(sizeof(float)*size);
        xhat0=malloc(sizeof(float)*size);
        yhat0=malloc(sizeof(float)*size);
        zhat0=malloc(sizeof(float)*size);
        t=malloc(sizeof(float)*size);
        x=malloc(sizeof(float)*size);
        y=malloc(sizeof(float)*size);
        z=malloc(sizeof(float)*size);
        xhat=malloc(sizeof(float)*size);
        yhat=malloc(sizeof(float)*size);
        zhat=malloc(sizeof(float)*size);
        
        for(i=0;i<size;i++)
        {
            x0[i]=rand()/(float)RAND_MAX;
            y0[i]=rand()/(float)RAND_MAX;
            z0[i]=rand()/(float)RAND_MAX;
            xhat0[i]=rand()/(float)RAND_MAX;
            yhat0[i]=rand()/(float)RAND_MAX;
            zhat0[i]=rand()/(float)RAND_MAX;
            t[i]=rand()/(float)RAND_MAX;
            x[i]=0;
            y[i]=0;
            z[i]=0;
            xhat[i]=0;
            yhat[i]=0;
            zhat[i]=0;
        }
    }
    else
    {
        x0=malloc(sizeof(float)*size/threads);
        y0=malloc(sizeof(float)*size/threads);
        z0=malloc(sizeof(float)*size/threads);
        xhat0=malloc(sizeof(float)*size/threads);
        yhat0=malloc(sizeof(float)*size/threads);
        zhat0=malloc(sizeof(float)*size/threads);
        t=malloc(sizeof(float)*size/threads);
        x=malloc(sizeof(float)*size/threads);
        y=malloc(sizeof(float)*size/threads);
        z=malloc(sizeof(float)*size/threads);
        xhat=malloc(sizeof(float)*size/threads);
        yhat=malloc(sizeof(float)*size/threads);
        zhat=malloc(sizeof(float)*size/threads);
    }
    
    if(rank==0)
    {
        TIME_GET(&send_start);
    }
    
    if(rank==0)
    {
        MPI_Scatter(x0, size/threads, MPI_FLOAT, MPI_IN_PLACE, size/threads, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Scatter(y0, size/threads, MPI_FLOAT, MPI_IN_PLACE, size/threads, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Scatter(z0, size/threads, MPI_FLOAT, MPI_IN_PLACE, size/threads, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Scatter(xhat0, size/threads, MPI_FLOAT, MPI_IN_PLACE, size/threads, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Scatter(yhat0, size/threads, MPI_FLOAT, MPI_IN_PLACE, size/threads, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Scatter(zhat0, size/threads, MPI_FLOAT, MPI_IN_PLACE, size/threads, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Scatter(t, size/threads, MPI_FLOAT, MPI_IN_PLACE, size/threads, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }
    else
    {
        MPI_Scatter(x0, size/threads, MPI_FLOAT, x0, size/threads, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Scatter(y0, size/threads, MPI_FLOAT, y0, size/threads, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Scatter(z0, size/threads, MPI_FLOAT, z0, size/threads, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Scatter(xhat0, size/threads, MPI_FLOAT, xhat0, size/threads, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Scatter(yhat0, size/threads, MPI_FLOAT, yhat0, size/threads, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Scatter(zhat0, size/threads, MPI_FLOAT, zhat0, size/threads, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Scatter(t, size/threads, MPI_FLOAT, t, size/threads, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }
    
    if(rank==0)
    {
        TIME_GET(&send_end);
        TIME_GET(&start);
    }
    
    for(i=0;i<size/threads;i++)
    {
        nt=n*t[i];
        sinnt=sin(nt);
        cosnt=cos(nt);
        x[i]=x0[i]*((4-3*cosnt)+sinnt*nr+(2-2*cosnt)*nr);
        y[i]=y0[i]*(6*(sinnt-nt)+1-(2-2*cosnt)*nr+(4*sinnt-3*nt)*nr);
        z[i]=z0[i]*(cosnt+sinnt*nr);
        xhat[i]=xhat0[i]*(n3*sinnt+cosnt+2*sinnt);
        yhat[i]=yhat0[i]*(-n6*(1-cosnt)-2*sinnt+(4*cosnt-3));
        zhat[i]=zhat0[i]*(-n*sinnt+cosnt);
    }
    
    if(rank==0)
    {
        TIME_GET(&end);
        TIME_GET(&recv_start);
    }
    
    if(rank!=0)
    {
        MPI_Gather(x, size/threads, MPI_FLOAT, NULL, size/threads, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Gather(y, size/threads, MPI_FLOAT, NULL, size/threads, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Gather(z, size/threads, MPI_FLOAT, NULL, size/threads, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Gather(xhat, size/threads, MPI_FLOAT, NULL, size/threads, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Gather(yhat, size/threads, MPI_FLOAT, NULL, size/threads, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Gather(zhat, size/threads, MPI_FLOAT, NULL, size/threads, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }
    else
    {
        MPI_Gather(MPI_IN_PLACE, size/threads, MPI_FLOAT, x, size/threads, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Gather(MPI_IN_PLACE, size/threads, MPI_FLOAT, y, size/threads, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Gather(MPI_IN_PLACE, size/threads, MPI_FLOAT, z, size/threads, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Gather(MPI_IN_PLACE, size/threads, MPI_FLOAT, xhat, size/threads, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Gather(MPI_IN_PLACE, size/threads, MPI_FLOAT, yhat, size/threads, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Gather(MPI_IN_PLACE, size/threads, MPI_FLOAT, zhat, size/threads, MPI_FLOAT, 0, MPI_COMM_WORLD);
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
    
    free(x0);
	free(y0);
	free(z0);
	free(xhat0);
	free(yhat0);
	free(zhat0);
	free(x);
	free(y);
	free(z);
	free(xhat);
	free(yhat);
	free(zhat);
	free(t);
    
    return runtime;
    
}

double cw_dpfp_mpi(int size, int threads)
{
    srand(SEED);
    int rank, num_cores;
    MPI_Comm_size(MPI_COMM_WORLD, &num_cores);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    int i;
    
    double *x0;
	double *y0;
	double *z0;
	double *xhat0;
	double *yhat0;
	double *zhat0;
	double *t;
	double *x;
	double *y;
	double *z;
	double *xhat;
	double *yhat;
	double *zhat;
    
    double n=0.0011;
	double nr=1/n;
	double n3=3*n;
	double n6=6*n;
	double nt=0;
	double sinnt=0;
	double cosnt=0;
    
    TIME_STRUCT start,end;
    TIME_STRUCT send_start,send_end;
    TIME_STRUCT recv_start,recv_end;
    double runtime=0.0;
    double sendtime=0.0;
    double recvtime=0.0;
    double calctime=0.0;
    
    if(rank==0)
    {
        x0=malloc(sizeof(double)*size);
        y0=malloc(sizeof(double)*size);
        z0=malloc(sizeof(double)*size);
        xhat0=malloc(sizeof(double)*size);
        yhat0=malloc(sizeof(double)*size);
        zhat0=malloc(sizeof(double)*size);
        t=malloc(sizeof(double)*size);
        x=malloc(sizeof(double)*size);
        y=malloc(sizeof(double)*size);
        z=malloc(sizeof(double)*size);
        xhat=malloc(sizeof(double)*size);
        yhat=malloc(sizeof(double)*size);
        zhat=malloc(sizeof(double)*size);
        
        for(i=0;i<size;i++)
        {
            x0[i]=rand()/(double)RAND_MAX;
            y0[i]=rand()/(double)RAND_MAX;
            z0[i]=rand()/(double)RAND_MAX;
            xhat0[i]=rand()/(double)RAND_MAX;
            yhat0[i]=rand()/(double)RAND_MAX;
            zhat0[i]=rand()/(double)RAND_MAX;
            t[i]=rand()/(double)RAND_MAX;
            x[i]=0;
            y[i]=0;
            z[i]=0;
            xhat[i]=0;
            yhat[i]=0;
            zhat[i]=0;
        }
    }
    else
    {
        x0=malloc(sizeof(double)*size/threads);
        y0=malloc(sizeof(double)*size/threads);
        z0=malloc(sizeof(double)*size/threads);
        xhat0=malloc(sizeof(double)*size/threads);
        yhat0=malloc(sizeof(double)*size/threads);
        zhat0=malloc(sizeof(double)*size/threads);
        t=malloc(sizeof(double)*size/threads);
        x=malloc(sizeof(double)*size/threads);
        y=malloc(sizeof(double)*size/threads);
        z=malloc(sizeof(double)*size/threads);
        xhat=malloc(sizeof(double)*size/threads);
        yhat=malloc(sizeof(double)*size/threads);
        zhat=malloc(sizeof(double)*size/threads);
    }
    
    if(rank==0)
    {
        TIME_GET(&send_start);
    }
    
    if(rank==0)
    {
        MPI_Scatter(x0, size/threads, MPI_DOUBLE, MPI_IN_PLACE, size/threads, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatter(y0, size/threads, MPI_DOUBLE, MPI_IN_PLACE, size/threads, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatter(z0, size/threads, MPI_DOUBLE, MPI_IN_PLACE, size/threads, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatter(xhat0, size/threads, MPI_DOUBLE, MPI_IN_PLACE, size/threads, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatter(yhat0, size/threads, MPI_DOUBLE, MPI_IN_PLACE, size/threads, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatter(zhat0, size/threads, MPI_DOUBLE, MPI_IN_PLACE, size/threads, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatter(t, size/threads, MPI_DOUBLE, MPI_IN_PLACE, size/threads, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    else
    {
        MPI_Scatter(x0, size/threads, MPI_DOUBLE, x0, size/threads, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatter(y0, size/threads, MPI_DOUBLE, y0, size/threads, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatter(z0, size/threads, MPI_DOUBLE, z0, size/threads, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatter(xhat0, size/threads, MPI_DOUBLE, xhat0, size/threads, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatter(yhat0, size/threads, MPI_DOUBLE, yhat0, size/threads, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatter(zhat0, size/threads, MPI_DOUBLE, zhat0, size/threads, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatter(t, size/threads, MPI_DOUBLE, t, size/threads, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    
    if(rank==0)
    {
        TIME_GET(&send_end);
        TIME_GET(&start);
    }
    
    for(i=0;i<size/threads;i++)
    {
        nt=n*t[i];
        sinnt=sin(nt);
        cosnt=cos(nt);
        x[i]=x0[i]*((4-3*cosnt)+sinnt*nr+(2-2*cosnt)*nr);
        y[i]=y0[i]*(6*(sinnt-nt)+1-(2-2*cosnt)*nr+(4*sinnt-3*nt)*nr);
        z[i]=z0[i]*(cosnt+sinnt*nr);
        xhat[i]=xhat0[i]*(n3*sinnt+cosnt+2*sinnt);
        yhat[i]=yhat0[i]*(-n6*(1-cosnt)-2*sinnt+(4*cosnt-3));
        zhat[i]=zhat0[i]*(-n*sinnt+cosnt);
    }
    
    if(rank==0)
    {
        TIME_GET(&end);
        TIME_GET(&recv_start);
    }
    
    if(rank!=0)
    {
        MPI_Gather(x, size/threads, MPI_DOUBLE, NULL, size/threads, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(y, size/threads, MPI_DOUBLE, NULL, size/threads, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(z, size/threads, MPI_DOUBLE, NULL, size/threads, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(xhat, size/threads, MPI_DOUBLE, NULL, size/threads, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(yhat, size/threads, MPI_DOUBLE, NULL, size/threads, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(zhat, size/threads, MPI_DOUBLE, NULL, size/threads, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    else
    {
        MPI_Gather(MPI_IN_PLACE, size/threads, MPI_DOUBLE, x, size/threads, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(MPI_IN_PLACE, size/threads, MPI_DOUBLE, y, size/threads, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(MPI_IN_PLACE, size/threads, MPI_DOUBLE, z, size/threads, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(MPI_IN_PLACE, size/threads, MPI_DOUBLE, xhat, size/threads, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(MPI_IN_PLACE, size/threads, MPI_DOUBLE, yhat, size/threads, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(MPI_IN_PLACE, size/threads, MPI_DOUBLE, zhat, size/threads, MPI_DOUBLE, 0, MPI_COMM_WORLD);
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
    
    free(x0);
	free(y0);
	free(z0);
	free(xhat0);
	free(yhat0);
	free(zhat0);
	free(x);
	free(y);
	free(z);
	free(xhat);
	free(yhat);
	free(zhat);
	free(t);
    
    return runtime;
}
