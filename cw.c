#include "benchmarks.h"

double cw_spfp(int size, int threads)
{
    int i;
    srand(SEED);
    TIME_STRUCT start,end;
    
    float *x0=malloc(sizeof(float)*size);
	float *y0=malloc(sizeof(float)*size);
	float *z0=malloc(sizeof(float)*size);
	float *xhat0=malloc(sizeof(float)*size);
	float *yhat0=malloc(sizeof(float)*size);
	float *zhat0=malloc(sizeof(float)*size);
	float *t=malloc(sizeof(float)*size);
	float *x=malloc(sizeof(float)*size);
	float *y=malloc(sizeof(float)*size);
	float *z=malloc(sizeof(float)*size);
	float *xhat=malloc(sizeof(float)*size);
	float *yhat=malloc(sizeof(float)*size);
	float *zhat=malloc(sizeof(float)*size);
    
    float n=0.0011;
	float nr=1/n;
	float n3=3*n;
	float n6=6*n;
	float nt=0;
	float sinnt=0;
	float cosnt=0;
    
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
    
    if(threads==1)
    {
        TIME_GET(&start);
        
        for(i=0;i<size;i++)
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
        
        TIME_GET(&end);
    }
    else
    {
        TIME_GET(&start);
        
        #pragma omp parallel for num_threads(threads) private(i)
        for(i=0;i<size;i++)
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
        
        TIME_GET(&end);
        
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
    
    return TIME_RUNTIME(start,end);
}

double cw_dpfp(int size, int threads)
{
    int i;
    srand(SEED);
    TIME_STRUCT start,end;
    
    double *x0=malloc(sizeof(double)*size);
	double *y0=malloc(sizeof(double)*size);
	double *z0=malloc(sizeof(double)*size);
	double *xhat0=malloc(sizeof(double)*size);
	double *yhat0=malloc(sizeof(double)*size);
	double *zhat0=malloc(sizeof(double)*size);
	double *t=malloc(sizeof(double)*size);
	double *x=malloc(sizeof(double)*size);
	double *y=malloc(sizeof(double)*size);
	double *z=malloc(sizeof(double)*size);
	double *xhat=malloc(sizeof(double)*size);
	double *yhat=malloc(sizeof(double)*size);
	double *zhat=malloc(sizeof(double)*size);
    
    double n=0.0011;
	double nr=1/n;
	double n3=3*n;
	double n6=6*n;
	double nt=0;
	double sinnt=0;
	double cosnt=0;
    
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
    
    if(threads==1)
    {
        TIME_GET(&start);
        
        for(i=0;i<size;i++)
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
        
        TIME_GET(&end);
    }
    else
    {
        TIME_GET(&start);
        
        #pragma omp parallel for num_threads(threads) private(i)
        for(i=0;i<size;i++)
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
        
        TIME_GET(&end);
        
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
    
    return TIME_RUNTIME(start,end);
}
