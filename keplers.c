#include "benchmarks.h"

double keplers_spfp(int size, int threads)
{
    int i,j;
    srand(SEED);
    TIME_STRUCT start, end;
    
    float threshold=1E-8;
    
    float *e=malloc(sizeof(float)*size);
    float *M=malloc(sizeof(float)*size);
    float *E=malloc(sizeof(float)*size);
    float *Eold=malloc(sizeof(float)*size);
    
    for(i=0;i<size;i++)
    {
        e[i]=rand()/(float)RAND_MAX;
        M[i]=2*M_PI*rand()/(float)RAND_MAX;
        E[i]=M[i];
        Eold[i]=0;
    }
    
    if(threads==1)
    {
        TIME_GET(&start);
        
        for(i=0;i<size;i++)
        {
            while(fabs(E[i]-Eold[i])>threshold)
            {
                Eold[i]=E[i];
				E[i]=E[i]-(E[i]-e[i]*sin(E[i])-M[i])/(1-e[i]*cos(E[i]));
            }
        }
        
        TIME_GET(&end);
        
    }
    else
    {
        TIME_GET(&start);
        
        #pragma omp parallel for num_threads(threads) private(i)
        for(i=0;i<size;i++)
        {
            while(fabs(E[i]-Eold[i])>threshold)
            {
                Eold[i]=E[i];
				E[i]=E[i]-(E[i]-e[i]*sin(E[i])-M[i])/(1-e[i]*cos(E[i]));
            }
        }
        
        TIME_GET(&end);
        
    }
    
    free(Eold);
    free(E);
    free(e);
    
    return TIME_RUNTIME(start,end);
                
}

double keplers_dpfp(int size, int threads)
{
    int i,j;
    srand(SEED);
    TIME_STRUCT start, end;
    
    double threshold=1E-8;
    
    double *e=malloc(sizeof(double)*size);
    double *M=malloc(sizeof(double)*size);
    double *E=malloc(sizeof(double)*size);
    double *Eold=malloc(sizeof(double)*size);
    
    for(i=0;i<size;i++)
    {
        e[i]=rand()/(double)RAND_MAX;
        M[i]=2*M_PI*rand()/(double)RAND_MAX;
        E[i]=M[i];
        Eold[i]=0;
    }
    
    if(threads==1)
    {
        TIME_GET(&start);
        
        for(i=0;i<size;i++)
        {
            while(fabs(E[i]-Eold[i])>threshold)
            {
                Eold[i]=E[i];
				E[i]=E[i]-(E[i]-e[i]*sin(E[i])-M[i])/(1-e[i]*cos(E[i]));
            }
        }
        
        TIME_GET(&end);
        
    }
    else
    {
        TIME_GET(&start);
        
        #pragma omp parallel for num_threads(threads) private(i)
        for(i=0;i<size;i++)
        {
            while(fabs(E[i]-Eold[i])>threshold)
            {
                Eold[i]=E[i];
				E[i]=E[i]-(E[i]-e[i]*sin(E[i])-M[i])/(1-e[i]*cos(E[i]));
            }
        }
        
        TIME_GET(&end);
        
    }
    
    free(Eold);
    free(E);
    free(e);
    
    return TIME_RUNTIME(start,end);
}
