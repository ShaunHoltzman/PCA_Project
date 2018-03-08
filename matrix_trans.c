#include "benchmarks.h"

double matrix_matrix_trans_int8(int size, int threads)
{
    int i,j;
    srand(SEED);
    TIME_STRUCT start,end;
    
    int8_t *A=malloc(sizeof(int8_t)*(size*size));
    int8_t *B=malloc(sizeof(int8_t)*(size*size));
    
    for(i=0;i<(size*size);i++)
    {
        A[i]=(int8_t)rand();
        B[i]=0;
    }
    
    if(threads==1)
    {
        TIME_GET(&start);
        
        for(i=0;i<size;i++)
        {
            for(j=0;j<size;j++)
            {
                B[j*size+i]=A[i*size+j];
            }
        }
        
        TIME_GET(&end);
    }
    else
    {
        TIME_GET(&start);
        
        #pragma omp parallel for num_threads(threads) private(i,j)
        for(i=0;i<size;i++)
        {
            for(j=0;j<size;j++)
            {
                B[j*size+i]=A[i*size+j];
            }
        }
        
        TIME_GET(&end);
    }
    
    free(B);
    free(A);
    
    return TIME_RUNTIME(start,end);
}

double matrix_matrix_trans_int16(int size, int threads)
{
    int i,j;
    srand(SEED);
    TIME_STRUCT start,end;
    
    int16_t *A=malloc(sizeof(int16_t)*(size*size));
    int16_t *B=malloc(sizeof(int16_t)*(size*size));
    
    for(i=0;i<(size*size);i++)
    {
        A[i]=(int16_t)rand();
        B[i]=0;
    }
    
    if(threads==1)
    {
        TIME_GET(&start);
        
        for(i=0;i<size;i++)
        {
            for(j=0;j<size;j++)
            {
                B[j*size+i]=A[i*size+j];
            }
        }
        
        TIME_GET(&end);
    }
    else
    {
        TIME_GET(&start);
        
        #pragma omp parallel for num_threads(threads) private(i,j)
        for(i=0;i<size;i++)
        {
            for(j=0;j<size;j++)
            {
                B[j*size+i]=A[i*size+j];
            }
        }
        
        TIME_GET(&end);
    }
    
    free(B);
    free(A);
    
    return TIME_RUNTIME(start,end);
}

double matrix_matrix_trans_int32(int size, int threads)
{
    int i,j;
    srand(SEED);
    TIME_STRUCT start,end;
    
    int32_t *A=malloc(sizeof(int32_t)*(size*size));
    int32_t *B=malloc(sizeof(int32_t)*(size*size));
    
    for(i=0;i<(size*size);i++)
    {
        A[i]=(int32_t)rand();
        B[i]=0;
    }
    
    if(threads==1)
    {
        TIME_GET(&start);
        
        for(i=0;i<size;i++)
        {
            for(j=0;j<size;j++)
            {
                B[j*size+i]=A[i*size+j];
            }
        }
        
        TIME_GET(&end);
    }
    else
    {
        TIME_GET(&start);
        
        #pragma omp parallel for num_threads(threads) private(i,j)
        for(i=0;i<size;i++)
        {
            for(j=0;j<size;j++)
            {
                B[j*size+i]=A[i*size+j];
            }
        }
        
        TIME_GET(&end);
    }
    
    free(B);
    free(A);
    
    return TIME_RUNTIME(start,end);
}

double matrix_matrix_trans_spfp(int size, int threads)
{
    int i,j;
    srand(SEED);
    TIME_STRUCT start,end;
    
    float *A=malloc(sizeof(float)*(size*size));
    float *B=malloc(sizeof(float)*(size*size));
    
    for(i=0;i<(size*size);i++)
    {
        A[i]=(float)rand();
        B[i]=0;
    }
    
    if(threads==1)
    {
        TIME_GET(&start);
        
        for(i=0;i<size;i++)
        {
            for(j=0;j<size;j++)
            {
                B[j*size+i]=A[i*size+j];
            }
        }
        
        TIME_GET(&end);
    }
    else
    {
        TIME_GET(&start);
        
        #pragma omp parallel for num_threads(threads) private(i,j)
        for(i=0;i<size;i++)
        {
            for(j=0;j<size;j++)
            {
                B[j*size+i]=A[i*size+j];
            }
        }
        
        TIME_GET(&end);
    }
    
    free(B);
    free(A);
    
    return TIME_RUNTIME(start,end);
}

double matrix_matrix_trans_dpfp(int size, int threads)
{
    int i,j;
    srand(SEED);
    TIME_STRUCT start,end;
    
    double *A=malloc(sizeof(double)*(size*size));
    double *B=malloc(sizeof(double)*(size*size));
    
    for(i=0;i<(size*size);i++)
    {
        A[i]=(double)rand();
        B[i]=0;
    }
    
    if(threads==1)
    {
        TIME_GET(&start);
        
        for(i=0;i<size;i++)
        {
            for(j=0;j<size;j++)
            {
                B[j*size+i]=A[i*size+j];
            }
        }
        
        TIME_GET(&end);
    }
    else
    {
        TIME_GET(&start);
        
        #pragma omp parallel for num_threads(threads) private(i,j)
        for(i=0;i<size;i++)
        {
            for(j=0;j<size;j++)
            {
                B[j*size+i]=A[i*size+j];
            }
        }
        
        TIME_GET(&end);
    }
    
    free(B);
    free(A);
    
    return TIME_RUNTIME(start,end);
}
