#include "benchmarks.h"

double matrix_matrix_mult_int8(size, threads)
{
    int i,j,k;
    srand(SEED);
    TIME_STRUCT start, end;
    
    int8_t *A=malloc(sizeof(int8_t)*(size*size));
    int8_t *B=malloc(sizeof(int8_t)*(size*size));
    int16_t *C=malloc(sizeof(int16_t)*(size*size));
    
    for(i=0; i<(size*size); i++)
    {
        A[i]=(int8_t)rand();
        B[i]=(int8_t)rand();
        C[i]=0;
    }
    
    if(threads==1)
    {
        TIME_GET(&start);
        
        for(i=0; i<size; i++)
        {
            for(k=0; k<size; k++)
            {
                for(j=0; j<size; j++)
                {
                    C[i*size+j]+=A[i*size+k]*B[k*size+j];
                }
            }
        }
        
        TIME_GET(&end);
    }
    else
    {
        TIME_GET(&start);
        
        #pragma omp parallel for num_threads(threads) shared(A,B,C) private(i,j,k)
        for(i=0; i<size; i++)
        {
            for(k=0; k<size; k++)
            {
                for(j=0; j<size; j++)
                {
                    C[i*size+j]+=A[i*size+k]*B[k*size+j];
                }
            }
        }
        
        TIME_GET(&end);
        
    }
    
    free(C);
    free(B);
    free(A);
    
    return TIME_RUNTIME(start,end);
}

double matrix_matrix_mult_int16(size, threads)
{
    int i,j,k;
    srand(SEED);
    TIME_STRUCT start, end;
    
    int16_t *A=malloc(sizeof(int16_t)*(size*size));
    int16_t *B=malloc(sizeof(int16_t)*(size*size));
    int32_t *C=malloc(sizeof(int32_t)*(size*size));
    
    for(i=0; i<(size*size); i++)
    {
        A[i]=(int16_t)rand();
        B[i]=(int16_t)rand();
        C[i]=0;
    }
    
    if(threads==1)
    {
        TIME_GET(&start);
        
        for(i=0; i<size; i++)
        {
            for(k=0; k<size; k++)
            {
                for(j=0; j<size; j++)
                {
                    C[i*size+j]+=A[i*size+k]*B[k*size+j];
                }
            }
        }
        
        TIME_GET(&end);
    }
    else
    {
        TIME_GET(&start);
        
        #pragma omp parallel for num_threads(threads) shared(A,B,C) private(i,j,k)
        for(i=0; i<size; i++)
        {
            for(k=0; k<size; k++)
            {
                for(j=0; j<size; j++)
                {
                    C[i*size+j]+=A[i*size+k]*B[k*size+j];
                }
            }
        }
        
        TIME_GET(&end);
        
    }
    
    free(C);
    free(B);
    free(A);
    
    return TIME_RUNTIME(start,end);
}

double matrix_matrix_mult_int32(size, threads)
{
    int i,j,k;
    srand(SEED);
    TIME_STRUCT start, end;
    
    int32_t *A=malloc(sizeof(int32_t)*(size*size));
    int32_t *B=malloc(sizeof(int32_t)*(size*size));
    int64_t *C=malloc(sizeof(int64_t)*(size*size));
    
    for(i=0; i<(size*size); i++)
    {
        A[i]=(int32_t)rand();
        B[i]=(int32_t)rand();
        C[i]=0;
    }
    
    if(threads==1)
    {
        TIME_GET(&start);
        
        for(i=0; i<size; i++)
        {
            for(k=0; k<size; k++)
            {
                for(j=0; j<size; j++)
                {
                    C[i*size+j]+=A[i*size+k]*B[k*size+j];
                }
            }
        }
        
        TIME_GET(&end);
    }
    else
    {
        TIME_GET(&start);
        
        #pragma omp parallel for num_threads(threads) shared(A,B,C) private(i,j,k)
        for(i=0; i<size; i++)
        {
            for(k=0; k<size; k++)
            {
                for(j=0; j<size; j++)
                {
                    C[i*size+j]+=A[i*size+k]*B[k*size+j];
                }
            }
        }
        
        TIME_GET(&end);
        
    }
    
    free(C);
    free(B);
    free(A);
    
    return TIME_RUNTIME(start,end);
        
        
}

double matrix_matrix_mult_spfp(size, threads)
{
    int i,j,k;
    srand(SEED);
    TIME_STRUCT start, end;
    
    float *A=malloc(sizeof(float)*(size*size));
    float *B=malloc(sizeof(float)*(size*size));
    float *C=malloc(sizeof(float)*(size*size));
    
    for(i=0; i<(size*size); i++)
    {
        A[i]=(float)rand();
        B[i]=(float)rand();
        C[i]=0;
    }
    
    if(threads==1)
    {
        TIME_GET(&start);
        
        for(i=0; i<size; i++)
        {
            for(k=0; k<size; k++)
            {
                for(j=0; j<size; j++)
                {
                    C[i*size+j]+=A[i*size+k]*B[k*size+j];
                }
            }
        }
        
        TIME_GET(&end);
    }
    else
    {
        TIME_GET(&start);
        
        #pragma omp parallel for num_threads(threads) shared(A,B,C) private(i,j,k)
        for(i=0; i<size; i++)
        {
            for(k=0; k<size; k++)
            {
                for(j=0; j<size; j++)
                {
                    C[i*size+j]+=A[i*size+k]*B[k*size+j];
                }
            }
        }
        
        TIME_GET(&end);
        
    }
    
    free(C);
    free(B);
    free(A);
    
    return TIME_RUNTIME(start,end);
}

double matrix_matrix_mult_dpfp(size, threads)
{
    int i,j,k;
    srand(SEED);
    TIME_STRUCT start, end;
    
    double *A=malloc(sizeof(double)*(size*size));
    double *B=malloc(sizeof(double)*(size*size));
    double *C=malloc(sizeof(double)*(size*size));
    
    for(i=0; i<(size*size); i++)
    {
        A[i]=(double)rand();
        B[i]=(double)rand();
        C[i]=0;
    }
    
    if(threads==1)
    {
        TIME_GET(&start);
        
        for(i=0; i<size; i++)
        {
            for(k=0; k<size; k++)
            {
                for(j=0; j<size; j++)
                {
                    C[i*size+j]+=A[i*size+k]*B[k*size+j];
                }
            }
        }
        
        TIME_GET(&end);
    }
    else
    {
        TIME_GET(&start);
        
        #pragma omp parallel for num_threads(threads) shared(A,B,C) private(i,j,k)
        for(i=0; i<size; i++)
        {
            for(k=0; k<size; k++)
            {
                for(j=0; j<size; j++)
                {
                    C[i*size+j]+=A[i*size+k]*B[k*size+j];
                }
            }
        }
        
        TIME_GET(&end);
        
    }
    
    free(C);
    free(B);
    free(A);
    
    return TIME_RUNTIME(start,end);
}
