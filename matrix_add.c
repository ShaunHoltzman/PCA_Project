#include "benchmarks.h"

double matrix_matrix_add_int8(int size, int threads)
{
    int i;
    int size2D=size*size;
    srand(SEED);
    TIME_STRUCT start, end;
    
    int8_t *A=malloc(sizeof(int8_t)*(size2D));
    int8_t *B=malloc(sizeof(int8_t)*(size2D));
    int8_t *C=malloc(sizeof(int8_t)*(size2D));
    
    for(i=0; i<size2D; i++)
    {
        A[i]=(int8_t)rand();
        B[i]=(int8_t)rand();
    }
    
    if(threads==1)
    {
        TIME_GET(&start);
        
        for(i=0; i<size2D; i++)
        {
            C[i]=A[i]+B[i];
        }
        
        TIME_GET(&end);
    }
    else
    {
        TIME_GET(&start);
        
        #pragma omp parallel for num_threads(threads) shared(A,B,C) private(i)
        for(i=0; i<size2D; i++)
        {
            C[i]=A[i]+B[i];
        }
        
        TIME_GET(&end);
    }
    
    free(C);
    free(B);
    free(A);
    
    return TIME_RUNTIME(start,end);
    
}

double matrix_matrix_add_int16(int size, int threads)
{
    int i;
    int size2D=size*size;
    srand(SEED);
    TIME_STRUCT start, end;
    
    int16_t *A=malloc(sizeof(int16_t)*(size2D));
    int16_t *B=malloc(sizeof(int16_t)*(size2D));
    int16_t *C=malloc(sizeof(int16_t)*(size2D));
    
    for(i=0; i<size2D; i++)
    {
        A[i]=(int16_t)rand();
        B[i]=(int16_t)rand();
    }
    
    if(threads==1)
    {
        TIME_GET(&start);
        
        for(i=0; i<size2D; i++)
        {
            C[i]=A[i]+B[i];
        }
        
        TIME_GET(&end);
    }
    else
    {
        TIME_GET(&start);
        
        #pragma omp parallel for num_threads(threads) shared(A,B,C) private(i)
        for(i=0; i<size2D; i++)
        {
            C[i]=A[i]+B[i];
        }
        
        TIME_GET(&end);
    }
    
    free(C);
    free(B);
    free(A);
    
    return TIME_RUNTIME(start,end);
}

double matrix_matrix_add_int32(int size, int threads)
{
    int i;
    int size2D=size*size;
    srand(SEED);
    TIME_STRUCT start, end;
    
    int32_t *A=malloc(sizeof(int32_t)*(size2D));
    int32_t *B=malloc(sizeof(int32_t)*(size2D));
    int32_t *C=malloc(sizeof(int32_t)*(size2D));
    
    for(i=0; i<size2D; i++)
    {
        A[i]=(int32_t)rand();
        B[i]=(int32_t)rand();
    }
    
    if(threads==1)
    {
        TIME_GET(&start);
        
        for(i=0; i<size2D; i++)
        {
            C[i]=A[i]+B[i];
        }
        
        TIME_GET(&end);
    }
    else
    {
        TIME_GET(&start);
        
        #pragma omp parallel for num_threads(threads) shared(A,B,C) private(i)
        for(i=0; i<size2D; i++)
        {
            C[i]=A[i]+B[i];
        }
        
        TIME_GET(&end);
    }
    
    free(C);
    free(B);
    free(A);
    
    return TIME_RUNTIME(start,end);
}

double matrix_matrix_add_spfp(int size, int threads)
{
    int i;
    int size2D=size*size;
    srand(SEED);
    TIME_STRUCT start, end;
    
    float *A=malloc(sizeof(float)*(size2D));
    float *B=malloc(sizeof(float)*(size2D));
    float *C=malloc(sizeof(float)*(size2D));
    
    for(i=0; i<size2D; i++)
    {
        A[i]=(float)rand();
        B[i]=(float)rand();
    }
    
    if(threads==1)
    {
        TIME_GET(&start);
        
        for(i=0; i<size2D; i++)
        {
            C[i]=A[i]+B[i];
        }
        
        TIME_GET(&end);
    }
    else
    {
        TIME_GET(&start);
        
        #pragma omp parallel for num_threads(threads) shared(A,B,C) private(i)
        for(i=0; i<size2D; i++)
        {
            C[i]=A[i]+B[i];
        }
        
        TIME_GET(&end);
    }
    
    free(C);
    free(B);
    free(A);
    
    return TIME_RUNTIME(start,end);
}

double matrix_matrix_add_dpfp(int size, int threads)
{
    int i;
    int size2D=size*size;
    srand(SEED);
    TIME_STRUCT start, end;
    
    double *A=malloc(sizeof(double)*(size2D));
    double *B=malloc(sizeof(double)*(size2D));
    double *C=malloc(sizeof(double)*(size2D));
    
    for(i=0; i<size2D; i++)
    {
        A[i]=(double)rand();
        B[i]=(double)rand();
    }
    
    if(threads==1)
    {
        TIME_GET(&start);
        
        for(i=0; i<size2D; i++)
        {
            C[i]=A[i]+B[i];
        }
        
        TIME_GET(&end);
    }
    else
    {
        TIME_GET(&start);
        
        #pragma omp parallel for num_threads(threads) shared(A,B,C) private(i)
        for(i=0; i<size2D; i++)
        {
            C[i]=A[i]+B[i];
        }
        
        TIME_GET(&end);
    }
    
    free(C);
    free(B);
    free(A);
    
    return TIME_RUNTIME(start,end);

}
