#include "benchmarks.h"

double sobel_int8(int size, int threads)
{
    srand(SEED);
    int i,j;
    TIME_STRUCT start,end;
    double runtime=0.0;
    
    int8_t *buf = malloc(sizeof(int8_t)*(size*size));
    int8_t *gx = malloc(sizeof(int8_t)*(size*size));
    int8_t *gy = malloc(sizeof(int8_t)*(size*size));
    int8_t *output = malloc(sizeof(int8_t)*(size*size));
    
    int x,y;
    
    for(i=0; i<(size*size); i++)
    {
        buf[i]=(int8_t)rand();
        output[i]=0;
    }
    
    
    
    if(threads==1)
    {
        TIME_GET(&start);
    
        for (i = 1; i < size-1; i++)
        {
            for (j = 1; j < size-1; j++)
            {
                gx[i*size + j] =
                    1*buf[(i-1)*size + (j-1)] + 0*buf[(i-1)*size + j] + -1*buf[(i-1)*size + (j+1)] +
                    2*buf[(i+0)*size + (j-1)] + 0*buf[(i+0)*size + j] + -2*buf[(i+0)*size + (j+1)] +
                    1*buf[(i+1)*size + (j-1)] + 0*buf[(i+1)*size + j] + -1*buf[(i+1)*size + (j+1)];
            
                gy[i*size + j] =
                    1*buf[(i-1)*size + (j-1)] +  2*buf[(i-1)*size + j] +  1*buf[(i-1)*size + (j+1)] +
                    0*buf[(i+0)*size + (j-1)] +  0*buf[(i+0)*size + j] +  0*buf[(i+0)*size + (j+1)] +
                    -1*buf[(i+1)*size + (j-1)] + -2*buf[(i+1)*size + j] + -1*buf[(i+1)*size + (j+1)];
            
                x = gx[i*size + j];
                y = gy[i*size + j];
            
                output[i*size + j] = sqrt(x*x + y*y);
            }
        }
        
        TIME_GET(&end);
    }
    else
    {
        TIME_GET(&start);
    
        #pragma omp parallel for num_threads(threads) shared(gx, gy, buf, output) private(x,y) collapse(2)
        for (i = 1; i < size-1; i++)
        {
            for (j = 1; j < size-1; j++)
            {
                gx[i*size + j] =
                    1*buf[(i-1)*size + (j-1)] + 0*buf[(i-1)*size + j] + -1*buf[(i-1)*size + (j+1)] +
                    2*buf[(i+0)*size + (j-1)] + 0*buf[(i+0)*size + j] + -2*buf[(i+0)*size + (j+1)] +
                    1*buf[(i+1)*size + (j-1)] + 0*buf[(i+1)*size + j] + -1*buf[(i+1)*size + (j+1)];
            
                gy[i*size + j] =
                    1*buf[(i-1)*size + (j-1)] +  2*buf[(i-1)*size + j] +  1*buf[(i-1)*size + (j+1)] +
                    0*buf[(i+0)*size + (j-1)] +  0*buf[(i+0)*size + j] +  0*buf[(i+0)*size + (j+1)] +
                    -1*buf[(i+1)*size + (j-1)] + -2*buf[(i+1)*size + j] + -1*buf[(i+1)*size + (j+1)];
            
                x = gx[i*size + j];
                y = gy[i*size + j];
            
                output[i*size + j] = sqrt(x*x + y*y);
            }
        }
        
        TIME_GET(&end);
        
    }
    
    
    
    runtime=TIME_RUNTIME(start,end);
    
    free(buf);
    free(output);
    free(gx);
    free(gy);
    
    return runtime;
    
}

double sobel_int16(int size, int threads)
{
    srand(SEED);
    int i,j;
    TIME_STRUCT start,end;
    double runtime=0.0;
    
    int16_t *buf = malloc(sizeof(int16_t)*(size*size));
    int16_t *gx = malloc(sizeof(int16_t)*(size*size));
    int16_t *gy = malloc(sizeof(int16_t)*(size*size));
    int16_t *output = malloc(sizeof(int16_t)*(size*size));
    
    int x,y;
    
    for(i=0; i<(size*size); i++)
    {
        buf[i]=(int16_t)rand();
        output[i]=0;
    }
    
    
    
    if(threads==1)
    {
        TIME_GET(&start);
    
        for (i = 1; i < size-1; i++)
        {
            for (j = 1; j < size-1; j++)
            {
                gx[i*size + j] =
                    1*buf[(i-1)*size + (j-1)] + 0*buf[(i-1)*size + j] + -1*buf[(i-1)*size + (j+1)] +
                    2*buf[(i+0)*size + (j-1)] + 0*buf[(i+0)*size + j] + -2*buf[(i+0)*size + (j+1)] +
                    1*buf[(i+1)*size + (j-1)] + 0*buf[(i+1)*size + j] + -1*buf[(i+1)*size + (j+1)];
            
                gy[i*size + j] =
                    1*buf[(i-1)*size + (j-1)] +  2*buf[(i-1)*size + j] +  1*buf[(i-1)*size + (j+1)] +
                    0*buf[(i+0)*size + (j-1)] +  0*buf[(i+0)*size + j] +  0*buf[(i+0)*size + (j+1)] +
                    -1*buf[(i+1)*size + (j-1)] + -2*buf[(i+1)*size + j] + -1*buf[(i+1)*size + (j+1)];
            
                x = gx[i*size + j];
                y = gy[i*size + j];
            
                output[i*size + j] = sqrt(x*x + y*y);
            }
        }
        
        TIME_GET(&end);
    }
    else
    {
        TIME_GET(&start);
    
        #pragma omp parallel for num_threads(threads) shared(gx, gy, buf, output) private(x,y) collapse(2)
        for (i = 1; i < size-1; i++)
        {
            for (j = 1; j < size-1; j++)
            {
                gx[i*size + j] =
                    1*buf[(i-1)*size + (j-1)] + 0*buf[(i-1)*size + j] + -1*buf[(i-1)*size + (j+1)] +
                    2*buf[(i+0)*size + (j-1)] + 0*buf[(i+0)*size + j] + -2*buf[(i+0)*size + (j+1)] +
                    1*buf[(i+1)*size + (j-1)] + 0*buf[(i+1)*size + j] + -1*buf[(i+1)*size + (j+1)];
            
                gy[i*size + j] =
                    1*buf[(i-1)*size + (j-1)] +  2*buf[(i-1)*size + j] +  1*buf[(i-1)*size + (j+1)] +
                    0*buf[(i+0)*size + (j-1)] +  0*buf[(i+0)*size + j] +  0*buf[(i+0)*size + (j+1)] +
                    -1*buf[(i+1)*size + (j-1)] + -2*buf[(i+1)*size + j] + -1*buf[(i+1)*size + (j+1)];
            
                x = gx[i*size + j];
                y = gy[i*size + j];
            
                output[i*size + j] = sqrt(x*x + y*y);
            }
        }
        
        TIME_GET(&end);
        
    }
    
    
    
    runtime=TIME_RUNTIME(start,end);
    
    free(buf);
    free(output);
    free(gx);
    free(gy);
    
    return runtime;
}

double sobel_int32(int size, int threads)
{
    srand(SEED);
    int i,j;
    TIME_STRUCT start,end;
    double runtime=0.0;
    
    int32_t *buf = malloc(sizeof(int32_t)*(size*size));
    int32_t *gx = malloc(sizeof(int32_t)*(size*size));
    int32_t *gy = malloc(sizeof(int32_t)*(size*size));
    int32_t *output = malloc(sizeof(int32_t)*(size*size));
    
    int x,y;
    
    for(i=0; i<(size*size); i++)
    {
        buf[i]=(int32_t)rand();
        output[i]=0;
    }
    
    
    
    if(threads==1)
    {
        TIME_GET(&start);
    
        for (i = 1; i < size-1; i++)
        {
            for (j = 1; j < size-1; j++)
            {
                gx[i*size + j] =
                    1*buf[(i-1)*size + (j-1)] + 0*buf[(i-1)*size + j] + -1*buf[(i-1)*size + (j+1)] +
                    2*buf[(i+0)*size + (j-1)] + 0*buf[(i+0)*size + j] + -2*buf[(i+0)*size + (j+1)] +
                    1*buf[(i+1)*size + (j-1)] + 0*buf[(i+1)*size + j] + -1*buf[(i+1)*size + (j+1)];
            
                gy[i*size + j] =
                    1*buf[(i-1)*size + (j-1)] +  2*buf[(i-1)*size + j] +  1*buf[(i-1)*size + (j+1)] +
                    0*buf[(i+0)*size + (j-1)] +  0*buf[(i+0)*size + j] +  0*buf[(i+0)*size + (j+1)] +
                    -1*buf[(i+1)*size + (j-1)] + -2*buf[(i+1)*size + j] + -1*buf[(i+1)*size + (j+1)];
            
                x = gx[i*size + j];
                y = gy[i*size + j];
            
                output[i*size + j] = sqrt(x*x + y*y);
            }
        }
        
        TIME_GET(&end);
    }
    else
    {
        TIME_GET(&start);
    
        #pragma omp parallel for num_threads(threads) shared(gx, gy, buf, output) private(x,y) collapse(2)
        for (i = 1; i < size-1; i++)
        {
            for (j = 1; j < size-1; j++)
            {
                gx[i*size + j] =
                    1*buf[(i-1)*size + (j-1)] + 0*buf[(i-1)*size + j] + -1*buf[(i-1)*size + (j+1)] +
                    2*buf[(i+0)*size + (j-1)] + 0*buf[(i+0)*size + j] + -2*buf[(i+0)*size + (j+1)] +
                    1*buf[(i+1)*size + (j-1)] + 0*buf[(i+1)*size + j] + -1*buf[(i+1)*size + (j+1)];
            
                gy[i*size + j] =
                    1*buf[(i-1)*size + (j-1)] +  2*buf[(i-1)*size + j] +  1*buf[(i-1)*size + (j+1)] +
                    0*buf[(i+0)*size + (j-1)] +  0*buf[(i+0)*size + j] +  0*buf[(i+0)*size + (j+1)] +
                    -1*buf[(i+1)*size + (j-1)] + -2*buf[(i+1)*size + j] + -1*buf[(i+1)*size + (j+1)];
            
                x = gx[i*size + j];
                y = gy[i*size + j];
            
                output[i*size + j] = sqrt(x*x + y*y);
            }
        }
        
        TIME_GET(&end);
        
    }
    
    
    
    runtime=TIME_RUNTIME(start,end);
    
    free(buf);
    free(output);
    free(gx);
    free(gy);
    
    return runtime;
}

double sobel_spfp(int size, int threads)
{
    srand(SEED);
    int i,j;
    TIME_STRUCT start,end;
    double runtime=0.0;
    
    float *buf = malloc(sizeof(float)*(size*size));
    float *gx = malloc(sizeof(float)*(size*size));
    float *gy = malloc(sizeof(float)*(size*size));
    float *output = malloc(sizeof(float)*(size*size));
    
    int x,y;
    
    for(i=0; i<(size*size); i++)
    {
        buf[i]=(float)rand();
        output[i]=0;
    }
    
    
    
    if(threads==1)
    {
        TIME_GET(&start);
    
        for (i = 1; i < size-1; i++)
        {
            for (j = 1; j < size-1; j++)
            {
                gx[i*size + j] =
                    1*buf[(i-1)*size + (j-1)] + 0*buf[(i-1)*size + j] + -1*buf[(i-1)*size + (j+1)] +
                    2*buf[(i+0)*size + (j-1)] + 0*buf[(i+0)*size + j] + -2*buf[(i+0)*size + (j+1)] +
                    1*buf[(i+1)*size + (j-1)] + 0*buf[(i+1)*size + j] + -1*buf[(i+1)*size + (j+1)];
            
                gy[i*size + j] =
                    1*buf[(i-1)*size + (j-1)] +  2*buf[(i-1)*size + j] +  1*buf[(i-1)*size + (j+1)] +
                    0*buf[(i+0)*size + (j-1)] +  0*buf[(i+0)*size + j] +  0*buf[(i+0)*size + (j+1)] +
                    -1*buf[(i+1)*size + (j-1)] + -2*buf[(i+1)*size + j] + -1*buf[(i+1)*size + (j+1)];
            
                x = gx[i*size + j];
                y = gy[i*size + j];
            
                output[i*size + j] = sqrt(x*x + y*y);
            }
        }
        
        TIME_GET(&end);
    }
    else
    {
        TIME_GET(&start);
    
        #pragma omp parallel for num_threads(threads) shared(gx, gy, buf, output) private(x,y) collapse(2)
        for (i = 1; i < size-1; i++)
        {
            for (j = 1; j < size-1; j++)
            {
                gx[i*size + j] =
                    1*buf[(i-1)*size + (j-1)] + 0*buf[(i-1)*size + j] + -1*buf[(i-1)*size + (j+1)] +
                    2*buf[(i+0)*size + (j-1)] + 0*buf[(i+0)*size + j] + -2*buf[(i+0)*size + (j+1)] +
                    1*buf[(i+1)*size + (j-1)] + 0*buf[(i+1)*size + j] + -1*buf[(i+1)*size + (j+1)];
            
                gy[i*size + j] =
                    1*buf[(i-1)*size + (j-1)] +  2*buf[(i-1)*size + j] +  1*buf[(i-1)*size + (j+1)] +
                    0*buf[(i+0)*size + (j-1)] +  0*buf[(i+0)*size + j] +  0*buf[(i+0)*size + (j+1)] +
                    -1*buf[(i+1)*size + (j-1)] + -2*buf[(i+1)*size + j] + -1*buf[(i+1)*size + (j+1)];
            
                x = gx[i*size + j];
                y = gy[i*size + j];
            
                output[i*size + j] = sqrt(x*x + y*y);
            }
        }
        
        TIME_GET(&end);
        
    }
    
    
    
    runtime=TIME_RUNTIME(start,end);
    
    free(buf);
    free(output);
    free(gx);
    free(gy);
    
    return runtime;
}

double sobel_dpfp(int size, int threads)
{
    srand(SEED);
    int i,j;
    TIME_STRUCT start,end;
    double runtime=0.0;
    
    double *buf = malloc(sizeof(double)*(size*size));
    double *gx = malloc(sizeof(double)*(size*size));
    double *gy = malloc(sizeof(double)*(size*size));
    double *output = malloc(sizeof(double)*(size*size));
    
    int x,y;
    
    for(i=0; i<(size*size); i++)
    {
        buf[i]=(double)rand();
        output[i]=0;
    }
    
    
    
    if(threads==1)
    {
        TIME_GET(&start);
    
        for (i = 1; i < size-1; i++)
        {
            for (j = 1; j < size-1; j++)
            {
                gx[i*size + j] =
                    1*buf[(i-1)*size + (j-1)] + 0*buf[(i-1)*size + j] + -1*buf[(i-1)*size + (j+1)] +
                    2*buf[(i+0)*size + (j-1)] + 0*buf[(i+0)*size + j] + -2*buf[(i+0)*size + (j+1)] +
                    1*buf[(i+1)*size + (j-1)] + 0*buf[(i+1)*size + j] + -1*buf[(i+1)*size + (j+1)];
            
                gy[i*size + j] =
                    1*buf[(i-1)*size + (j-1)] +  2*buf[(i-1)*size + j] +  1*buf[(i-1)*size + (j+1)] +
                    0*buf[(i+0)*size + (j-1)] +  0*buf[(i+0)*size + j] +  0*buf[(i+0)*size + (j+1)] +
                    -1*buf[(i+1)*size + (j-1)] + -2*buf[(i+1)*size + j] + -1*buf[(i+1)*size + (j+1)];
            
                x = gx[i*size + j];
                y = gy[i*size + j];
            
                output[i*size + j] = sqrt(x*x + y*y);
            }
        }
        
        TIME_GET(&end);
    }
    else
    {
        TIME_GET(&start);
    
        #pragma omp parallel for num_threads(threads) shared(gx, gy, buf, output) private(x,y) collapse(2)
        for (i = 1; i < size-1; i++)
        {
            for (j = 1; j < size-1; j++)
            {
                gx[i*size + j] =
                    1*buf[(i-1)*size + (j-1)] + 0*buf[(i-1)*size + j] + -1*buf[(i-1)*size + (j+1)] +
                    2*buf[(i+0)*size + (j-1)] + 0*buf[(i+0)*size + j] + -2*buf[(i+0)*size + (j+1)] +
                    1*buf[(i+1)*size + (j-1)] + 0*buf[(i+1)*size + j] + -1*buf[(i+1)*size + (j+1)];
            
                gy[i*size + j] =
                    1*buf[(i-1)*size + (j-1)] +  2*buf[(i-1)*size + j] +  1*buf[(i-1)*size + (j+1)] +
                    0*buf[(i+0)*size + (j-1)] +  0*buf[(i+0)*size + j] +  0*buf[(i+0)*size + (j+1)] +
                    -1*buf[(i+1)*size + (j-1)] + -2*buf[(i+1)*size + j] + -1*buf[(i+1)*size + (j+1)];
            
                x = gx[i*size + j];
                y = gy[i*size + j];
            
                output[i*size + j] = sqrt(x*x + y*y);
            }
        }
        
        TIME_GET(&end);
        
    }
    
    
    
    runtime=TIME_RUNTIME(start,end);
    
    free(buf);
    free(output);
    free(gx);
    free(gy);
    
    return runtime;
}
