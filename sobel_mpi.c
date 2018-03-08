#include "benchmarks.h"

double sobel_int8_mpi(int size, int threads)
{
    int rank, num_cores;
    MPI_Comm_size(MPI_COMM_WORLD, &num_cores);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    srand(SEED);
    int i,j;
    TIME_STRUCT start,end;
    TIME_STRUCT send_start,send_end;
    TIME_STRUCT recv_start,recv_end;
    double runtime=0.0;
    double sendtime=0.0;
    double recvtime=0.0;
    double calctime=0.0;
    
    int8_t *buf;
    int8_t *gx;
    int8_t *gy;
    int8_t *recv_buf;
    int recv_size;
    
    int W=size;
    int H=size;
    
    if(rank==0)
    {
        buf=malloc(sizeof(int8_t)*(size*size));
        for(i=0;i<(W*H);i++)
        {
            buf[i]=(int8_t)rand();
        }
    }
    
    int H_new;
    int remainder;
    
    H_new=floor(H/num_cores);
    
    remainder=H-H_new*num_cores;
    
    int *send_count = malloc(sizeof(int)*num_cores);
    int *disp_count = malloc(sizeof(int)*num_cores);
    
    for(i=0; i<num_cores; i++)
    {
        if(i==0)
        {
            send_count[i] = W*(H_new+1);
            disp_count[i] = 0;
        }
        else if(i==num_cores-1)
        {
            send_count[i] = W*(H_new+1+remainder);
            disp_count[i] = W*H_new*i - W;
        }
        else
        {
            send_count[i] = W*(H_new+2);
            disp_count[i] = W*H_new*i - W;
        }
    }
    
    if(rank == 0)
	{
		recv_size = W*(H_new+1);
		recv_buf = malloc(recv_size*sizeof(int8_t));
		gx = malloc(W * H_new * sizeof(int8_t));
		gy = malloc(W * H_new * sizeof(int8_t));
	}
	else if(rank==num_cores-1)
	{
		recv_size = W*(H_new+1+remainder);
		recv_buf = malloc(recv_size*sizeof(int8_t));
		gx = malloc(W * (H_new+remainder) * sizeof(int8_t));
		gy = malloc(W * (H_new+remainder) * sizeof(int8_t));
	}
	else
	{
		recv_size = W*(H_new+2);
		recv_buf = malloc(recv_size*sizeof(int8_t));
		gx = malloc(W * (H_new+1) * sizeof(int8_t));
		gy = malloc(W * (H_new+1) * sizeof(int8_t));
	}
    
    if(rank==0)
    {
        TIME_GET(&send_start);
    }
    
    MPI_Scatterv(buf, send_count, disp_count, MPI_CHAR, recv_buf, recv_size, MPI_CHAR, 0, MPI_COMM_WORLD);
    
    if(rank==0)
    {
        TIME_GET(&send_end);
        TIME_GET(&start);
    }
    
    if(rank==0)
	{
		for (i = 1; i < H_new; i++)
			for (j = 1; j < W-1; j++)
				gx[i*W + j] =
					1*recv_buf[(i-1)*W + (j-1)] + 0*recv_buf[(i-1)*W + j] + -1*recv_buf[(i-1)*W + (j+1)] +
					2*recv_buf[(i+0)*W + (j-1)] + 0*recv_buf[(i+0)*W + j] + -2*recv_buf[(i+0)*W + (j+1)] +
					1*recv_buf[(i+1)*W + (j-1)] + 0*recv_buf[(i+1)*W + j] + -1*recv_buf[(i+1)*W + (j+1)];
		for (i = 1; i < H_new; i++)
			for (j = 1; j < W-1; j++)
				gy[i*W + j] =
					 1*recv_buf[(i-1)*W + (j-1)] +  2*recv_buf[(i-1)*W + j] +  1*recv_buf[(i-1)*W + (j+1)] +
					 0*recv_buf[(i+0)*W + (j-1)] +  0*recv_buf[(i+0)*W + j] +  0*recv_buf[(i+0)*W + (j+1)] +
					-1*recv_buf[(i+1)*W + (j-1)] + -2*recv_buf[(i+1)*W + j] + -1*recv_buf[(i+1)*W + (j+1)];
		for (i = 1; i < H_new; i++)
			for (j = 1; j < W-1; j++) {
				int x = gx[i*W + j];
				int y = gy[i*W + j];
				recv_buf[i*W + j] = sqrt(x*x + y*y);
			}
	}
	else if(rank==num_cores-1)
	{
		for (i = 1; i < H_new+remainder; i++)
			for (j = 1; j < W-1; j++)
				gx[i*W + j] =
					1*recv_buf[(i-1)*W + (j-1)] + 0*recv_buf[(i-1)*W + j] + -1*recv_buf[(i-1)*W + (j+1)] +
					2*recv_buf[(i+0)*W + (j-1)] + 0*recv_buf[(i+0)*W + j] + -2*recv_buf[(i+0)*W + (j+1)] +
					1*recv_buf[(i+1)*W + (j-1)] + 0*recv_buf[(i+1)*W + j] + -1*recv_buf[(i+1)*W + (j+1)];
		for (i = 1; i < H_new+remainder; i++)
			for (j = 1; j < W-1; j++)
				gy[i*W + j] =
					 1*recv_buf[(i-1)*W + (j-1)] +  2*recv_buf[(i-1)*W + j] +  1*recv_buf[(i-1)*W + (j+1)] +
					 0*recv_buf[(i+0)*W + (j-1)] +  0*recv_buf[(i+0)*W + j] +  0*recv_buf[(i+0)*W + (j+1)] +
					-1*recv_buf[(i+1)*W + (j-1)] + -2*recv_buf[(i+1)*W + j] + -1*recv_buf[(i+1)*W + (j+1)];
		for (i = 1; i < H_new+remainder; i++)
			for (j = 1; j < W-1; j++) {
				int x = gx[i*W + j];
				int y = gy[i*W + j];
				recv_buf[i*W + j] = sqrt(x*x + y*y);
			}
	}
	else
	{
		for (i = 1; i < H_new+1; i++)
			for (j = 1; j < W-1; j++)
				gx[i*W + j] =
					1*recv_buf[(i-1)*W + (j-1)] + 0*recv_buf[(i-1)*W + j] + -1*recv_buf[(i-1)*W + (j+1)] +
					2*recv_buf[(i+0)*W + (j-1)] + 0*recv_buf[(i+0)*W + j] + -2*recv_buf[(i+0)*W + (j+1)] +
					1*recv_buf[(i+1)*W + (j-1)] + 0*recv_buf[(i+1)*W + j] + -1*recv_buf[(i+1)*W + (j+1)];
		for (i = 1; i < H_new+1; i++)
			for (j = 1; j < W-1; j++)
				gy[i*W + j] =
					 1*recv_buf[(i-1)*W + (j-1)] +  2*recv_buf[(i-1)*W + j] +  1*recv_buf[(i-1)*W + (j+1)] +
					 0*recv_buf[(i+0)*W + (j-1)] +  0*recv_buf[(i+0)*W + j] +  0*recv_buf[(i+0)*W + (j+1)] +
					-1*recv_buf[(i+1)*W + (j-1)] + -2*recv_buf[(i+1)*W + j] + -1*recv_buf[(i+1)*W + (j+1)];
		for (i = 1; i < H_new+1; i++)
			for (j = 1; j < W-1; j++) {
				int x = gx[i*W + j];
				int y = gy[i*W + j];
				recv_buf[i*W + j] = sqrt(x*x + y*y);
			}
	}
    
    if(rank==0)
    {
        TIME_GET(&end);
    }
    
    int *recv_count = malloc(sizeof(int)*num_cores);
	int *disp_count_r = malloc(sizeof(int)*num_cores);
	int index;
	int size2;
	
	for(int i=0; i<num_cores; i++)
	{
		
		if(i == 0)
		{
			recv_count[i] = W*(H_new);
			disp_count_r[i] = 0;
		}
		else if(i == num_cores-1)
		{
			recv_count[i] = W*(H_new+remainder);
			disp_count_r[i] = W*(H_new)*i;
		}
		else
		{
			recv_count[i] = W*(H_new);
			disp_count_r[i] = W*(H_new)*i;
		}	
		
	}
	
	if(rank == 0)
		index=0;
	else
		index=W;
	
	if(rank==num_cores-1)
		size2 = W*(H_new+remainder);
	else
		size2 = W*(H_new);
    
    if(rank==0)
    {
        TIME_GET(&recv_start);
    }
    
    MPI_Gatherv(&recv_buf[index], size2, MPI_CHAR, buf, recv_count, disp_count_r, MPI_CHAR, 0, MPI_COMM_WORLD);
    
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
    
    if(rank==0)
    {
        free(buf);
    }
    free(gx);
    free(gy);
    free(recv_buf);
    
    return runtime;
    
}

double sobel_int16_mpi(int size, int threads)
{
    int rank, num_cores;
    MPI_Comm_size(MPI_COMM_WORLD, &num_cores);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    srand(SEED);
    int i,j;
    TIME_STRUCT start,end;
    TIME_STRUCT send_start,send_end;
    TIME_STRUCT recv_start,recv_end;
    double runtime=0.0;
    double sendtime=0.0;
    double recvtime=0.0;
    double calctime=0.0;
    
    int16_t *buf;
    int16_t *gx;
    int16_t *gy;
    int16_t *recv_buf;
    int recv_size;
    
    int W=size;
    int H=size;
    
    if(rank==0)
    {
        buf=malloc(sizeof(int16_t)*(size*size));
        for(i=0;i<(W*H);i++)
        {
            buf[i]=(int16_t)rand();
        }
    }
    
    int H_new;
    int remainder;
    
    H_new=floor(H/num_cores);
    
    remainder=H-H_new*num_cores;
    
    int *send_count = malloc(sizeof(int)*num_cores);
    int *disp_count = malloc(sizeof(int)*num_cores);
    
    for(i=0; i<num_cores; i++)
    {
        if(i==0)
        {
            send_count[i] = W*(H_new+1);
            disp_count[i] = 0;
        }
        else if(i==num_cores-1)
        {
            send_count[i] = W*(H_new+1+remainder);
            disp_count[i] = W*H_new*i - W;
        }
        else
        {
            send_count[i] = W*(H_new+2);
            disp_count[i] = W*H_new*i - W;
        }
    }
    
    if(rank == 0)
	{
		recv_size = W*(H_new+1);
		recv_buf = malloc(recv_size*sizeof(int16_t));
		gx = malloc(W * H_new * sizeof(int16_t));
		gy = malloc(W * H_new * sizeof(int16_t));
	}
	else if(rank==num_cores-1)
	{
		recv_size = W*(H_new+1+remainder);
		recv_buf = malloc(recv_size*sizeof(int16_t));
		gx = malloc(W * (H_new+remainder) * sizeof(int16_t));
		gy = malloc(W * (H_new+remainder) * sizeof(int16_t));
	}
	else
	{
		recv_size = W*(H_new+2);
		recv_buf = malloc(recv_size*sizeof(int16_t));
		gx = malloc(W * (H_new+1) * sizeof(int16_t));
		gy = malloc(W * (H_new+1) * sizeof(int16_t));
	}
    
    if(rank==0)
    {
        TIME_GET(&send_start);
    }
    
    MPI_Scatterv(buf, send_count, disp_count, MPI_SHORT, recv_buf, recv_size, MPI_SHORT, 0, MPI_COMM_WORLD);
    
    if(rank==0)
    {
        TIME_GET(&send_end);
        TIME_GET(&start);
    }
    
    if(rank==0)
	{
		for (i = 1; i < H_new; i++)
			for (j = 1; j < W-1; j++)
				gx[i*W + j] =
					1*recv_buf[(i-1)*W + (j-1)] + 0*recv_buf[(i-1)*W + j] + -1*recv_buf[(i-1)*W + (j+1)] +
					2*recv_buf[(i+0)*W + (j-1)] + 0*recv_buf[(i+0)*W + j] + -2*recv_buf[(i+0)*W + (j+1)] +
					1*recv_buf[(i+1)*W + (j-1)] + 0*recv_buf[(i+1)*W + j] + -1*recv_buf[(i+1)*W + (j+1)];
		for (i = 1; i < H_new; i++)
			for (j = 1; j < W-1; j++)
				gy[i*W + j] =
					 1*recv_buf[(i-1)*W + (j-1)] +  2*recv_buf[(i-1)*W + j] +  1*recv_buf[(i-1)*W + (j+1)] +
					 0*recv_buf[(i+0)*W + (j-1)] +  0*recv_buf[(i+0)*W + j] +  0*recv_buf[(i+0)*W + (j+1)] +
					-1*recv_buf[(i+1)*W + (j-1)] + -2*recv_buf[(i+1)*W + j] + -1*recv_buf[(i+1)*W + (j+1)];
		for (i = 1; i < H_new; i++)
			for (j = 1; j < W-1; j++) {
				int x = gx[i*W + j];
				int y = gy[i*W + j];
				recv_buf[i*W + j] = sqrt(x*x + y*y);
			}
	}
	else if(rank==num_cores-1)
	{
		for (i = 1; i < H_new+remainder; i++)
			for (j = 1; j < W-1; j++)
				gx[i*W + j] =
					1*recv_buf[(i-1)*W + (j-1)] + 0*recv_buf[(i-1)*W + j] + -1*recv_buf[(i-1)*W + (j+1)] +
					2*recv_buf[(i+0)*W + (j-1)] + 0*recv_buf[(i+0)*W + j] + -2*recv_buf[(i+0)*W + (j+1)] +
					1*recv_buf[(i+1)*W + (j-1)] + 0*recv_buf[(i+1)*W + j] + -1*recv_buf[(i+1)*W + (j+1)];
		for (i = 1; i < H_new+remainder; i++)
			for (j = 1; j < W-1; j++)
				gy[i*W + j] =
					 1*recv_buf[(i-1)*W + (j-1)] +  2*recv_buf[(i-1)*W + j] +  1*recv_buf[(i-1)*W + (j+1)] +
					 0*recv_buf[(i+0)*W + (j-1)] +  0*recv_buf[(i+0)*W + j] +  0*recv_buf[(i+0)*W + (j+1)] +
					-1*recv_buf[(i+1)*W + (j-1)] + -2*recv_buf[(i+1)*W + j] + -1*recv_buf[(i+1)*W + (j+1)];
		for (i = 1; i < H_new+remainder; i++)
			for (j = 1; j < W-1; j++) {
				int x = gx[i*W + j];
				int y = gy[i*W + j];
				recv_buf[i*W + j] = sqrt(x*x + y*y);
			}
	}
	else
	{
		for (i = 1; i < H_new+1; i++)
			for (j = 1; j < W-1; j++)
				gx[i*W + j] =
					1*recv_buf[(i-1)*W + (j-1)] + 0*recv_buf[(i-1)*W + j] + -1*recv_buf[(i-1)*W + (j+1)] +
					2*recv_buf[(i+0)*W + (j-1)] + 0*recv_buf[(i+0)*W + j] + -2*recv_buf[(i+0)*W + (j+1)] +
					1*recv_buf[(i+1)*W + (j-1)] + 0*recv_buf[(i+1)*W + j] + -1*recv_buf[(i+1)*W + (j+1)];
		for (i = 1; i < H_new+1; i++)
			for (j = 1; j < W-1; j++)
				gy[i*W + j] =
					 1*recv_buf[(i-1)*W + (j-1)] +  2*recv_buf[(i-1)*W + j] +  1*recv_buf[(i-1)*W + (j+1)] +
					 0*recv_buf[(i+0)*W + (j-1)] +  0*recv_buf[(i+0)*W + j] +  0*recv_buf[(i+0)*W + (j+1)] +
					-1*recv_buf[(i+1)*W + (j-1)] + -2*recv_buf[(i+1)*W + j] + -1*recv_buf[(i+1)*W + (j+1)];
		for (i = 1; i < H_new+1; i++)
			for (j = 1; j < W-1; j++) {
				int x = gx[i*W + j];
				int y = gy[i*W + j];
				recv_buf[i*W + j] = sqrt(x*x + y*y);
			}
	}
    
    if(rank==0)
    {
        TIME_GET(&end);
    }
    
    int *recv_count = malloc(sizeof(int)*num_cores);
	int *disp_count_r = malloc(sizeof(int)*num_cores);
	int index;
	int size2;
	
	for(int i=0; i<num_cores; i++)
	{
		
		if(i == 0)
		{
			recv_count[i] = W*(H_new);
			disp_count_r[i] = 0;
		}
		else if(i == num_cores-1)
		{
			recv_count[i] = W*(H_new+remainder);
			disp_count_r[i] = W*(H_new)*i;
		}
		else
		{
			recv_count[i] = W*(H_new);
			disp_count_r[i] = W*(H_new)*i;
		}	
		
	}
	
	if(rank == 0)
		index=0;
	else
		index=W;
	
	if(rank==num_cores-1)
		size2 = W*(H_new+remainder);
	else
		size2 = W*(H_new);
    
    if(rank==0)
    {
        TIME_GET(&recv_start);
    }
    
    MPI_Gatherv(&recv_buf[index], size2, MPI_SHORT, buf, recv_count, disp_count_r, MPI_SHORT, 0, MPI_COMM_WORLD);
    
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
    
    if(rank==0)
    {
        free(buf);
    }
    free(gx);
    free(gy);
    free(recv_buf);
    
    return runtime;
}

double sobel_int32_mpi(int size, int threads)
{
    int rank, num_cores;
    MPI_Comm_size(MPI_COMM_WORLD, &num_cores);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    srand(SEED);
    int i,j;
    TIME_STRUCT start,end;
    TIME_STRUCT send_start,send_end;
    TIME_STRUCT recv_start,recv_end;
    double runtime=0.0;
    double sendtime=0.0;
    double recvtime=0.0;
    double calctime=0.0;
    
    int32_t *buf;
    int32_t *gx;
    int32_t *gy;
    int32_t *recv_buf;
    int recv_size;
    
    int W=size;
    int H=size;
    
    if(rank==0)
    {
        buf=malloc(sizeof(int32_t)*(size*size));
        for(i=0;i<(W*H);i++)
        {
            buf[i]=(int32_t)rand();
        }
    }
    
    int H_new;
    int remainder;
    
    H_new=floor(H/num_cores);
    
    remainder=H-H_new*num_cores;
    
    int *send_count = malloc(sizeof(int)*num_cores);
    int *disp_count = malloc(sizeof(int)*num_cores);
    
    for(i=0; i<num_cores; i++)
    {
        if(i==0)
        {
            send_count[i] = W*(H_new+1);
            disp_count[i] = 0;
        }
        else if(i==num_cores-1)
        {
            send_count[i] = W*(H_new+1+remainder);
            disp_count[i] = W*H_new*i - W;
        }
        else
        {
            send_count[i] = W*(H_new+2);
            disp_count[i] = W*H_new*i - W;
        }
    }
    
    if(rank == 0)
	{
		recv_size = W*(H_new+1);
		recv_buf = malloc(recv_size*sizeof(int32_t));
		gx = malloc(W * H_new * sizeof(int32_t));
		gy = malloc(W * H_new * sizeof(int32_t));
	}
	else if(rank==num_cores-1)
	{
		recv_size = W*(H_new+1+remainder);
		recv_buf = malloc(recv_size*sizeof(int32_t));
		gx = malloc(W * (H_new+remainder) * sizeof(int32_t));
		gy = malloc(W * (H_new+remainder) * sizeof(int32_t));
	}
	else
	{
		recv_size = W*(H_new+2);
		recv_buf = malloc(recv_size*sizeof(int32_t));
		gx = malloc(W * (H_new+1) * sizeof(int32_t));
		gy = malloc(W * (H_new+1) * sizeof(int32_t));
	}
    
    if(rank==0)
    {
        TIME_GET(&send_start);
    }
    
    MPI_Scatterv(buf, send_count, disp_count, MPI_INT, recv_buf, recv_size, MPI_INT, 0, MPI_COMM_WORLD);
    
    if(rank==0)
    {
        TIME_GET(&send_end);
        TIME_GET(&start);
    }
    
    if(rank==0)
	{
		for (i = 1; i < H_new; i++)
			for (j = 1; j < W-1; j++)
				gx[i*W + j] =
					1*recv_buf[(i-1)*W + (j-1)] + 0*recv_buf[(i-1)*W + j] + -1*recv_buf[(i-1)*W + (j+1)] +
					2*recv_buf[(i+0)*W + (j-1)] + 0*recv_buf[(i+0)*W + j] + -2*recv_buf[(i+0)*W + (j+1)] +
					1*recv_buf[(i+1)*W + (j-1)] + 0*recv_buf[(i+1)*W + j] + -1*recv_buf[(i+1)*W + (j+1)];
		for (i = 1; i < H_new; i++)
			for (j = 1; j < W-1; j++)
				gy[i*W + j] =
					 1*recv_buf[(i-1)*W + (j-1)] +  2*recv_buf[(i-1)*W + j] +  1*recv_buf[(i-1)*W + (j+1)] +
					 0*recv_buf[(i+0)*W + (j-1)] +  0*recv_buf[(i+0)*W + j] +  0*recv_buf[(i+0)*W + (j+1)] +
					-1*recv_buf[(i+1)*W + (j-1)] + -2*recv_buf[(i+1)*W + j] + -1*recv_buf[(i+1)*W + (j+1)];
		for (i = 1; i < H_new; i++)
			for (j = 1; j < W-1; j++) {
				int x = gx[i*W + j];
				int y = gy[i*W + j];
				recv_buf[i*W + j] = sqrt(x*x + y*y);
			}
	}
	else if(rank==num_cores-1)
	{
		for (i = 1; i < H_new+remainder; i++)
			for (j = 1; j < W-1; j++)
				gx[i*W + j] =
					1*recv_buf[(i-1)*W + (j-1)] + 0*recv_buf[(i-1)*W + j] + -1*recv_buf[(i-1)*W + (j+1)] +
					2*recv_buf[(i+0)*W + (j-1)] + 0*recv_buf[(i+0)*W + j] + -2*recv_buf[(i+0)*W + (j+1)] +
					1*recv_buf[(i+1)*W + (j-1)] + 0*recv_buf[(i+1)*W + j] + -1*recv_buf[(i+1)*W + (j+1)];
		for (i = 1; i < H_new+remainder; i++)
			for (j = 1; j < W-1; j++)
				gy[i*W + j] =
					 1*recv_buf[(i-1)*W + (j-1)] +  2*recv_buf[(i-1)*W + j] +  1*recv_buf[(i-1)*W + (j+1)] +
					 0*recv_buf[(i+0)*W + (j-1)] +  0*recv_buf[(i+0)*W + j] +  0*recv_buf[(i+0)*W + (j+1)] +
					-1*recv_buf[(i+1)*W + (j-1)] + -2*recv_buf[(i+1)*W + j] + -1*recv_buf[(i+1)*W + (j+1)];
		for (i = 1; i < H_new+remainder; i++)
			for (j = 1; j < W-1; j++) {
				int x = gx[i*W + j];
				int y = gy[i*W + j];
				recv_buf[i*W + j] = sqrt(x*x + y*y);
			}
	}
	else
	{
		for (i = 1; i < H_new+1; i++)
			for (j = 1; j < W-1; j++)
				gx[i*W + j] =
					1*recv_buf[(i-1)*W + (j-1)] + 0*recv_buf[(i-1)*W + j] + -1*recv_buf[(i-1)*W + (j+1)] +
					2*recv_buf[(i+0)*W + (j-1)] + 0*recv_buf[(i+0)*W + j] + -2*recv_buf[(i+0)*W + (j+1)] +
					1*recv_buf[(i+1)*W + (j-1)] + 0*recv_buf[(i+1)*W + j] + -1*recv_buf[(i+1)*W + (j+1)];
		for (i = 1; i < H_new+1; i++)
			for (j = 1; j < W-1; j++)
				gy[i*W + j] =
					 1*recv_buf[(i-1)*W + (j-1)] +  2*recv_buf[(i-1)*W + j] +  1*recv_buf[(i-1)*W + (j+1)] +
					 0*recv_buf[(i+0)*W + (j-1)] +  0*recv_buf[(i+0)*W + j] +  0*recv_buf[(i+0)*W + (j+1)] +
					-1*recv_buf[(i+1)*W + (j-1)] + -2*recv_buf[(i+1)*W + j] + -1*recv_buf[(i+1)*W + (j+1)];
		for (i = 1; i < H_new+1; i++)
			for (j = 1; j < W-1; j++) {
				int x = gx[i*W + j];
				int y = gy[i*W + j];
				recv_buf[i*W + j] = sqrt(x*x + y*y);
			}
	}
    
    if(rank==0)
    {
        TIME_GET(&end);
    }
    
    int *recv_count = malloc(sizeof(int)*num_cores);
	int *disp_count_r = malloc(sizeof(int)*num_cores);
	int index;
	int size2;
	
	for(int i=0; i<num_cores; i++)
	{
		
		if(i == 0)
		{
			recv_count[i] = W*(H_new);
			disp_count_r[i] = 0;
		}
		else if(i == num_cores-1)
		{
			recv_count[i] = W*(H_new+remainder);
			disp_count_r[i] = W*(H_new)*i;
		}
		else
		{
			recv_count[i] = W*(H_new);
			disp_count_r[i] = W*(H_new)*i;
		}	
		
	}
	
	if(rank == 0)
		index=0;
	else
		index=W;
	
	if(rank==num_cores-1)
		size2 = W*(H_new+remainder);
	else
		size2 = W*(H_new);
    
    if(rank==0)
    {
        TIME_GET(&recv_start);
    }
    
    MPI_Gatherv(&recv_buf[index], size2, MPI_INT, buf, recv_count, disp_count_r, MPI_INT, 0, MPI_COMM_WORLD);
    
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
    
    if(rank==0)
    {
        free(buf);
    }
    free(gx);
    free(gy);
    free(recv_buf);
    
    return runtime;
}

double sobel_spfp_mpi(int size, int threads)
{
    int rank, num_cores;
    MPI_Comm_size(MPI_COMM_WORLD, &num_cores);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    srand(SEED);
    int i,j;
    TIME_STRUCT start,end;
    TIME_STRUCT send_start,send_end;
    TIME_STRUCT recv_start,recv_end;
    double runtime=0.0;
    double sendtime=0.0;
    double recvtime=0.0;
    double calctime=0.0;
    
    float *buf;
    float *gx;
    float *gy;
    float *recv_buf;
    int recv_size;
    
    int W=size;
    int H=size;
    
    if(rank==0)
    {
        buf=malloc(sizeof(float)*(size*size));
        for(i=0;i<(W*H);i++)
        {
            buf[i]=(int32_t)rand();
        }
    }
    
    int H_new;
    int remainder;
    
    H_new=floor(H/num_cores);
    
    remainder=H-H_new*num_cores;
    
    int *send_count = malloc(sizeof(int)*num_cores);
    int *disp_count = malloc(sizeof(int)*num_cores);
    
    for(i=0; i<num_cores; i++)
    {
        if(i==0)
        {
            send_count[i] = W*(H_new+1);
            disp_count[i] = 0;
        }
        else if(i==num_cores-1)
        {
            send_count[i] = W*(H_new+1+remainder);
            disp_count[i] = W*H_new*i - W;
        }
        else
        {
            send_count[i] = W*(H_new+2);
            disp_count[i] = W*H_new*i - W;
        }
    }
    
    if(rank == 0)
	{
		recv_size = W*(H_new+1);
		recv_buf = malloc(recv_size*sizeof(float));
		gx = malloc(W * H_new * sizeof(float));
		gy = malloc(W * H_new * sizeof(float));
	}
	else if(rank==num_cores-1)
	{
		recv_size = W*(H_new+1+remainder);
		recv_buf = malloc(recv_size*sizeof(float));
		gx = malloc(W * (H_new+remainder) * sizeof(float));
		gy = malloc(W * (H_new+remainder) * sizeof(float));
	}
	else
	{
		recv_size = W*(H_new+2);
		recv_buf = malloc(recv_size*sizeof(float));
		gx = malloc(W * (H_new+1) * sizeof(float));
		gy = malloc(W * (H_new+1) * sizeof(float));
	}
    
    if(rank==0)
    {
        TIME_GET(&send_start);
    }
    
    MPI_Scatterv(buf, send_count, disp_count, MPI_FLOAT, recv_buf, recv_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
    
    if(rank==0)
    {
        TIME_GET(&send_end);
        TIME_GET(&start);
    }
    
    if(rank==0)
	{
		for (i = 1; i < H_new; i++)
			for (j = 1; j < W-1; j++)
				gx[i*W + j] =
					1*recv_buf[(i-1)*W + (j-1)] + 0*recv_buf[(i-1)*W + j] + -1*recv_buf[(i-1)*W + (j+1)] +
					2*recv_buf[(i+0)*W + (j-1)] + 0*recv_buf[(i+0)*W + j] + -2*recv_buf[(i+0)*W + (j+1)] +
					1*recv_buf[(i+1)*W + (j-1)] + 0*recv_buf[(i+1)*W + j] + -1*recv_buf[(i+1)*W + (j+1)];
		for (i = 1; i < H_new; i++)
			for (j = 1; j < W-1; j++)
				gy[i*W + j] =
					 1*recv_buf[(i-1)*W + (j-1)] +  2*recv_buf[(i-1)*W + j] +  1*recv_buf[(i-1)*W + (j+1)] +
					 0*recv_buf[(i+0)*W + (j-1)] +  0*recv_buf[(i+0)*W + j] +  0*recv_buf[(i+0)*W + (j+1)] +
					-1*recv_buf[(i+1)*W + (j-1)] + -2*recv_buf[(i+1)*W + j] + -1*recv_buf[(i+1)*W + (j+1)];
		for (i = 1; i < H_new; i++)
			for (j = 1; j < W-1; j++) {
				int x = gx[i*W + j];
				int y = gy[i*W + j];
				recv_buf[i*W + j] = sqrt(x*x + y*y);
			}
	}
	else if(rank==num_cores-1)
	{
		for (i = 1; i < H_new+remainder; i++)
			for (j = 1; j < W-1; j++)
				gx[i*W + j] =
					1*recv_buf[(i-1)*W + (j-1)] + 0*recv_buf[(i-1)*W + j] + -1*recv_buf[(i-1)*W + (j+1)] +
					2*recv_buf[(i+0)*W + (j-1)] + 0*recv_buf[(i+0)*W + j] + -2*recv_buf[(i+0)*W + (j+1)] +
					1*recv_buf[(i+1)*W + (j-1)] + 0*recv_buf[(i+1)*W + j] + -1*recv_buf[(i+1)*W + (j+1)];
		for (i = 1; i < H_new+remainder; i++)
			for (j = 1; j < W-1; j++)
				gy[i*W + j] =
					 1*recv_buf[(i-1)*W + (j-1)] +  2*recv_buf[(i-1)*W + j] +  1*recv_buf[(i-1)*W + (j+1)] +
					 0*recv_buf[(i+0)*W + (j-1)] +  0*recv_buf[(i+0)*W + j] +  0*recv_buf[(i+0)*W + (j+1)] +
					-1*recv_buf[(i+1)*W + (j-1)] + -2*recv_buf[(i+1)*W + j] + -1*recv_buf[(i+1)*W + (j+1)];
		for (i = 1; i < H_new+remainder; i++)
			for (j = 1; j < W-1; j++) {
				int x = gx[i*W + j];
				int y = gy[i*W + j];
				recv_buf[i*W + j] = sqrt(x*x + y*y);
			}
	}
	else
	{
		for (i = 1; i < H_new+1; i++)
			for (j = 1; j < W-1; j++)
				gx[i*W + j] =
					1*recv_buf[(i-1)*W + (j-1)] + 0*recv_buf[(i-1)*W + j] + -1*recv_buf[(i-1)*W + (j+1)] +
					2*recv_buf[(i+0)*W + (j-1)] + 0*recv_buf[(i+0)*W + j] + -2*recv_buf[(i+0)*W + (j+1)] +
					1*recv_buf[(i+1)*W + (j-1)] + 0*recv_buf[(i+1)*W + j] + -1*recv_buf[(i+1)*W + (j+1)];
		for (i = 1; i < H_new+1; i++)
			for (j = 1; j < W-1; j++)
				gy[i*W + j] =
					 1*recv_buf[(i-1)*W + (j-1)] +  2*recv_buf[(i-1)*W + j] +  1*recv_buf[(i-1)*W + (j+1)] +
					 0*recv_buf[(i+0)*W + (j-1)] +  0*recv_buf[(i+0)*W + j] +  0*recv_buf[(i+0)*W + (j+1)] +
					-1*recv_buf[(i+1)*W + (j-1)] + -2*recv_buf[(i+1)*W + j] + -1*recv_buf[(i+1)*W + (j+1)];
		for (i = 1; i < H_new+1; i++)
			for (j = 1; j < W-1; j++) {
				int x = gx[i*W + j];
				int y = gy[i*W + j];
				recv_buf[i*W + j] = sqrt(x*x + y*y);
			}
	}
    
    if(rank==0)
    {
        TIME_GET(&end);
    }
    
    int *recv_count = malloc(sizeof(int)*num_cores);
	int *disp_count_r = malloc(sizeof(int)*num_cores);
	int index;
	int size2;
	
	for(int i=0; i<num_cores; i++)
	{
		
		if(i == 0)
		{
			recv_count[i] = W*(H_new);
			disp_count_r[i] = 0;
		}
		else if(i == num_cores-1)
		{
			recv_count[i] = W*(H_new+remainder);
			disp_count_r[i] = W*(H_new)*i;
		}
		else
		{
			recv_count[i] = W*(H_new);
			disp_count_r[i] = W*(H_new)*i;
		}	
		
	}
	
	if(rank == 0)
		index=0;
	else
		index=W;
	
	if(rank==num_cores-1)
		size2 = W*(H_new+remainder);
	else
		size2 = W*(H_new);
    
    if(rank==0)
    {
        TIME_GET(&recv_start);
    }
    
    MPI_Gatherv(&recv_buf[index], size2, MPI_FLOAT, buf, recv_count, disp_count_r, MPI_FLOAT, 0, MPI_COMM_WORLD);
    
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
    
    if(rank==0)
    {
        free(buf);
    }
    free(gx);
    free(gy);
    free(recv_buf);
    
    return runtime;
}

double sobel_dpfp_mpi(int size, int threads)
{
    int rank, num_cores;
    MPI_Comm_size(MPI_COMM_WORLD, &num_cores);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    srand(SEED);
    int i,j;
    TIME_STRUCT start,end;
    TIME_STRUCT send_start,send_end;
    TIME_STRUCT recv_start,recv_end;
    double runtime=0.0;
    double sendtime=0.0;
    double recvtime=0.0;
    double calctime=0.0;
    
    double *buf;
    double *gx;
    double *gy;
    double *recv_buf;
    int recv_size;
    
    int W=size;
    int H=size;
    
    if(rank==0)
    {
        buf=malloc(sizeof(double)*(size*size));
        for(i=0;i<(W*H);i++)
        {
            buf[i]=(int32_t)rand();
        }
    }
    
    int H_new;
    int remainder;
    
    H_new=floor(H/num_cores);
    
    remainder=H-H_new*num_cores;
    
    int *send_count = malloc(sizeof(int)*num_cores);
    int *disp_count = malloc(sizeof(int)*num_cores);
    
    for(i=0; i<num_cores; i++)
    {
        if(i==0)
        {
            send_count[i] = W*(H_new+1);
            disp_count[i] = 0;
        }
        else if(i==num_cores-1)
        {
            send_count[i] = W*(H_new+1+remainder);
            disp_count[i] = W*H_new*i - W;
        }
        else
        {
            send_count[i] = W*(H_new+2);
            disp_count[i] = W*H_new*i - W;
        }
    }
    
    if(rank == 0)
	{
		recv_size = W*(H_new+1);
		recv_buf = malloc(recv_size*sizeof(double));
		gx = malloc(W * H_new * sizeof(double));
		gy = malloc(W * H_new * sizeof(double));
	}
	else if(rank==num_cores-1)
	{
		recv_size = W*(H_new+1+remainder);
		recv_buf = malloc(recv_size*sizeof(double));
		gx = malloc(W * (H_new+remainder) * sizeof(double));
		gy = malloc(W * (H_new+remainder) * sizeof(double));
	}
	else
	{
		recv_size = W*(H_new+2);
		recv_buf = malloc(recv_size*sizeof(double));
		gx = malloc(W * (H_new+1) * sizeof(double));
		gy = malloc(W * (H_new+1) * sizeof(double));
	}
    
    if(rank==0)
    {
        TIME_GET(&send_start);
    }
    
    MPI_Scatterv(buf, send_count, disp_count, MPI_DOUBLE, recv_buf, recv_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    if(rank==0)
    {
        TIME_GET(&send_end);
        TIME_GET(&start);
    }
    
    if(rank==0)
	{
		for (i = 1; i < H_new; i++)
			for (j = 1; j < W-1; j++)
				gx[i*W + j] =
					1*recv_buf[(i-1)*W + (j-1)] + 0*recv_buf[(i-1)*W + j] + -1*recv_buf[(i-1)*W + (j+1)] +
					2*recv_buf[(i+0)*W + (j-1)] + 0*recv_buf[(i+0)*W + j] + -2*recv_buf[(i+0)*W + (j+1)] +
					1*recv_buf[(i+1)*W + (j-1)] + 0*recv_buf[(i+1)*W + j] + -1*recv_buf[(i+1)*W + (j+1)];
		for (i = 1; i < H_new; i++)
			for (j = 1; j < W-1; j++)
				gy[i*W + j] =
					 1*recv_buf[(i-1)*W + (j-1)] +  2*recv_buf[(i-1)*W + j] +  1*recv_buf[(i-1)*W + (j+1)] +
					 0*recv_buf[(i+0)*W + (j-1)] +  0*recv_buf[(i+0)*W + j] +  0*recv_buf[(i+0)*W + (j+1)] +
					-1*recv_buf[(i+1)*W + (j-1)] + -2*recv_buf[(i+1)*W + j] + -1*recv_buf[(i+1)*W + (j+1)];
		for (i = 1; i < H_new; i++)
			for (j = 1; j < W-1; j++) {
				int x = gx[i*W + j];
				int y = gy[i*W + j];
				recv_buf[i*W + j] = sqrt(x*x + y*y);
			}
	}
	else if(rank==num_cores-1)
	{
		for (i = 1; i < H_new+remainder; i++)
			for (j = 1; j < W-1; j++)
				gx[i*W + j] =
					1*recv_buf[(i-1)*W + (j-1)] + 0*recv_buf[(i-1)*W + j] + -1*recv_buf[(i-1)*W + (j+1)] +
					2*recv_buf[(i+0)*W + (j-1)] + 0*recv_buf[(i+0)*W + j] + -2*recv_buf[(i+0)*W + (j+1)] +
					1*recv_buf[(i+1)*W + (j-1)] + 0*recv_buf[(i+1)*W + j] + -1*recv_buf[(i+1)*W + (j+1)];
		for (i = 1; i < H_new+remainder; i++)
			for (j = 1; j < W-1; j++)
				gy[i*W + j] =
					 1*recv_buf[(i-1)*W + (j-1)] +  2*recv_buf[(i-1)*W + j] +  1*recv_buf[(i-1)*W + (j+1)] +
					 0*recv_buf[(i+0)*W + (j-1)] +  0*recv_buf[(i+0)*W + j] +  0*recv_buf[(i+0)*W + (j+1)] +
					-1*recv_buf[(i+1)*W + (j-1)] + -2*recv_buf[(i+1)*W + j] + -1*recv_buf[(i+1)*W + (j+1)];
		for (i = 1; i < H_new+remainder; i++)
			for (j = 1; j < W-1; j++) {
				int x = gx[i*W + j];
				int y = gy[i*W + j];
				recv_buf[i*W + j] = sqrt(x*x + y*y);
			}
	}
	else
	{
		for (i = 1; i < H_new+1; i++)
			for (j = 1; j < W-1; j++)
				gx[i*W + j] =
					1*recv_buf[(i-1)*W + (j-1)] + 0*recv_buf[(i-1)*W + j] + -1*recv_buf[(i-1)*W + (j+1)] +
					2*recv_buf[(i+0)*W + (j-1)] + 0*recv_buf[(i+0)*W + j] + -2*recv_buf[(i+0)*W + (j+1)] +
					1*recv_buf[(i+1)*W + (j-1)] + 0*recv_buf[(i+1)*W + j] + -1*recv_buf[(i+1)*W + (j+1)];
		for (i = 1; i < H_new+1; i++)
			for (j = 1; j < W-1; j++)
				gy[i*W + j] =
					 1*recv_buf[(i-1)*W + (j-1)] +  2*recv_buf[(i-1)*W + j] +  1*recv_buf[(i-1)*W + (j+1)] +
					 0*recv_buf[(i+0)*W + (j-1)] +  0*recv_buf[(i+0)*W + j] +  0*recv_buf[(i+0)*W + (j+1)] +
					-1*recv_buf[(i+1)*W + (j-1)] + -2*recv_buf[(i+1)*W + j] + -1*recv_buf[(i+1)*W + (j+1)];
		for (i = 1; i < H_new+1; i++)
			for (j = 1; j < W-1; j++) {
				int x = gx[i*W + j];
				int y = gy[i*W + j];
				recv_buf[i*W + j] = sqrt(x*x + y*y);
			}
	}
    
    if(rank==0)
    {
        TIME_GET(&end);
    }
    
    int *recv_count = malloc(sizeof(int)*num_cores);
	int *disp_count_r = malloc(sizeof(int)*num_cores);
	int index;
	int size2;
	
	for(int i=0; i<num_cores; i++)
	{
		
		if(i == 0)
		{
			recv_count[i] = W*(H_new);
			disp_count_r[i] = 0;
		}
		else if(i == num_cores-1)
		{
			recv_count[i] = W*(H_new+remainder);
			disp_count_r[i] = W*(H_new)*i;
		}
		else
		{
			recv_count[i] = W*(H_new);
			disp_count_r[i] = W*(H_new)*i;
		}	
		
	}
	
	if(rank == 0)
		index=0;
	else
		index=W;
	
	if(rank==num_cores-1)
		size2 = W*(H_new+remainder);
	else
		size2 = W*(H_new);
    
    if(rank==0)
    {
        TIME_GET(&recv_start);
    }
    
    MPI_Gatherv(&recv_buf[index], size2, MPI_DOUBLE, buf, recv_count, disp_count_r, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
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
    
    if(rank==0)
    {
        free(buf);
    }
    free(gx);
    free(gy);
    free(recv_buf);
    
    return runtime;
}
