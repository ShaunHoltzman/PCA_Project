#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>


#define TIME_STRUCT struct timeval
#define TIME_GET(time) gettimeofday((time),NULL)
#define TIME_DOUBLE(time) ((time).tv_sec+1E-6*(time).tv_usec)
#define TIME_RUNTIME(start,end) TIME_DOUBLE(end)-TIME_DOUBLE(start)

//select random seed method
#ifdef FIXED_SEED
	//fixed
	#define SEED 376134299
#else
	//based on system time
	#define SEED time(NULL)
#endif

/* Matrix-Matrix Multiplication */

double matrix_matrix_mult_int8(int size, int threads);
double matrix_matrix_mult_int16(int size, int threads);
double matrix_matrix_mult_int32(int size, int threads);
double matrix_matrix_mult_spfp(int size, int threads);
double matrix_matrix_mult_dpfp(int size, int threads);

double matrix_matrix_mult_int8_mpi(int size, int threads);
double matrix_matrix_mult_int16_mpi(int size, int threads);
double matrix_matrix_mult_int32_mpi(int size, int threads);
double matrix_matrix_mult_spfp_mpi(int size, int threads);
double matrix_matrix_mult_dpfp_mpi(int size, int threads);

/* Matrix-Matrix Addition */

double matrix_matrix_add_int8(int size, int threads);
double matrix_matrix_add_int16(int size, int threads);
double matrix_matrix_add_int32(int size, int threads);
double matrix_matrix_add_spfp(int size, int threads);
double matrix_matrix_add_dpfp(int size, int threads);

double matrix_matrix_add_int8_mpi(int size, int threads);
double matrix_matrix_add_int16_mpi(int size, int threads);
double matrix_matrix_add_int32_mpi(int size, int threads);
double matrix_matrix_add_spfp_mpi(int size, int threads);
double matrix_matrix_add_dpfp_mpi(int size, int threads);

/* Matrix-Matrix Subtraction */

double matrix_matrix_sub_int8(int size, int threads);
double matrix_matrix_sub_int16(int size, int threads);
double matrix_matrix_sub_int32(int size, int threads);
double matrix_matrix_sub_spfp(int size, int threads);
double matrix_matrix_sub_dpfp(int size, int threads);

double matrix_matrix_sub_int8_mpi(int size, int threads);
double matrix_matrix_sub_int16_mpi(int size, int threads);
double matrix_matrix_sub_int32_mpi(int size, int threads);
double matrix_matrix_sub_spfp_mpi(int size, int threads);
double matrix_matrix_sub_dpfp_mpi(int size, int threads);
