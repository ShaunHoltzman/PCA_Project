#include "benchmarks.h"

int main(int argc, char **argv)
{
    int rank, num_cores;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_cores);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    if( (argc == 2 && strcmp(argv[1], "--help")==0) || argc != 5)
    {
        if(rank==0)
        {
        printf("\nUsage:\n");
        printf("mpiexec -n [num_cores] ./benchmarks [OpenMP/OpenMPI] [benchmark] [datatype] [size]\n\n");
        printf("OpenMP/OpenMPI:\n");
        printf("1: OpenMP\n");
        printf("2: OpenMPI\n");
        printf("*Note: if num_cores==0, serial calculations will be done\n\n");
        printf("Benchmarks:\n");
        printf("1: matrix_matrix_mult\n");
        printf("2: matrix_matrix_add\n");
        printf("3: matrix_matrix_sub\n");
        printf("4: matrix_matrix_trans\n");
        printf("5: keplers_equation\n");
        printf("6: clohessy_wiltshire_equations\n");
        printf("7: sobel_filter\n\n");
        printf("Datatypes:\n");
        printf("1: int8\n");
        printf("2: int16\n");
        printf("3: int32\n");
        printf("4: SPFP\n");
        printf("5: DPFP\n\n");
        }
        MPI_Finalize();
        exit(0);
    }
    
    int mp_or_mpi=atoi(argv[1]);
    int benchmark=atoi(argv[2]);
    int data_type=atoi(argv[3]);
    int size=atoi(argv[4]);
    
    double runtime;
    
    const char *benchmarks[7]={"matrix_matrix_mult",
                               "matrix_matrix_add",
                               "matrix_matrix_sub",
                               "matrix_matrix_trans",
                               "keplers_equation",
                               "clohessy_wiltshire_equations",
                               "sobel_filter"
                               };
    
    const char *data_types[5]={"int8","int16","int32","spfp","dpfp"};
    
    char *parallel_type;
    
    if(num_cores==1)
    {
        parallel_type = "Serial";
    }
    else if(mp_or_mpi==1)
    {
        parallel_type = "OpenMP";
    }
    else
    {
        parallel_type = "OpenMPI";
    }
    
    
    if(benchmark==1)
    {
        if( (num_cores==1 || mp_or_mpi==1) && rank==0)
        {
            switch(data_type)
            {
                case(1):
                    runtime=matrix_matrix_mult_int8(size, num_cores);
                    break;
                case(2):
                    runtime=matrix_matrix_mult_int16(size, num_cores);
                    break;
                case(3):
                    runtime=matrix_matrix_mult_int32(size, num_cores);
                    break;
                case(4):
                    runtime=matrix_matrix_mult_spfp(size, num_cores);
                    break;
                case(5):
                    runtime=matrix_matrix_mult_dpfp(size, num_cores);
                    break;
            }
        }
        else if(mp_or_mpi==2)
        {
            switch(data_type)
            {
                case(1):
                    runtime=matrix_matrix_mult_int8_mpi(size, num_cores);
                    break;
                case(2):
                    runtime=matrix_matrix_mult_int16_mpi(size, num_cores);
                    break;
                case(3):
                    runtime=matrix_matrix_mult_int32_mpi(size, num_cores);
                    break;
                case(4):
                    runtime=matrix_matrix_mult_spfp_mpi(size, num_cores);
                    break;
                case(5):
                    runtime=matrix_matrix_mult_dpfp_mpi(size, num_cores);
                    break;
            }
        }
    }
    else if(benchmark==2)
    {
        if( (num_cores==1 || mp_or_mpi==1) && rank==0)
        {
            switch(data_type)
            {
                case(1):
                    runtime=matrix_matrix_add_int8(size, num_cores);
                    break;
                case(2):
                    runtime=matrix_matrix_add_int16(size, num_cores);
                    break;
                case(3):
                    runtime=matrix_matrix_add_int32(size, num_cores);
                    break;
                case(4):
                    runtime=matrix_matrix_add_spfp(size, num_cores);
                    break;
                case(5):
                    runtime=matrix_matrix_add_dpfp(size, num_cores);
                    break;
            }
        }
        else if(mp_or_mpi==2)
        {
            switch(data_type)
            {
                case(1):
                    runtime=matrix_matrix_add_int8_mpi(size, num_cores);
                    break;
                case(2):
                    runtime=matrix_matrix_add_int16_mpi(size, num_cores);
                    break;
                case(3):
                    runtime=matrix_matrix_add_int32_mpi(size, num_cores);
                    break;
                case(4):
                    runtime=matrix_matrix_add_spfp_mpi(size, num_cores);
                    break;
                case(5):
                    runtime=matrix_matrix_add_dpfp_mpi(size, num_cores);
                    break;
            }
        }
        
    }
    else if(benchmark==3)
    {
        if( (num_cores==1 || mp_or_mpi==1) && rank==0)
        {
            switch(data_type)
            {
                case(1):
                    runtime=matrix_matrix_sub_int8(size, num_cores);
                    break;
                case(2):
                    runtime=matrix_matrix_sub_int16(size, num_cores);
                    break;
                case(3):
                    runtime=matrix_matrix_sub_int32(size, num_cores);
                    break;
                case(4):
                    runtime=matrix_matrix_sub_spfp(size, num_cores);
                    break;
                case(5):
                    runtime=matrix_matrix_sub_dpfp(size, num_cores);
                    break;
            }
        }
        else if(mp_or_mpi==2)
        {
            switch(data_type)
            {
                case(1):
                    runtime=matrix_matrix_sub_int8_mpi(size, num_cores);
                    break;
                case(2):
                    runtime=matrix_matrix_sub_int16_mpi(size, num_cores);
                    break;
                case(3):
                    runtime=matrix_matrix_sub_int32_mpi(size, num_cores);
                    break;
                case(4):
                    runtime=matrix_matrix_sub_spfp_mpi(size, num_cores);
                    break;
                case(5):
                    runtime=matrix_matrix_sub_dpfp_mpi(size, num_cores);
                    break;
            }
        }
    }
    else if(benchmark==4)
    {
        if( (num_cores==1 || mp_or_mpi==1) && rank==0)
        {
            switch(data_type)
            {
                case(1):
                    runtime=matrix_matrix_trans_int8(size, num_cores);
                    break;
                case(2):
                    runtime=matrix_matrix_trans_int16(size, num_cores);
                    break;
                case(3):
                    runtime=matrix_matrix_trans_int32(size, num_cores);
                    break;
                case(4):
                    runtime=matrix_matrix_trans_spfp(size, num_cores);
                    break;
                case(5):
                    runtime=matrix_matrix_trans_dpfp(size, num_cores);
                    break;
            }
        }
        else if(mp_or_mpi==2)
        {
            switch(data_type)
            {
                case(1):
                    runtime=matrix_matrix_trans_int8_mpi(size, num_cores);
                    break;
                case(2):
                    runtime=matrix_matrix_trans_int16_mpi(size, num_cores);
                    break;
                case(3):
                    runtime=matrix_matrix_trans_int32_mpi(size, num_cores);
                    break;
                case(4):
                    runtime=matrix_matrix_trans_spfp_mpi(size, num_cores);
                    break;
                case(5):
                    runtime=matrix_matrix_trans_dpfp_mpi(size, num_cores);
                    break;
            }
        }
    }
    else if(benchmark==5)
    {
        if( (num_cores==1 || mp_or_mpi==1) && rank==0)
        {
            switch(data_type)
            {
                case(4):
                    runtime=keplers_spfp(size, num_cores);
                    break;
                case(5):
                    runtime=keplers_dpfp(size, num_cores);
                    break;
            }
        }
        else if(mp_or_mpi==2)
        {
            switch(data_type)
            {
                case(4):
                    runtime=keplers_spfp_mpi(size, num_cores);
                    break;
                case(5):
                    runtime=keplers_dpfp_mpi(size, num_cores);
                    break;
            }
        }
    }
    else if(benchmark==6)
    {
        if( (num_cores==1 || mp_or_mpi==1) && rank==0)
        {
            switch(data_type)
            {
                case(4):
                    runtime=cw_spfp(size, num_cores);
                    break;
                case(5):
                    runtime=cw_dpfp(size, num_cores);
                    break;
            }
        }
        else if(mp_or_mpi==2)
        {
            switch(data_type)
            {
                case(4):
                    runtime=cw_spfp_mpi(size, num_cores);
                    break;
                case(5):
                    runtime=cw_dpfp_mpi(size, num_cores);
                    break;
            }
        }
    }
    else if(benchmark==7)
    {
        if( (num_cores==1 || mp_or_mpi==1) && rank==0)
        {
            switch(data_type)
            {
                case(1):
                    runtime=sobel_int8(size, num_cores);
                    break;
                case(2):
                    runtime=sobel_int16(size, num_cores);
                    break;
                case(3):
                    runtime=sobel_int32(size, num_cores);
                    break;
                case(4):
                    runtime=sobel_spfp(size, num_cores);
                    break;
                case(5):
                    runtime=sobel_dpfp(size, num_cores);
                    break;
            }
        }
        else if(mp_or_mpi==2)
        {
            switch(data_type)
            {
                case(1):
                    runtime=sobel_int8_mpi(size, num_cores);
                    break;
                case(2):
                    runtime=sobel_int16_mpi(size, num_cores);
                    break;
                case(3):
                    runtime=sobel_int32_mpi(size, num_cores);
                    break;
                case(4):
                    runtime=sobel_spfp_mpi(size, num_cores);
                    break;
                case(5):
                    runtime=sobel_dpfp_mpi(size, num_cores);
                    break;
            }
        }
    }
    
    
    
    if(rank==0)
    {
        printf("%s, %s, %s, size=%d, threads=%d, runtime=%f\n\n", benchmarks[benchmark-1], parallel_type,data_types[data_type-1],size,num_cores,runtime);
    }
    
    
    
    
    MPI_Finalize();
    
}
