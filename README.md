Compiling
-----------

make

Running
-----------

For help:

mpiexec -n [any number] ./benchmarks —help

For running benchmarks:

mpiexec -n [num_cores] ./benchmarks [OpenMP/OpenMPI] [benchmark] [datatype] [size]

OpenMP/OpenMPI:
1: OpenMP
2: OpenMPI
*Note: if num_cores==0, serial calculations will be done

Benchmarks:
1: matrix_matrix_mult
2: matrix_matrix_add
3: matrix_matrix_sub

Datatypes:
1: int8
2: int16
3: int32
4: SPFP
5: DPFP

