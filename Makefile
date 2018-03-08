CC=mpicc

BENCHMARKS += matrix_add.c matrix_sub.c matrix_mult.c matrix_add_mpi.c matrix_mult_mpi.c matrix_sub_mpi.c matrix_trans.c matrix_trans_mpi.c keplers.c keplers_mpi.c cw.c cw_mpi.c sobel.c sobel_mpi.c

OBJECTS = benchmarks.h benchmarks.c $(BENCHMARKS)
FLAGS += -O3 -fopenmp -std=c99
LDFLAGS += -lm
DEFINES += -DFIXED_SEED

benchmarks:
	$(CC) $(OBJECTS) $(FLAGS) $(DEFINES) $(LIBS) -o $@ $(LDFLAGS)

clean:
	rm benchmarks
