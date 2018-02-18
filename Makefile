CC=mpicc

BENCHMARKS += matrix_add.c matrix_sub.c matrix_mult.c matrix_add_mpi.c

OBJECTS = benchmarks.h benchmarks.c $(BENCHMARKS)
FLAGS += -O3 -fopenmp
LDFLAGS += -lm
DEFINES += -DFIXED_SEED

benchmarks:
	$(CC) $(OBJECTS) $(FLAGS) $(DEFINES) $(LIBS) -o $@ $(LDFLAGS)

clean:
	rm benchmarks
