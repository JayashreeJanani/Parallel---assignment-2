#include <mpi.h>
#include <stdio.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 0;
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len = 0;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Get_processor_name(processor_name, &name_len);

    printf("Hello from rank %d of %d on %s\n", rank, size, processor_name);

    MPI_Finalize();
    return 0;
}
