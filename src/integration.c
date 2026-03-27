#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
//2. f(x) function-->for the simpson formula function
double f(double x, int fun_id){
    switch(fun_id){
        case 0:
            return sin(x) + 0.5 * cos(3.0 * x);
        case 1:
            return 1.0/(1.0 + 100.0 * (x - 0.3) * (x - 0.3) );
        case 2:
            return sin(200.0 * x) * exp(-x);
        default:
            return 0.0;
    }
}
//1. simpson formula
double simpson(double a, double b, int fun_id)
{
    double m = (a+b)/2.0;
    double part_a  = (b-a)/6.0;
    double part_b = (f(a,fun_id) + (4*f(m,fun_id)) + f(b,fun_id));
    return part_a*part_b;
}

int main(int argc, char **argv){

    int rank = 0;
    int size = 0;
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len = 0;
    //fun id from the mpi run
    int funId = atoi(argv[1]);
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Get_processor_name(processor_name, &name_len);

    if (rank == 0){
        double val = simpson(0.0, 1.0, funId);
        printf("Simpson estimate = %f/n", val);
    }

    MPI_Finalize();
    return 0;

}
