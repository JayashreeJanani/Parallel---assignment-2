#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
//1.2. f(x) function-->for the simpson formula function
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
//===========================================================================
//1.1. simpson formula
double simpson(double a, double b, int fun_id)
{
    double m = (a+b)/2.0;
    double part_a  = (b-a)/6.0;
    double part_b = (f(a,fun_id) + (4*f(m,fun_id)) + f(b,fun_id));
    return part_a*part_b;
}
//============================================================================
//2.1 adaptive simpson function
/**********************
Arguments passed:

1. a, b------>integration limits
2. tol ------>allowed error(tolerance point)
3. fun_id---->which function to integrate
4. accepted-->counts how many intervals passed tolerance

*************************/
double adaptive_simpson(double a, double b, double tol, int fun_id, int *accepted){

    double m = (a+b) /2.0;
    double S = simpson(a, b, fun_id);//s(a,b)
    double S1 = simpson(a,m,fun_id);//S(a,m)
    double S2 = simpson(m,b,fun_id);//s(m,b)
    /*
    Checking error estimates:
    1. S(refined) = S(a,m)+S(m,b)
    2. error estimate = |S(refined)-S(a,b)|
     */
    double cummulative_simpson = S1+S2-S;
    double error = fabs(cummulative_simpson);

    if( error < 15 * tol){
        (*accepted)++;
        return (S1 + S2 + cummulative_simpson) / 15.0;
    }

    return adaptive_simpson(a,m, tol/2.0, fun_id, accepted)+ adaptive_simpson(m,b, tol/2.0, fun_id, accepted);

}

int main(int argc, char **argv){
    MPI_Init(&argc, &argv);

        int rank = 0;
        int size = 0;
        char processor_name[MPI_MAX_PROCESSOR_NAME];
        int name_len = 0;
        //fun id from the mpi run
        int funId = atoi(argv[1]);
        
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
