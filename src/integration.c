#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
//Assignment 3=> combo of Assignment 2 and OpenMP
#include <omp.h>
//tags
#define TAG_REQUEST 1
#define TAG_WORK 2
#define TAG_RESULT 3
// #define TAG_NEWTASK 4
#define TAG_STOP 4

#define MAX_TASKS 100000
//============================================================================

typedef struct{
    double left;
    double right;
    double tol;
}Task;
//============================================================================
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
//================================================
//serial part

void run_serial(int fun_id, double tol)
{
    int accepted = 0;
    double start = MPI_Wtime();
    double result = adaptive_simpson(0.0, 1.0, tol, fun_id, &accepted);
    double end = MPI_Wtime();

    printf("Mode 0: Serial Baseline\n");
    printf("Function ID          :%d\n", fun_id);
    printf("Tolerance            :%10g\n",tol);
    printf("Integral Result      :%.12f\n",result);
    printf("Accepted Intervals   :%d\n",accepted);
    printf("Runtime (seconds)    :%.6f\n",end-start);
}

//==============================================================
//static part

void run_static(int rank, int size, int fun_id, double tol)
{
    int K=64;
    double h = 1.0/K;

    double local_sum =0.0;
    int local_accepted = 0;

    double start = MPI_Wtime();

    for(int i = rank; i < K; i+=size){

        double a = i * h;
        double b = (i + 1) * h;

        local_sum += adaptive_simpson(a,b,tol/K, fun_id, &local_accepted);
    }

    double global_sum = 0.0;
    int global_accepted = 0;

    MPI_Reduce(&local_sum,&global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_accepted, &global_accepted, 1, MPI_INT,MPI_SUM, 0, MPI_COMM_WORLD);

    double end = MPI_Wtime();

    if(rank == 0){

        printf("Mode 1: Static MPI Decomposition \n");
        printf("Function ID            :%d\n", fun_id);
        printf("Tolerance              :%.10g\n", tol);
        printf("K                      :%d\n",K);
        printf("Integral Result        :%.12f\n", global_sum);
        printf("Accepted Intervals     :%d\n",global_accepted);
        printf("Runtime (seconds)      :%.6f\n",end-start);
    }

}
//==============================================================
int process_task(Task *task, int fun_id, double *result, Task *new_task, int *accepted){
    double a = task->left;
    double b = task->right;
    double tol = task->tol;
    double m  = (a+b) /2.0;

    double S = simpson(a, b, fun_id);//s(a,b)
    double S1 = simpson(a,m,fun_id);//S(a,m)
    double S2 = simpson(m,b,fun_id);//s(m,b)

    double error = fabs(S1 + S2 - S);

    if(error <15.0 *tol)
    {
        *accepted =1;
        *result = (S1 + S2 + (S1 + S2 - S)) / 15.0;
        return 1; //checks whether task is accepted
    }

    *accepted = 0;

    task->right = m;
    task->tol = tol / 2.0;

    new_task->left = m;
    new_task->right = b;
    new_task->tol = tol / 2.0;

    *result = 0.0;

    return 0;
    
}
//=============================================================
void run_dynamic(int rank, int size, int func_id, double tol)
{
    Task queue[MAX_TASKS];
    int front = 0, rear = 0;

    double total_result = 0.0;
    int total_accepted = 0;

    int workers_done = 0;
    int tasks_sent = 0;
    int tasks_completed = 0;

    int K = 64;
    double h = 1.0 / K;
     
    double start = MPI_Wtime();

    if (rank == 0)
    {
        // create initial coarse tasks
        for (int i = 0; i < K; i++)
        {
            queue[rear].left = i * h;
            queue[rear].right = (i + 1) * h;
            queue[rear].tol = tol / K;
            rear++;
        }

        while (workers_done < size - 1)
        {
            MPI_Status status;
            MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            int worker = status.MPI_SOURCE;

            if (status.MPI_TAG == TAG_REQUEST)
            {
                MPI_Recv(NULL, 0, MPI_CHAR, worker, TAG_REQUEST, MPI_COMM_WORLD, &status);

                if (front < rear)
                {
                    MPI_Send(&queue[front], sizeof(Task), MPI_BYTE, worker, TAG_WORK, MPI_COMM_WORLD);
                    front++;
                    tasks_sent++;
                }
                else
                {
                    MPI_Send(NULL, 0, MPI_CHAR, worker, TAG_STOP, MPI_COMM_WORLD);
                    workers_done++;
                }
            }
            else if (status.MPI_TAG == TAG_RESULT)
            {
                double data[2];
                MPI_Recv(data, 2, MPI_DOUBLE, worker, TAG_RESULT, MPI_COMM_WORLD, &status);

                total_result += data[0];
                total_accepted += (int)data[1];
                tasks_completed++;
            }
        }

        double end = MPI_Wtime();

        printf("Mode 2: Dynamic MPI Master/Worker\n");
        printf("Function ID        : %d\n", func_id);
        printf("Tolerance          : %.10g\n", tol);
        printf("K                  : %d\n", K);
        printf("Integral Result    : %.12f\n", total_result);
        printf("Accepted Intervals : %d\n", total_accepted);
        printf("Runtime (seconds)  : %.6f\n", end - start);
    }
    else
    {
        while (1)
        {
            MPI_Send(NULL, 0, MPI_CHAR, 0, TAG_REQUEST, MPI_COMM_WORLD);

            MPI_Status status;
            MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            if (status.MPI_TAG == TAG_STOP)
            {
                MPI_Recv(NULL, 0, MPI_CHAR, 0, TAG_STOP, MPI_COMM_WORLD, &status);
                break;
            }

            Task task;
            MPI_Recv(&task, sizeof(Task), MPI_BYTE, 0, TAG_WORK, MPI_COMM_WORLD, &status);

            int accepted = 0;
            double result = adaptive_simpson(task.left, task.right, task.tol, func_id, &accepted);

            double data[2];
            data[0] = result;
            data[1] = (double)accepted;

            MPI_Send(data, 2, MPI_DOUBLE, 0, TAG_RESULT, MPI_COMM_WORLD);
        }
    }
}
//==============================================================
//Assignment 3
double adaptive_simpson_omp(double a, double b, double tol, int fun_id, int *accepted){
    double m = (a+b) /2.0;
    double S = simpson(a, b, fun_id);//s(a,b)
    double S1 = simpson(a,m,fun_id);//S(a,m)
    double S2 = simpson(m,b,fun_id);//s(m,b)

    double cummulative_simpson = S1+S2-S;
    double error = fabs(cummulative_simpson);

    if( error < 15 * tol){
        (*accepted)++;
        return (S1 + S2 + cummulative_simpson) / 15.0;
    }

    double left_result = 0.0;
    double right_result = 0.0;

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            left_result = adaptive_simpson_omp(a,m, tol/2.0, fun_id, accepted);
        }
        #pragma omp section
        {
            right_result = adaptive_simpson_omp(m,b, tol/2.0, fun_id, accepted);
        }
    }

    return left_result + right_result;
}
//=============================================================
void run_hybrid(int rank, int size, int func_id, double tol)
{
    int K = 64;
    double h = 1.0 / K;

    double local_sum = 0.0;
    int local_accepted = 0;

    double global_sum = 0.0;
    int global_accepted = 0;

    double start = MPI_Wtime();

    for (int i = rank; i < K; i += size)
    {
        double a = i * h;
        double b = (i + 1) * h;

        double interval_result = 0.0;
        int interval_accepted = 0;

        #pragma omp parallel
        {
            #pragma omp single
            {
                interval_result = adaptive_simpson_omp(a, b, tol / K, func_id, &interval_accepted);
            }
        }

        local_sum += interval_result;
        local_accepted += interval_accepted;
    }

    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_accepted, &global_accepted, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    double end = MPI_Wtime();

    if (rank == 0)
    {
        printf("Mode 2: Hybrid MPI/OpenMP\n");
        printf("Function ID            : %d\n", func_id);
        printf("Tolerance              : %.10g\n", tol);
        printf("K                      : %d\n", K);
        printf("Integral Result        : %.12f\n", global_sum);
        printf("Accepted Intervals     : %d\n", global_accepted);
        printf("Runtime (seconds)      : %.6f\n", end - start);
    }
}
//====================================================================================================================

int main(int argc, char **argv){
     MPI_Init(&argc, &argv);

        int rank = 0;
        int size = 0;
        char processor_name[MPI_MAX_PROCESSOR_NAME];
        int name_len = 0;
        int funId = 0;
        int mode = 0;
        double tol = 0.0;

        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        MPI_Get_processor_name(processor_name, &name_len);

        if (argc < 4) {
            if (rank == 0) {
                fprintf(stderr, "Usage: %s <function_id> <mode> <tolerance>\n", argv[0]);
                fprintf(stderr, "Example: %s 0 0 1e-6\n", argv[0]);
            }
            MPI_Finalize();
            return 1;
        }

        // fun id, execution mode, and tolerance from the command line
        funId = atoi(argv[1]);
        mode = atoi(argv[2]);
        tol = atof(argv[3]);

        if(mode == 0){
            if (rank == 0){
            run_serial(funId, tol);
            }
        }
        else if(mode == 1)
        {
            run_static(rank, size, funId, tol);
        }
        else if(mode == 2)
        {
            run_dynamic(rank, size, funId, tol);
        }
        

    MPI_Finalize();

    fflush(stdout);

}
