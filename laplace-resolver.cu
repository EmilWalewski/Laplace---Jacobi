
#include <stdio.h>
#include <stdlib.h>
#define THREADMAX 1000



#define ROWS 10
#define COLUMNS 10
#define PRECISION 1e-9
#define RANGE 100


__host__ int checkConvergence(int* arr, int size){

    int sum = 0;

    for(int i=0; i<size; i++)
        sum += arr[i];

    return sum;
            
}

__global__ void jacobi(double* std_vector, double* out_vector, int* er, int size, int blockSize){

    
    int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;
    

    if(globalIndex < size){

    int thIdx = threadIdx.x;
    int row = globalIndex/ROWS;
    int col = (fmod((double)globalIndex/COLUMNS, 1) * COLUMNS) + 1e-9;
    extern __shared__ int redc[];
    double jacob_val = 0;
    
    
    redc[thIdx] = 0;

    
    if(row > 0 && col < COLUMNS-1 && row < ROWS-1 && col > 0){

        jacob_val = 0.25 * (
            std_vector[(row-1)*COLUMNS+col] +
            std_vector[row*COLUMNS+col+1] +
            std_vector[(row+1)*COLUMNS+col] +
            std_vector[row*COLUMNS+col-1]
        );

        out_vector[row*COLUMNS+col] = jacob_val;


        if (fabsf(jacob_val - std_vector[row*COLUMNS+col]) < PRECISION)
            redc[thIdx] = 1;
 
    }

    __syncthreads();

    for (int s = blockSize/2; s>0; s/=2) {
        if (thIdx<s)
            redc[thIdx] += redc[thIdx+s];
        __syncthreads();
    }

    printf(" \n redc[%d] = %d \n", thIdx, redc[thIdx]);


    if(threadIdx.x == 0)
        er[blockIdx.x] = redc[0];       

    }


}

void init(double* m);
void initEr(int* er, int blockAmount);


int main(int aArgc, char* aArgv[])
{

    double *matrix, *j_matrix;
    double *d_matrix, *d_j_matrix;
    int *er, *d_er;

    long int size = ROWS * COLUMNS * sizeof(double);
    int erSize = 0;

    srand(time(NULL));


    matrix = (double*)malloc(size);

    j_matrix = (double*)malloc(size);


    int blockSize = COLUMNS < THREADMAX ? COLUMNS : THREADMAX; 
    int blockAmount = ROWS*(COLUMNS/THREADMAX <= 0 ? 1 : COLUMNS/THREADMAX) + 
                    ceil((ceil(COLUMNS/THREADMAX) - COLUMNS/THREADMAX));

    erSize = sizeof(int) * blockAmount;
    er = (int*)malloc(erSize);

    init(matrix);
    initEr(er, blockAmount);

    printf("\n %d %d \n", blockSize, blockAmount);
    

 
    cudaMalloc((void**) &d_matrix, size);
    cudaMalloc((void**) &d_j_matrix, size);
    cudaMalloc((void**) &d_er, erSize);
    

    cudaMemcpy(d_er, er, erSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrix, matrix, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_j_matrix, matrix, size, cudaMemcpyHostToDevice);

    int counter = 0;
    

    while(checkConvergence(er, blockAmount) <= 0){


        jacobi<<<blockAmount, THREADMAX, blockSize>>>
        (d_matrix, d_j_matrix, d_er, ROWS*COLUMNS, blockSize);
        cudaDeviceSynchronize();

        cudaMemcpy(d_matrix, d_j_matrix, size, cudaMemcpyDeviceToDevice);
        cudaMemcpy(er, d_er, erSize, cudaMemcpyDeviceToHost);
       
        counter++;
    }

    printf("\n Counter = %d \n", counter);


    cudaMemcpy(j_matrix, d_j_matrix, size, cudaMemcpyDeviceToHost);


    printf("\n\n");

    // for(int i=0; i<ROWS; i++){
    //     for(int j=0; j<COLUMNS; j++){
    //         printf("%.2f ", j_matrix[i*COLUMNS+j]);
    //     }
    //     printf("\n");
    // }


    free(matrix);
    free(j_matrix);
    free(er);

    cudaFree(d_matrix);
    cudaFree(d_j_matrix);
    cudaFree(d_er);


    return 0;
}

void init(double* m){

    //read only values
    double r = 0;
    for(int i=0; i<ROWS; i++){
        for(int j=0; j<COLUMNS; j++){

            r = (double)(rand()%RANGE);
            if(i == 0 || j == 0 || j == COLUMNS-1 || i == ROWS-1)
                m[i*COLUMNS+j] = r;
            else
                m[i*COLUMNS+j] = r;

            // e[i*COLUMNS+j] = 0;
            // printf("%f ", m[i*COLUMNS+j]);
        }
        // printf("\n");
    }

}

void initEr(int* er, int blockAmount){

    for(int i=0;i<blockAmount; i++)
        er[i] = 0;
}