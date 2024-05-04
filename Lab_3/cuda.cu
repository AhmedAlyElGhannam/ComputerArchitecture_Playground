#include <iostream>
#include <cstdlib>
#include <ctime>
#include <fstream>

using namespace std;

// try N = 5, 50, 500
#define N 2

// available matrix operations
typedef enum
{
    ADD = 1,
    MUL = 2
} matOp;

__global__ void matrixMulCUDA(int* matrixA, int* matrixB, int* matrixRes) 
{
    // Compute each thread's global row and column index
    int rowIndex = blockIdx.y * blockDim.y + threadIdx.y;
    int colIndex = blockIdx.x * blockDim.x + threadIdx.x;

    // Iterate over row, and down column
    matrixRes[rowIndex][colIndex] = 0;
    for (int k = 0; k < N; k++) 
    {
        // Accumulate results for a single element
        matrixRes[rowIndex][colIndex] += matrixA[rowIndex][k] * matrixB[k][colIndex];
    }
}
/**
if condition to see if row && col are less than N
then simply add
*/
__global__ void matrixAddCUDA(int* matrixA, int* matrixB, int* matrixRes) 
{
    // Compute each thread's global row and column index
    int rowIndex = blockIdx.y * blockDim.y + threadIdx.y;
    int colIndex = blockIdx.x * blockDim.x + threadIdx.x;

    // simply add
    matrixRes[rowIndex * N + colIndex] = matrixA[rowIndex * N + colIndex] + matrixB[rowIndex * N + colIndex];
}

// h_sth => host variable (PC)
// d_sth => device variable (GPU)

void matrixOperationCudaWrapper(const int (&h_matrixA)[N][N], const int (&h_matrixB)[N][N], int (&h_matrixRes)[N][N], unsigned char operation)
{
    // create pointers to gpu
    int* d_cudaA = 0;
    int* d_cudaB = 0;
    int* d_cudaRes = 0;

    // defining size
    size_t sizeInBytes = N * N * sizeof(int);

    // allocate memory in gpu
    cudaMalloc((void**)(&d_cudaA), sizeInBytes);
    cudaMalloc((void**)(&d_cudaB), sizeInBytes);
    cudaMalloc((void**)(&d_cudaRes), sizeInBytes);

    // copy vectors into gpu cudaMemcpy(d_input, inputImg.data, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cudaA, &h_matrixA, sizeInBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cudaB, &h_matrixB, sizeInBytes, cudaMemcpyHostToDevice);

    // defining CTA and grid dimensions
    int threads = 16;
    int blocks = (N + threads - 1) / threads;

    // setting up kernel launch parameters
    dim3 BLOCKS(blocks, blocks);
    dim3 THREADS(threads, threads);

    // launch kernel for chosen operation
    if (operation == ADD)
        matrixAddCUDA<<<BLOCKS, THREADS>>>(d_cudaA, d_cudaB, d_cudaRes);
    else if (operation == MUL)
        matrixMulCUDA<<<BLOCKS, THREADS>>>(d_cudaA, d_cudaB, d_cudaRes);
    else
    {
        cout << "yo, what the duck?" << endl;
        return; // do not continue this mess!
    }
        
    // copy result from gpu memory
    cudaMemcpy(&h_matrixRes, d_cudaRes, sizeInBytes, cudaMemcpyDeviceToHost);

    // free allocated gpu memory
    cudaFree(d_cudaA);
    cudaFree(d_cudaB);
    cudaFree(d_cudaRes);

    return;
}

void populateMatrix(int (&matrix)[N][N]) 
{
    for (int i = 0; i < N; i++) 
    {
        for (int j = 0; j < N; j++)
        {
            matrix[i][j] = rand() % 100; // Generate random numbers between 0 and N-1
        }
    }
}

void printMatrix(const int (&matrix)[N][N]) 
{
    for (int i = 0; i < N; i++) 
    {
        for (int j = 0; j < N; j++)
        {
            cout << matrix[i][j] << "\t";
        }
        cout << endl;
    }
}

void csvMatrix(const int (&matrix)[N][N], const char *filename) {
    std::ofstream file(filename);
    if (!file.is_open()) 
    {
        std::cerr << "Error opening file " << filename << std::endl;
        return;
    }

    for (int i = 0; i < N; i++) 
    {
        for (int j = 0; j < N; j++) 
        {
            file << matrix[i][j];
            if (j < N - 1) 
            {
                file << ",";
            }
        }
        file << "\n";
    }
    file.close();
}

int main() 
{
    // random number generation shenanigans
    srand(time(NULL));

    // define matrices
    int matA[N][N];
    int matB[N][N];
    int matC[N][N];
    int matRes[N][N];
    int matTemp[N][N];

    // populate matrix A && B
    populateMatrix(matA);
    populateMatrix(matB);
    populateMatrix(matC);

    // output matrix A && B as csv files for references
    csvMatrix(matA, "MatrixA.csv");
    csvMatrix(matB, "MatrixB.csv");
    csvMatrix(matC, "MatrixC.csv");
    
    // // C * ((A * B) + (B * A))
    // multMatrix(matA, matB, matRes); // A * B = res
    
    // multMatrix(matB, matA, matTemp); // B * A = temp
    
    // addMatrix(matRes, matTemp, matTemp); // res + temp = temp

    // multMatrix(matC, matTemp, matRes); // C * temp = res
    
    // // output matrix C result for reference
    // csvMatrix(matRes, "Result.csv");

    return 0;
}