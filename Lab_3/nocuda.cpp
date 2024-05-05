#include <iostream>
#include <cstdlib>
#include <ctime>
#include <fstream>

using namespace std;
// 50, 100, 500, 1000, 2000, and 5000.
#define N 4500 // matrix row/col

void populateMatrix(unsigned int (&matrix)[N][N]) 
{
    for (int i = 0; i < N; i++) 
    {
        for (int j = 0; j < N; j++)
        {
            matrix[i][j] = rand() % 50; // Generate random numbers between 0 and 49 (inclusive)
        }
    }
}

void printMatrix(const unsigned int (&matrix)[N][N]) 
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

void addMatrix(const unsigned int (&matrixA)[N][N], const unsigned int (&matrixB)[N][N], unsigned int (&matrixC)[N][N])
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            matrixC[i][j] = matrixA[i][j] + matrixB[i][j];
        }
    }
}

void multMatrix(const unsigned int (&matrixA)[N][N], const unsigned int (&matrixB)[N][N], unsigned int (&matrixC)[N][N])
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            matrixC[i][j] = 0;
            for (int k = 0; k < N; k++)
            {
                matrixC[i][j] += matrixA[i][k] * matrixB[k][j];
            }
        }
    }
}

void csvMatrix(const unsigned int (&matrix)[N][N], const char *filename) {
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
    unsigned int matA[N][N];
    unsigned int matB[N][N];
    unsigned int matC[N][N];
    unsigned int matRes[N][N];
    unsigned int matTemp[N][N];

    // populate matrix A && B
    populateMatrix(matA);
    populateMatrix(matB);
    populateMatrix(matC);

    // output matrix A && B as csv files for references
    csvMatrix(matA, "MatrixA.csv");
    csvMatrix(matB, "MatrixB.csv");
    csvMatrix(matC, "MatrixC.csv");
    
    // C * ((A * B) + (B * A))
    multMatrix(matA, matB, matRes); // A * B = res
    
    multMatrix(matB, matA, matTemp); // B * A = temp
    
    addMatrix(matRes, matTemp, matTemp); // res + temp = temp

    multMatrix(matC, matTemp, matRes); // C * temp = res
    
    // output matrix C result for reference
    csvMatrix(matRes, "Result.csv");

    return 0;
}