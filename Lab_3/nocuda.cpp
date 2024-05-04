#include <iostream>
#include <cstdlib>
#include <ctime>
#include <fstream>

using namespace std;

// try N = 5, 50, 500
// a jump from N=1000 to N=2000 yields 10x the time
#define N 2000

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

void printMatrix(int (&matrix)[N][N]) 
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

void addMatrix(int (&matrixA)[N][N], int (&matrixB)[N][N], int (&matrixC)[N][N])
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            matrixC[i][j] = matrixA[i][j] + matrixB[i][j];
        }
    }
}

void multMatrix(int (&matrixA)[N][N], int (&matrixB)[N][N], int (&matrixC)[N][N])
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

void csvMatrix(int (&matrix)[N][N], const char *filename) {
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
    
    // C * ((A * B) + (B * A))
    multMatrix(matA, matB, matRes); // A * B = res
    
    multMatrix(matB, matA, matTemp); // B * A = temp
    
    addMatrix(matRes, matTemp, matTemp); // res + temp = temp

    multMatrix(matC, matTemp, matRes); // C * temp = res
    
    // output matrix C result for reference
    csvMatrix(matRes, "Result.csv");

    return 0;
}