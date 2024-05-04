#include <iostream>
#include <fstream>

void save_csv(int rows, int cols, int array[][4], const char *filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file " << filename << std::endl;
        return;
    }

    // Iterate over each element of the array and write it to the file
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            file << array[i][j];
            // Add comma if not the last element in the row
            if (j < cols - 1) {
                file << ",";
            }
        }
        // Add newline after each row
        file << "\n";
    }

    file.close();
}

int main() {
    int rows = 3;
    int cols = 4;
    int array[3][4] = {{1, 2, 3, 4},
                       {5, 6, 7, 8},
                       {9, 10, 11, 12}};
    save_csv(rows, cols, array, "output.csv");
    std::cout << "CSV file saved successfully." << std::endl;
    return 0;
}
