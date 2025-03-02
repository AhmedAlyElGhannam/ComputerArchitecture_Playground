# Matrix Multiplication Using CUDA/C++ For Performance Assessment vs. CPUs

## Introduction
This repository contains the final project of the elective course `35x CSE: Computer Architecture` taught in the final year in Electronics & Communications Engineering, Alexandria University, Egypt. As specified by the included [lab manual](https://github.com/AhmedAlyElGhannam/ComputerArchitecture_Playground/blob/main/Lab_3/GPU_Lab_Manual.pdf), it was required to create a program that calculates the result of matrix multiplication using CUDA. However, since this was the last project I would have made in college, I took great care to produce the finest output I could at the time: converting the project into a fully-detailed report on the efficiency of doing said calculations on GPU instead of CPU using C++ using `nvprof` and `gprof`, comparing GPU performance at different thread count, and plotting the results using a simple MATLAB script. The report can be found [here](https://github.com/AhmedAlyElGhannam/ComputerArchitecture_Playground/blob/main/Lab_3/GPU_Lab_Report.pdf). 

## Used Profiling Commands For Reference
> For GPU profiling shenanigans, use `nvprof` before launching your executable to generate a full analysis report.
> For profiling CPU c++ code, use: `gprof nocuda gmon.out > analysis.txt` 
