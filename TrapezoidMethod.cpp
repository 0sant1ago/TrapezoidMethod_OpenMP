#include <iostream>
#include <omp.h>
#include <chrono>

#define N 1000000 // Number of elements (dx)

double f(double x) {
    // Define the function to integrate (e.g., f(x) = x^2)
    return 4 / (1 + x * x);
}

int main() {
    double a = 0.0; // Lower limit
    double b = 1.0; // Upper limit
    double dx = (b - a) / N; // Width of each trapezoid
    double begin1, end1, total1, begin2, end2, total2;

    // Sequential computation
    auto start_seq = std::chrono::high_resolution_clock::now();

    begin1 = omp_get_wtime();
    double integral_seq = 0.0;

    for (int i = 1; i < N; i++) {
        double x_left = a + i * dx;
        double x_right = a + (i + 1) * dx;

        integral_seq += 0.5 * (f(x_left) + f(x_right)) * dx;
    }
    end1 = omp_get_wtime();
    total1 = end1 - begin1;

    auto end_seq = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seq = end_seq - start_seq;

    begin2 = omp_get_wtime();
    // Parallel computation
    omp_set_num_threads(4);

    auto start_par = std::chrono::high_resolution_clock::now();

    double integral_shared = 0.0;
    double integral_reduction = 0.0;

    #pragma omp parallel
    {
        int threadID = omp_get_thread_num();

        // Compute local integral for each thread
        double local_integral = 0.0;

        #pragma omp for
        for (int i = 1; i < N; i++) {
            double x_left = a + i * dx;
            double x_right = a + (i + 1) * dx;

            local_integral += 0.5 * (f(x_left) + f(x_right)) * dx;
        }

        // shared variable and critical section
        #pragma omp critical
        integral_shared += local_integral;

        // Using the reduction clause
        ///#pragma omp atomic
        ///integral_reduction += local_integral;
    }
    end2 = omp_get_wtime();
    total2 = end2 - begin2;

    auto end_par = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_par = end_par - start_par;

    // Output the results
    std::cout << "Sequential Integral: " << integral_seq << std::endl;
    std::cout << "Parallel Integral (shared): " << integral_shared << std::endl;
    ////std::cout << "Parallel Integral (reduction): " << integral_reduction << std::endl;

    // Measure and output the execution time
    std::cout << "Sequential Time: " << elapsed_seq.count() << " seconds" << std::endl;
    std::cout << "Parallel Time: " << elapsed_par.count() << " seconds" << std::endl;

    std::cout << "sequential time = " << total1 << std::endl;
    std::cout << "parallel time = " << total2 << std::endl;

    // Calculate speedup
    double speedup = elapsed_seq.count() / elapsed_par.count();
    std::cout << "Speedup: " << speedup << std::endl;

    std::cout << "speedup = " << total1 / total2 << std::endl;

    return 0;
}
