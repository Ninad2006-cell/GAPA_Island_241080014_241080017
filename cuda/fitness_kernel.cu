#include <cuda_runtime.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Rastrigin fitness function kernel
// Evaluates pop_size individuals concurrently.
// genes buffer layout: [ind_0_gene_0, ind_0_gene_1, ..., ind_1_gene_0, ...]
__global__ void rastrigin_kernel(const double* __restrict__ genes, 
                                double* __restrict__ fitness, 
                                int gene_length, 
                                int pop_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < pop_size) {
        const double A = 10.0;
        double sum = 0.0;
        int base = idx * gene_length;
        
        for (int i = 0; i < gene_length; i++) {
            double gene = genes[base + i];
            sum += (gene * gene) - (A * cos(2.0 * M_PI * gene));
        }
        
        fitness[idx] = (A * gene_length) + sum;
    }
}

// Sphere fitness function kernel
__global__ void sphere_kernel(const double* __restrict__ genes, 
                             double* __restrict__ fitness, 
                             int gene_length, 
                             int pop_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < pop_size) {
        double sum = 0.0;
        int base = idx * gene_length;
        for (int i = 0; i < gene_length; i++) {
            double gene = genes[base + i];
            sum += gene * gene;
        }
        fitness[idx] = sum;
    }
}
