#ifndef FITNESS_H
#define FITNESS_H

#include <vector>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef M_E
#define M_E 2.71828182845904523536
#endif

// Rastrigin function - multimodal benchmark minimization problem
// f(x) = 10*D + sum_{i=1}^D [x_i^2 - 10*cos(2*pi*x_i)]
// Global minimum is 0.0 at x = [0, 0, ..., 0]
inline double rastrigin(const std::vector<double>& genes) {
    const double A = 10.0;
    const int n = static_cast<int>(genes.size());
    double sum = 0.0;
    
    for (int i = 0; i < n; i++) {
        sum += (genes[i] * genes[i]) - (A * std::cos(2.0 * M_PI * genes[i]));
    }
    
    return (A * n) + sum;
}

// Raw pointer array version for direct CPU evaluation matching CUDA buffer layout
inline double rastrigin_array(const double* genes, int gene_length) {
    const double A = 10.0;
    double sum = 0.0;
    for (int i = 0; i < gene_length; i++) {
        sum += (genes[i] * genes[i]) - (A * std::cos(2.0 * M_PI * genes[i]));
    }
    return (A * gene_length) + sum;
}

// Sphere function - simple unimodal
inline double sphere(const std::vector<double>& genes) {
    double sum = 0.0;
    for (double gene : genes) {
        sum += gene * gene;
    }
    return sum;
}

// Rosenbrock function - non-convex banana valley
inline double rosenbrock(const std::vector<double>& genes) {
    double sum = 0.0;
    const int n = static_cast<int>(genes.size());
    for (int i = 0; i < n - 1; i++) {
        double term1 = 100.0 * (genes[i+1] - genes[i] * genes[i]);
        double term2 = genes[i] - 1.0;
        sum += term1 * term1 + term2 * term2;
    }
    return sum;
}

// Ackley function - multimodal
inline double ackley(const std::vector<double>& genes) {
    const int n = static_cast<int>(genes.size());
    if (n == 0) return 0.0;
    double sum_sq = 0.0;
    double sum_cos = 0.0;
    
    for (double gene : genes) {
        sum_sq += gene * gene;
        sum_cos += std::cos(2.0 * M_PI * gene);
    }
    
    double term1 = -20.0 * std::exp(-0.2 * std::sqrt(sum_sq / n));
    double term2 = -std::exp(sum_cos / n);
    
    return term1 + term2 + 20.0 + M_E;
}

typedef double (*FitnessFunc)(const std::vector<double>&);

#endif // FITNESS_H
