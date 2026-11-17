#ifndef CONFIG_H
#define CONFIG_H

// Default GA Hyperparameters
#define DEFAULT_POP_SIZE 100
#define DEFAULT_GENE_LENGTH 10
#define DEFAULT_NUM_GENERATIONS 50
#define DEFAULT_CROSSOVER_RATE 0.8
#define DEFAULT_MUTATION_RATE 0.1
#define DEFAULT_MUTATION_STRENGTH 0.1
#define DEFAULT_TOURNAMENT_SIZE 3
#define DEFAULT_GENE_MIN -5.12
#define DEFAULT_GENE_MAX 5.12

// Island-Based Parameters
#define DEFAULT_NUM_ISLANDS 4

// CUDA Execution Parameters
#define DEFAULT_CUDA_THREADS_PER_BLOCK 256
#define DEFAULT_GPU_DEVICE_ID 0

// Experimental Benchmarking Parameters
#define DEFAULT_NUM_TRIALS 3
#define DEFAULT_BASE_SEED 42

#endif // CONFIG_H
