#ifndef GA_UTILS_H
#define GA_UTILS_H

#include <vector>
#include <random>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <string>
#include <sstream>

#ifdef _OPENMP
#include <omp.h>
#endif

// Individual candidate solution
struct Individual {
    std::vector<double> genes;
    double fitness;
    
    Individual() : fitness(1e30) {}
    explicit Individual(size_t gene_length) : genes(gene_length, 0.0), fitness(1e30) {}
    Individual(const std::vector<double>& g, double f = 1e30) : genes(g), fitness(f) {}
};

// Independent high-resolution CPU stopwatch
class ScopeTimer {
private:
    std::chrono::steady_clock::time_point start_pt;
    bool running;
public:
    ScopeTimer() : running(false) {}

    void start() {
        start_pt = std::chrono::steady_clock::now();
        running = true;
    }

    double stop() {
        if (!running) return 0.0;
        auto end_pt = std::chrono::steady_clock::now();
        running = false;
        return std::chrono::duration<double>(end_pt - start_pt).count();
    }
};

// Comprehensive timing tracker with independent timers per category
class TimingTracker {
public:
    double total_time = 0.0;
    double init_time = 0.0;
    double selection_time = 0.0;
    double crossover_time = 0.0;
    double mutation_time = 0.0;
    double fitness_time = 0.0;
    double communication_time = 0.0;
    double other_time = 0.0;
    
    // Explicit GPU timing breakdown (via CUDA events)
    double gpu_host_to_device_time = 0.0;
    double gpu_kernel_time = 0.0;
    double gpu_device_to_host_time = 0.0;
    double gpu_sync_time = 0.0;
    double gpu_total_time = 0.0;

    void reset() {
        total_time = 0.0;
        init_time = 0.0;
        selection_time = 0.0;
        crossover_time = 0.0;
        mutation_time = 0.0;
        fitness_time = 0.0;
        communication_time = 0.0;
        other_time = 0.0;
        gpu_host_to_device_time = 0.0;
        gpu_kernel_time = 0.0;
        gpu_device_to_host_time = 0.0;
        gpu_sync_time = 0.0;
        gpu_total_time = 0.0;
    }
    
    void printSummary() const {
        std::cout << "\n=== Timing Summary ===" << std::endl;
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "total                    : " << total_time << "s" << std::endl;
        std::cout << "initialization           : " << init_time << "s" << std::endl;
        std::cout << "selection                : " << selection_time << "s" << std::endl;
        std::cout << "crossover                : " << crossover_time << "s" << std::endl;
        std::cout << "mutation                 : " << mutation_time << "s" << std::endl;
        std::cout << "fitness                  : " << fitness_time << "s" << std::endl;
        std::cout << "communication            : " << communication_time << "s" << std::endl;
        std::cout << "other                    : " << other_time << "s" << std::endl;
        
        if (gpu_total_time > 0.0 || gpu_kernel_time > 0.0) {
            std::cout << "\n=== GPU Timing (CUDA Events) ===" << std::endl;
            std::cout << "Host to Device Transfer : " << gpu_host_to_device_time << "s" << std::endl;
            std::cout << "Kernel Execution        : " << gpu_kernel_time << "s" << std::endl;
            std::cout << "Device to Host Transfer : " << gpu_device_to_host_time << "s" << std::endl;
            std::cout << "Synchronization Overhead: " << gpu_sync_time << "s" << std::endl;
            std::cout << "Total GPU Time          : " << gpu_total_time << "s" << std::endl;
        }
    }
};

class RandomGenerator {
private:
    std::mt19937 gen;
    
public:
    RandomGenerator() : gen(1337) {}
    explicit RandomGenerator(unsigned int seed) : gen(seed) {}

    void setSeed(unsigned int seed) {
        gen.seed(seed);
    }
    
    double getDouble(double min_val, double max_val) {
        std::uniform_real_distribution<double> dist(min_val, max_val);
        return dist(gen);
    }
    
    int getInt(int min_val, int max_val) {
        std::uniform_int_distribution<int> dist(min_val, max_val);
        return dist(gen);
    }
    
    double getGaussian(double mean, double stddev) {
        std::normal_distribution<double> dist(mean, stddev);
        return dist(gen);
    }
};

inline std::vector<Individual> initialize_population(int pop_size, int gene_length, 
                                                     double gene_min, double gene_max, 
                                                     RandomGenerator& rng) {
    std::vector<Individual> population;
    population.reserve(pop_size);
    
    for (int i = 0; i < pop_size; i++) {
        Individual ind(gene_length);
        for (int j = 0; j < gene_length; j++) {
            ind.genes[j] = rng.getDouble(gene_min, gene_max);
        }
        population.push_back(ind);
    }
    
    return population;
}

inline Individual tournament_selection(const std::vector<Individual>& population, 
                                      int tournament_size, RandomGenerator& rng) {
    const int pop_size = static_cast<int>(population.size());
    if (pop_size <= 0) return Individual();
    
    int best_idx = rng.getInt(0, pop_size - 1);
    double best_fitness = population[best_idx].fitness;
    
    for (int i = 1; i < tournament_size && i < pop_size; i++) {
        int idx = rng.getInt(0, pop_size - 1);
        if (population[idx].fitness < best_fitness) {
            best_fitness = population[idx].fitness;
            best_idx = idx;
        }
    }
    
    return population[best_idx];
}

inline void crossover(const Individual& parent1, const Individual& parent2,
                      Individual& child1, Individual& child2, 
                      double crossover_rate, RandomGenerator& rng) {
    const int gene_length = static_cast<int>(parent1.genes.size());
    child1.genes.resize(gene_length);
    child2.genes.resize(gene_length);
    
    if (rng.getDouble(0.0, 1.0) > crossover_rate) {
        child1.genes = parent1.genes;
        child2.genes = parent2.genes;
        return;
    }
    
    for (int i = 0; i < gene_length; i++) {
        if (rng.getDouble(0.0, 1.0) < 0.5) {
            child1.genes[i] = parent1.genes[i];
            child2.genes[i] = parent2.genes[i];
        } else {
            child1.genes[i] = parent2.genes[i];
            child2.genes[i] = parent1.genes[i];
        }
    }
}

inline void mutate(Individual& individual, double mutation_rate, 
                   double mutation_strength, double gene_min, double gene_max,
                   RandomGenerator& rng) {
    const int gene_length = static_cast<int>(individual.genes.size());
    for (int i = 0; i < gene_length; i++) {
        if (rng.getDouble(0.0, 1.0) < mutation_rate) {
            individual.genes[i] += rng.getGaussian(0.0, mutation_strength);
            individual.genes[i] = std::max(gene_min, std::min(gene_max, individual.genes[i]));
        }
    }
}

inline std::vector<int> calculate_island_sizes(int total_pop, int num_islands) {
    if (num_islands <= 0) return {total_pop};
    if (num_islands > total_pop) num_islands = total_pop;
    
    int base_size = total_pop / num_islands;
    int remainder = total_pop % num_islands;
    
    std::vector<int> sizes(num_islands, base_size);
    for (int i = 0; i < remainder; i++) {
        sizes[i] += 1;
    }
    return sizes;
}

#endif 
