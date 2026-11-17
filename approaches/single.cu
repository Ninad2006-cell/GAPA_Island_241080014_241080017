#include "../common/common.h"
#include "../cuda/gpu_utils.cuh"
#include <iostream>
#include <vector>
#include <memory>
#include <algorithm>

class SingleProcessGA {
private:
    int pop_size;
    int gene_length;
    int num_generations;
    double crossover_rate;
    double mutation_rate;
    double mutation_strength;
    int tournament_size;
    double gene_min, gene_max;
    unsigned int seed;
    int cuda_block_size;
    int gpu_id;

    TimingTracker timing;
    GPUTimingTracker gpu_timing;
    std::vector<double> best_fitness_history;
    RandomGenerator rng;
    std::unique_ptr<GPUFitnessEvaluator> gpu_evaluator;

    // Contiguous host buffers for GPU offload
    std::vector<double> host_genes;
    std::vector<double> host_fitness;

public:
    SingleProcessGA(int ps = DEFAULT_POP_SIZE, int gl = DEFAULT_GENE_LENGTH, 
                    int ng = DEFAULT_NUM_GENERATIONS, double cr = DEFAULT_CROSSOVER_RATE, 
                    double mr = DEFAULT_MUTATION_RATE, double ms = DEFAULT_MUTATION_STRENGTH,
                    int ts = DEFAULT_TOURNAMENT_SIZE, double gmin = DEFAULT_GENE_MIN, 
                    double gmax = DEFAULT_GENE_MAX, unsigned int s = DEFAULT_BASE_SEED,
                    int block_size = DEFAULT_CUDA_THREADS_PER_BLOCK, int gid = DEFAULT_GPU_DEVICE_ID)
        : pop_size(ps), gene_length(gl), num_generations(ng),
          crossover_rate(cr), mutation_rate(mr), mutation_strength(ms),
          tournament_size(ts), gene_min(gmin), gene_max(gmax), seed(s),
          cuda_block_size(block_size), gpu_id(gid), rng(s) {
        
        host_genes.resize(static_cast<size_t>(pop_size) * gene_length);
        host_fitness.resize(pop_size);

        if (GPUFitnessEvaluator::isCudaAvailable()) {
            gpu_evaluator.reset(new GPUFitnessEvaluator(pop_size, gene_length, gpu_id, cuda_block_size));
        }
    }

    void evaluate_population_fitness(std::vector<Individual>& population) {
        ScopeTimer timer;
        timer.start();

        if (gpu_evaluator && gpu_evaluator->isInitialized()) {
            // Flatten population into contiguous host buffer
            for (int i = 0; i < pop_size; i++) {
                int base = i * gene_length;
                for (int j = 0; j < gene_length; j++) {
                    host_genes[base + j] = population[i].genes[j];
                }
            }

            gpu_evaluator->evaluate_rastrigin(host_genes.data(), host_fitness.data(), pop_size, gpu_timing);

            for (int i = 0; i < pop_size; i++) {
                population[i].fitness = host_fitness[i];
            }
        } else {
            // CPU Fallback
            for (int i = 0; i < pop_size; i++) {
                population[i].fitness = rastrigin(population[i].genes);
            }
        }

        timing.fitness_time += timer.stop();
    }

    struct Result {
        double best_fitness;
        std::vector<double> best_genes;
        std::vector<double> fitness_history;
        TimingTracker timings;
        bool used_cuda;
    };

    Result run() {
        timing.reset();
        gpu_timing.reset();
        best_fitness_history.clear();

        ScopeTimer total_timer;
        total_timer.start();

        // 1. Initialization
        ScopeTimer init_timer;
        init_timer.start();
        std::vector<Individual> population = initialize_population(pop_size, gene_length, gene_min, gene_max, rng);
        timing.init_time = init_timer.stop();

        // Evaluate initial fitness
        evaluate_population_fitness(population);

        // Find initial best
        Individual global_best = population[0];
        for (const auto& ind : population) {
            if (ind.fitness < global_best.fitness) {
                global_best = ind;
            }
        }
        best_fitness_history.push_back(global_best.fitness);

        // 2. Evolutionary Loop
        for (int gen = 0; gen < num_generations; gen++) {
            std::vector<Individual> offspring;
            offspring.reserve(pop_size);

            // Reserve 1 slot for Elitism
            int pairs_needed = (pop_size - 1 + 1) / 2;

            for (int p = 0; p < pairs_needed && (int)offspring.size() < pop_size - 1; p++) {
                // Selection
                ScopeTimer sel_timer;
                sel_timer.start();
                Individual p1 = tournament_selection(population, tournament_size, rng);
                Individual p2 = tournament_selection(population, tournament_size, rng);
                timing.selection_time += sel_timer.stop();

                // Crossover
                ScopeTimer cross_timer;
                cross_timer.start();
                Individual c1(gene_length), c2(gene_length);
                crossover(p1, p2, c1, c2, crossover_rate, rng);
                timing.crossover_time += cross_timer.stop();

                // Mutation
                ScopeTimer mut_timer;
                mut_timer.start();
                mutate(c1, mutation_rate, mutation_strength, gene_min, gene_max, rng);
                mutate(c2, mutation_rate, mutation_strength, gene_min, gene_max, rng);
                timing.mutation_time += mut_timer.stop();

                offspring.push_back(c1);
                if ((int)offspring.size() < pop_size - 1) {
                    offspring.push_back(c2);
                }
            }

            // Elitism: carry best individual forward
            offspring.push_back(global_best);

            // Fitness evaluation on GPU
            evaluate_population_fitness(offspring);
            population = offspring;

            // Track global best
            for (const auto& ind : population) {
                if (ind.fitness < global_best.fitness) {
                    global_best = ind;
                }
            }
            best_fitness_history.push_back(global_best.fitness);

            if (gen % 10 == 0 || gen == num_generations - 1) {
                std::cout << "Generation " << gen << ": Best fitness = " << global_best.fitness << std::endl;
            }
        }

        timing.total_time = total_timer.stop();

        // Accumulate GPU timing
        gpu_timing.computeTotal();
        timing.gpu_host_to_device_time = gpu_timing.host_to_device_time;
        timing.gpu_kernel_time = gpu_timing.kernel_time;
        timing.gpu_device_to_host_time = gpu_timing.device_to_host_time;
        timing.gpu_sync_time = gpu_timing.sync_time;
        timing.gpu_total_time = gpu_timing.total_gpu_time;

        Result res;
        res.best_fitness = global_best.fitness;
        res.best_genes = global_best.genes;
        res.fitness_history = best_fitness_history;
        res.timings = timing;
        res.used_cuda = (gpu_evaluator && gpu_evaluator->isInitialized());
        return res;
    }
};

#ifndef GA_NO_MAIN
int main(int argc, char** argv) {
    int pop_size = DEFAULT_POP_SIZE;
    int gene_length = DEFAULT_GENE_LENGTH;
    int num_generations = DEFAULT_NUM_GENERATIONS;
    unsigned int seed = DEFAULT_BASE_SEED;

    if (argc > 1) pop_size = std::stoi(argv[1]);
    if (argc > 2) gene_length = std::stoi(argv[2]);
    if (argc > 3) num_generations = std::stoi(argv[3]);
    if (argc > 4) seed = static_cast<unsigned int>(std::stoul(argv[4]));

    std::cout << "============================================================" << std::endl;
    std::cout << "Approach 1: Single Process (S Mode) with CUDA" << std::endl;
    std::cout << "============================================================" << std::endl;

    SingleProcessGA ga(pop_size, gene_length, num_generations,
                       DEFAULT_CROSSOVER_RATE, DEFAULT_MUTATION_RATE, DEFAULT_MUTATION_STRENGTH,
                       DEFAULT_TOURNAMENT_SIZE, DEFAULT_GENE_MIN, DEFAULT_GENE_MAX, seed);

    auto result = ga.run();

    std::cout << "\n============================================================" << std::endl;
    std::cout << "Results:" << std::endl;
    std::cout << "============================================================" << std::endl;
    std::cout << "CUDA used: " << (result.used_cuda ? "Yes" : "No") << std::endl;
    std::cout << "Best fitness: " << result.best_fitness << std::endl;
    result.timings.printSummary();

    return 0;
}
#endif // GA_NO_MAIN
