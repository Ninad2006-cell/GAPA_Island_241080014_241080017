#include "../common/common.h"
#include "../cuda/gpu_utils.cuh"
#include <iostream>
#include <vector>
#include <memory>
#include <algorithm>
#include <omp.h>

class MGA {
private:
    int pop_size;
    int gene_length;
    int num_generations;
    double crossover_rate;
    double mutation_rate;
    double mutation_strength;
    int tournament_size;
    double gene_min, gene_max;
    int num_threads;
    unsigned int seed;
    int cuda_block_size;
    int gpu_id;

    TimingTracker timing;
    GPUTimingTracker gpu_timing;
    std::vector<double> best_fitness_history;
    RandomGenerator master_rng;
    std::vector<RandomGenerator> thread_rngs;
    std::unique_ptr<GPUFitnessEvaluator> gpu_evaluator;

    std::vector<double> host_genes;
    std::vector<double> host_fitness;

public:
    MGA(int ps = DEFAULT_POP_SIZE, int gl = DEFAULT_GENE_LENGTH, 
        int ng = DEFAULT_NUM_GENERATIONS, double cr = DEFAULT_CROSSOVER_RATE, 
        double mr = DEFAULT_MUTATION_RATE, double ms = DEFAULT_MUTATION_STRENGTH,
        int ts = DEFAULT_TOURNAMENT_SIZE, double gmin = DEFAULT_GENE_MIN, 
        double gmax = DEFAULT_GENE_MAX, int nt = 4, unsigned int s = DEFAULT_BASE_SEED,
        int block_size = DEFAULT_CUDA_THREADS_PER_BLOCK, int gid = DEFAULT_GPU_DEVICE_ID)
        : pop_size(ps), gene_length(gl), num_generations(ng),
          crossover_rate(cr), mutation_rate(mr), mutation_strength(ms),
          tournament_size(ts), gene_min(gmin), gene_max(gmax),
          num_threads(nt), seed(s), cuda_block_size(block_size), gpu_id(gid), master_rng(s) {
        
        host_genes.resize(static_cast<size_t>(pop_size) * gene_length);
        host_fitness.resize(pop_size);

        // Independent deterministic RNG per thread
        thread_rngs.resize(num_threads);
        for (int i = 0; i < num_threads; i++) {
            thread_rngs[i].setSeed(seed + (i + 1) * 7919);
        }

        if (GPUFitnessEvaluator::isCudaAvailable()) {
            gpu_evaluator.reset(new GPUFitnessEvaluator(pop_size, gene_length, gpu_id, cuda_block_size));
        }
    }

    void mutate_parallel(std::vector<Individual>& population, int count) {
        ScopeTimer timer;
        timer.start();

        #pragma omp parallel for num_threads(num_threads) schedule(static)
        for (int i = 0; i < count; i++) {
            int tid = omp_get_thread_num();
            mutate(population[i], mutation_rate, mutation_strength, gene_min, gene_max, thread_rngs[tid]);
        }

        timing.mutation_time += timer.stop();
    }

    void evaluate_population_fitness(std::vector<Individual>& population) {
        ScopeTimer timer;
        timer.start();

        if (gpu_evaluator && gpu_evaluator->isInitialized()) {
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
            #pragma omp parallel for num_threads(num_threads) schedule(static)
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
        int num_threads;
        bool used_cuda;
    };

    Result run() {
        timing.reset();
        gpu_timing.reset();
        best_fitness_history.clear();

        ScopeTimer total_timer;
        total_timer.start();

        // 1. Parallel Population Initialization
        ScopeTimer init_timer;
        init_timer.start();
        std::vector<Individual> population(pop_size, Individual(gene_length));
        
        #pragma omp parallel for num_threads(num_threads) schedule(static)
        for (int i = 0; i < pop_size; i++) {
            int tid = omp_get_thread_num();
            for (int j = 0; j < gene_length; j++) {
                population[i].genes[j] = thread_rngs[tid].getDouble(gene_min, gene_max);
            }
        }
        timing.init_time = init_timer.stop();

        evaluate_population_fitness(population);

        Individual global_best = population[0];
        for (const auto& ind : population) {
            if (ind.fitness < global_best.fitness) {
                global_best = ind;
            }
        }
        best_fitness_history.push_back(global_best.fitness);

        // 2. Evolution Loop
        for (int gen = 0; gen < num_generations; gen++) {
            std::vector<Individual> offspring;
            offspring.reserve(pop_size);

            int pairs_needed = (pop_size - 1 + 1) / 2;

            for (int p = 0; p < pairs_needed && (int)offspring.size() < pop_size - 1; p++) {
                ScopeTimer sel_timer;
                sel_timer.start();
                Individual p1 = tournament_selection(population, tournament_size, master_rng);
                Individual p2 = tournament_selection(population, tournament_size, rng_dummy(p));
                timing.selection_time += sel_timer.stop();

                ScopeTimer cross_timer;
                cross_timer.start();
                Individual c1(gene_length), c2(gene_length);
                crossover(p1, p2, c1, c2, crossover_rate, master_rng);
                timing.crossover_time += cross_timer.stop();

                offspring.push_back(c1);
                if ((int)offspring.size() < pop_size - 1) {
                    offspring.push_back(c2);
                }
            }

            // Parallel Mutation across threads (M Mode core design)
            mutate_parallel(offspring, static_cast<int>(offspring.size()));

            // Elitism: append global best untouched
            offspring.push_back(global_best);

            evaluate_population_fitness(offspring);
            population = offspring;

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
        res.num_threads = num_threads;
        res.used_cuda = (gpu_evaluator && gpu_evaluator->isInitialized());
        return res;
    }

private:
    RandomGenerator& rng_dummy(int) {
        return master_rng;
    }
};

#ifndef GA_NO_MAIN
int main(int argc, char** argv) {
    int pop_size = DEFAULT_POP_SIZE;
    int gene_length = DEFAULT_GENE_LENGTH;
    int num_generations = DEFAULT_NUM_GENERATIONS;
    int num_threads = 4;
    unsigned int seed = DEFAULT_BASE_SEED;

    #ifdef _OPENMP
    num_threads = omp_get_max_threads();
    #endif

    if (argc > 1) pop_size = std::stoi(argv[1]);
    if (argc > 2) gene_length = std::stoi(argv[2]);
    if (argc > 3) num_generations = std::stoi(argv[3]);
    if (argc > 4) num_threads = std::stoi(argv[4]);
    if (argc > 5) seed = static_cast<unsigned int>(std::stoul(argv[5]));

    std::cout << "============================================================" << std::endl;
    std::cout << "Approach 3: M (Multi-process Genetic Operations) with CUDA" << std::endl;
    std::cout << "============================================================" << std::endl;
    std::cout << "Worker Threads: " << num_threads << std::endl;

    MGA ga(pop_size, gene_length, num_generations,
           DEFAULT_CROSSOVER_RATE, DEFAULT_MUTATION_RATE, DEFAULT_MUTATION_STRENGTH,
           DEFAULT_TOURNAMENT_SIZE, DEFAULT_GENE_MIN, DEFAULT_GENE_MAX, num_threads, seed);

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
