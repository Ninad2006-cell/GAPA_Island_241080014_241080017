#include "../common/common.h"
#include "../cuda/gpu_utils.cuh"
#include <iostream>
#include <vector>
#include <memory>
#include <algorithm>
#include <omp.h>

struct IslandResult {
    double best_fitness;
    std::vector<double> best_genes;
    int pop_size;
};

class IslandGA {
private:
    int pop_size;
    int gene_length;
    int num_generations;
    double crossover_rate;
    double mutation_rate;
    double mutation_strength;
    int tournament_size;
    double gene_min, gene_max;
    int num_islands;
    unsigned int seed;
    int cuda_block_size;
    int gpu_id;

    TimingTracker timing;
    GPUTimingTracker gpu_timing;
    std::vector<double> global_best_fitness_history;
    std::vector<RandomGenerator> island_rngs;
    std::unique_ptr<GPUFitnessEvaluator> gpu_evaluator;

    std::vector<int> island_sizes;
    std::vector<int> island_offsets;
    std::vector<std::vector<Individual>> islands;

    std::vector<double> host_genes;
    std::vector<double> host_fitness;

public:
    IslandGA(int ps = DEFAULT_POP_SIZE, int gl = DEFAULT_GENE_LENGTH, 
             int ng = DEFAULT_NUM_GENERATIONS, double cr = DEFAULT_CROSSOVER_RATE, 
             double mr = DEFAULT_MUTATION_RATE, double ms = DEFAULT_MUTATION_STRENGTH,
             int ts = DEFAULT_TOURNAMENT_SIZE, double gmin = DEFAULT_GENE_MIN, 
             double gmax = DEFAULT_GENE_MAX, int ni = DEFAULT_NUM_ISLANDS, 
             unsigned int s = DEFAULT_BASE_SEED, int block_size = DEFAULT_CUDA_THREADS_PER_BLOCK, 
             int gid = DEFAULT_GPU_DEVICE_ID)
        : pop_size(ps), gene_length(gl), num_generations(ng),
          crossover_rate(cr), mutation_rate(mr), mutation_strength(ms),
          tournament_size(ts), gene_min(gmin), gene_max(gmax),
          num_islands(ni), seed(s), cuda_block_size(block_size), gpu_id(gid) {
        
        // Ensure valid island count
        if (num_islands <= 0) num_islands = 1;
        if (num_islands > pop_size) num_islands = pop_size;

        island_sizes = calculate_island_sizes(pop_size, num_islands);
        island_offsets.resize(num_islands, 0);
        for (int i = 1; i < num_islands; i++) {
            island_offsets[i] = island_offsets[i - 1] + island_sizes[i - 1];
        }

        // Initialize deterministic per-island RNGs
        island_rngs.resize(num_islands);
        for (int i = 0; i < num_islands; i++) {
            island_rngs[i].setSeed(seed + (i + 1) * 104729);
        }

        host_genes.resize(static_cast<size_t>(pop_size) * gene_length);
        host_fitness.resize(pop_size);

        if (GPUFitnessEvaluator::isCudaAvailable()) {
            gpu_evaluator.reset(new GPUFitnessEvaluator(pop_size, gene_length, gpu_id, cuda_block_size));
        }
    }

    // Unified Batched CUDA Fitness Evaluation
    void evaluate_all_islands_batched() {
        ScopeTimer timer;
        timer.start();

        ScopeTimer pack_timer;
        pack_timer.start();

        // 1. Pack all islands into contiguous host buffer
        #pragma omp parallel for num_threads(num_islands) schedule(static)
        for (int isl = 0; isl < num_islands; isl++) {
            int offset = island_offsets[isl];
            int isize = island_sizes[isl];
            for (int i = 0; i < isize; i++) {
                int global_idx = offset + i;
                int base = global_idx * gene_length;
                for (int j = 0; j < gene_length; j++) {
                    host_genes[base + j] = islands[isl][i].genes[j];
                }
            }
        }
        timing.communication_time += pack_timer.stop();

        // 2. Offload to GPU via single batched transfer & kernel execution
        if (gpu_evaluator && gpu_evaluator->isInitialized()) {
            gpu_evaluator->evaluate_rastrigin(host_genes.data(), host_fitness.data(), pop_size, gpu_timing);
        } else {
            // Multithreaded CPU fallback
            #pragma omp parallel for num_threads(num_islands) schedule(static)
            for (int i = 0; i < pop_size; i++) {
                host_fitness[i] = rastrigin_array(&host_genes[i * gene_length], gene_length);
            }
        }

        // 3. Unpack fitness back to individual islands
        ScopeTimer unpack_timer;
        unpack_timer.start();
        #pragma omp parallel for num_threads(num_islands) schedule(static)
        for (int isl = 0; isl < num_islands; isl++) {
            int offset = island_offsets[isl];
            int isize = island_sizes[isl];
            for (int i = 0; i < isize; i++) {
                islands[isl][i].fitness = host_fitness[offset + i];
            }
        }
        timing.communication_time += unpack_timer.stop();

        timing.fitness_time += timer.stop();
    }

    struct Result {
        double best_fitness;
        std::vector<double> best_genes;
        std::vector<double> fitness_history;
        TimingTracker timings;
        int num_islands;
        std::vector<IslandResult> island_results;
        bool used_cuda;
    };

    Result run() {
        timing.reset();
        gpu_timing.reset();
        global_best_fitness_history.clear();

        ScopeTimer total_timer;
        total_timer.start();

        // 1. Independent Population Initialization for each Island
        ScopeTimer init_timer;
        init_timer.start();
        islands.resize(num_islands);
        
        #pragma omp parallel for num_threads(num_islands) schedule(static)
        for (int isl = 0; isl < num_islands; isl++) {
            islands[isl] = initialize_population(island_sizes[isl], gene_length, 
                                                 gene_min, gene_max, island_rngs[isl]);
        }
        timing.init_time = init_timer.stop();

        // Evaluate initial fitness across all islands using batched GPU call
        evaluate_all_islands_batched();

        // Track local and global bests
        std::vector<Individual> local_bests(num_islands);
        Individual global_best;
        global_best.fitness = 1e30;

        for (int isl = 0; isl < num_islands; isl++) {
            local_bests[isl] = islands[isl][0];
            for (const auto& ind : islands[isl]) {
                if (ind.fitness < local_bests[isl].fitness) {
                    local_bests[isl] = ind;
                }
            }
            if (local_bests[isl].fitness < global_best.fitness) {
                global_best = local_bests[isl];
            }
        }
        global_best_fitness_history.push_back(global_best.fitness);

        // 2. Evolutionary Loop
        for (int gen = 0; gen < num_generations; gen++) {
            std::vector<std::vector<Individual>> next_islands(num_islands);

            std::vector<double> island_selection_seconds(num_islands, 0.0);
            std::vector<double> island_crossover_seconds(num_islands, 0.0);
            std::vector<double> island_mutation_seconds(num_islands, 0.0);

            // Parallel evolution within each island
            #pragma omp parallel for num_threads(num_islands) schedule(static)
            for (int isl = 0; isl < num_islands; isl++) {
                int isize = island_sizes[isl];
                RandomGenerator& rng = island_rngs[isl];
                std::vector<Individual>& curr_pop = islands[isl];
                std::vector<Individual> offspring;
                offspring.reserve(isize);

                // Reserve 1 slot for local elitism
                int pairs_needed = (isize - 1 + 1) / 2;

                ScopeTimer selection_timer;
                ScopeTimer crossover_timer;
                ScopeTimer mutation_timer;

                for (int p = 0; p < pairs_needed && (int)offspring.size() < isize - 1; p++) {
                    selection_timer.start();
                    Individual p1 = tournament_selection(curr_pop, tournament_size, rng);
                    Individual p2 = tournament_selection(curr_pop, tournament_size, rng);
                    island_selection_seconds[isl] += selection_timer.stop();

                    Individual c1(gene_length), c2(gene_length);
                    crossover_timer.start();
                    crossover(p1, p2, c1, c2, crossover_rate, rng);
                    island_crossover_seconds[isl] += crossover_timer.stop();

                    mutation_timer.start();
                    mutate(c1, mutation_rate, mutation_strength, gene_min, gene_max, rng);
                    mutate(c2, mutation_rate, mutation_strength, gene_min, gene_max, rng);
                    island_mutation_seconds[isl] += mutation_timer.stop();

                    offspring.push_back(c1);
                    if ((int)offspring.size() < isize - 1) {
                        offspring.push_back(c2);
                    }
                }

                // Island Elitism: carry local best forward
                offspring.push_back(local_bests[isl]);
                next_islands[isl] = offspring;
            }

            // Use the slowest island's stage duration as the parallel wall-time
            // estimate; do not invent percentages of a combined timer.
            timing.selection_time += *std::max_element(island_selection_seconds.begin(), island_selection_seconds.end());
            timing.crossover_time += *std::max_element(island_crossover_seconds.begin(), island_crossover_seconds.end());
            timing.mutation_time += *std::max_element(island_mutation_seconds.begin(), island_mutation_seconds.end());

            // Replace populations
            islands = next_islands;

            // Batched GPU Fitness Evaluation for all islands combined
            evaluate_all_islands_batched();

            // Update local and global bests
            for (int isl = 0; isl < num_islands; isl++) {
                for (const auto& ind : islands[isl]) {
                    if (ind.fitness < local_bests[isl].fitness) {
                        local_bests[isl] = ind;
                    }
                }
                if (local_bests[isl].fitness < global_best.fitness) {
                    global_best = local_bests[isl];
                }
            }
            global_best_fitness_history.push_back(global_best.fitness);

            if (gen % 10 == 0 || gen == num_generations - 1) {
                std::cout << "Generation " << gen << ": Global best fitness = " << global_best.fitness << std::endl;
            }
        }

        timing.total_time = total_timer.stop();

        gpu_timing.computeTotal();
        timing.gpu_host_to_device_time = gpu_timing.host_to_device_time;
        timing.gpu_kernel_time = gpu_timing.kernel_time;
        timing.gpu_device_to_host_time = gpu_timing.device_to_host_time;
        timing.gpu_sync_time = gpu_timing.sync_time;
        timing.gpu_total_time = gpu_timing.total_gpu_time;

        std::vector<IslandResult> isl_res(num_islands);
        for (int i = 0; i < num_islands; i++) {
            isl_res[i].best_fitness = local_bests[i].fitness;
            isl_res[i].best_genes = local_bests[i].genes;
            isl_res[i].pop_size = island_sizes[i];
        }

        Result res;
        res.best_fitness = global_best.fitness;
        res.best_genes = global_best.genes;
        res.fitness_history = global_best_fitness_history;
        res.timings = timing;
        res.num_islands = num_islands;
        res.island_results = isl_res;
        res.used_cuda = (gpu_evaluator && gpu_evaluator->isInitialized());
        return res;
    }
};

#ifndef GA_NO_MAIN
int main(int argc, char** argv) {
    int pop_size = DEFAULT_POP_SIZE;
    int gene_length = DEFAULT_GENE_LENGTH;
    int num_generations = DEFAULT_NUM_GENERATIONS;
    int num_islands = DEFAULT_NUM_ISLANDS;
    unsigned int seed = DEFAULT_BASE_SEED;

    if (argc > 1) pop_size = std::stoi(argv[1]);
    if (argc > 2) gene_length = std::stoi(argv[2]);
    if (argc > 3) num_generations = std::stoi(argv[3]);
    if (argc > 4) num_islands = std::stoi(argv[4]);
    if (argc > 5) seed = static_cast<unsigned int>(std::stoul(argv[5]));

    std::cout << "============================================================" << std::endl;
    std::cout << "Approach 5: Island-Based GA with Batched CUDA Fitness (Proposed)" << std::endl;
    std::cout << "============================================================" << std::endl;
    std::cout << "Islands: " << num_islands << ", Total Population: " << pop_size << std::endl;

    IslandGA ga(pop_size, gene_length, num_generations,
                DEFAULT_CROSSOVER_RATE, DEFAULT_MUTATION_RATE, DEFAULT_MUTATION_STRENGTH,
                DEFAULT_TOURNAMENT_SIZE, DEFAULT_GENE_MIN, DEFAULT_GENE_MAX, 
                num_islands, seed);

    auto result = ga.run();

    std::cout << "\n============================================================" << std::endl;
    std::cout << "Results:" << std::endl;
    std::cout << "============================================================" << std::endl;
    std::cout << "CUDA used: " << (result.used_cuda ? "Yes" : "No") << std::endl;
    std::cout << "Best fitness: " << result.best_fitness << std::endl;
    std::cout << "Number of islands: " << result.num_islands << std::endl;

    for (size_t i = 0; i < result.island_results.size(); i++) {
        std::cout << "  Island " << i << " (pop=" << result.island_results[i].pop_size 
                  << "): Best fitness = " << result.island_results[i].best_fitness << std::endl;
    }

    result.timings.printSummary();

    return 0;
}
#endif 
