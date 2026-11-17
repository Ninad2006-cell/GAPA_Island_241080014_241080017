#include "../common/common.h"
#include "../cuda/gpu_utils.cuh"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <sstream>
#include <cstdlib>

// Forward declarations of approach runner interfaces
// Approach 1
#include "../approaches/single.cu"
// Approach 2
#include "../approaches/sm.cu"
// Approach 3
#include "../approaches/m.cu"
// Approach 4
#include "../approaches/mnm.cu"
// Approach 5
#include "../approaches/island.cu"

struct BenchmarkConfig {
    int pop_size = DEFAULT_POP_SIZE;
    int gene_length = DEFAULT_GENE_LENGTH;
    int num_generations = DEFAULT_NUM_GENERATIONS;
    double crossover_rate = DEFAULT_CROSSOVER_RATE;
    double mutation_rate = DEFAULT_MUTATION_RATE;
    double mutation_strength = DEFAULT_MUTATION_STRENGTH;
    int tournament_size = DEFAULT_TOURNAMENT_SIZE;
    double gene_min = DEFAULT_GENE_MIN;
    double gene_max = DEFAULT_GENE_MAX;
    int num_islands = DEFAULT_NUM_ISLANDS;
    int omp_threads = 4;
    int cuda_block_size = DEFAULT_CUDA_THREADS_PER_BLOCK;
    int gpu_id = DEFAULT_GPU_DEVICE_ID;
    int num_trials = DEFAULT_NUM_TRIALS;
    unsigned int base_seed = DEFAULT_BASE_SEED;
    std::string csv_output = "../results/benchmark_results.csv";
};

void run_all_benchmarks(const BenchmarkConfig& cfg) {
    std::ofstream csv(cfg.csv_output);
    if (!csv.is_open()) {
        std::cerr << "Error: Could not open CSV output file: " << cfg.csv_output << std::endl;
        return;
    }

    // Write standard CSV header schema as required
    csv << "approach,trial,seed,population_size,generations,dimensions,island_count,omp_threads,"
        << "cuda_block_size,gpu_id,used_cuda,best_fitness,total_time,initialization_time,"
        << "selection_time,crossover_time,mutation_time,fitness_time,kernel_time,h2d_time,"
        << "d2h_time,sync_time,communication_time\n";

    std::cout << "\n================================================================================" << std::endl;
    std::cout << "Starting Comprehensive Benchmark Suite (" << cfg.num_trials << " trials per approach)" << std::endl;
    std::cout << "Population: " << cfg.pop_size << ", Generations: " << cfg.num_generations 
              << ", Dimensions: " << cfg.gene_length << ", Threads: " << cfg.omp_threads 
              << ", Islands: " << cfg.num_islands << std::endl;
    std::cout << "================================================================================" << std::endl;

    const std::vector<std::string> approaches = {"single", "sm", "m", "mnm", "island"};

    for (const auto& app : approaches) {
        std::cout << "\n>>> Benchmarking Approach: [" << app << "] <<<" << std::endl;

        for (int trial = 1; trial <= cfg.num_trials; trial++) {
            unsigned int trial_seed = cfg.base_seed + (trial - 1) * 10007;
            std::cout << "  Trial " << trial << "/" << cfg.num_trials << " (Seed: " << trial_seed << ")... " << std::flush;

            double best_fit = 0.0;
            TimingTracker t;
            bool used_cuda = false;

            if (app == "single") {
                SingleProcessGA ga(cfg.pop_size, cfg.gene_length, cfg.num_generations,
                                   cfg.crossover_rate, cfg.mutation_rate, cfg.mutation_strength,
                                   cfg.tournament_size, cfg.gene_min, cfg.gene_max, trial_seed,
                                   cfg.cuda_block_size, cfg.gpu_id);
                auto res = ga.run();
                best_fit = res.best_fitness;
                t = res.timings;
                used_cuda = res.used_cuda;
            } else if (app == "sm") {
                SMGA ga(cfg.pop_size, cfg.gene_length, cfg.num_generations,
                        cfg.crossover_rate, cfg.mutation_rate, cfg.mutation_strength,
                        cfg.tournament_size, cfg.gene_min, cfg.gene_max, cfg.omp_threads, trial_seed,
                        cfg.cuda_block_size, cfg.gpu_id);
                auto res = ga.run();
                best_fit = res.best_fitness;
                t = res.timings;
                used_cuda = res.used_cuda;
            } else if (app == "m") {
                MGA ga(cfg.pop_size, cfg.gene_length, cfg.num_generations,
                       cfg.crossover_rate, cfg.mutation_rate, cfg.mutation_strength,
                       cfg.tournament_size, cfg.gene_min, cfg.gene_max, cfg.omp_threads, trial_seed,
                       cfg.cuda_block_size, cfg.gpu_id);
                auto res = ga.run();
                best_fit = res.best_fitness;
                t = res.timings;
                used_cuda = res.used_cuda;
            } else if (app == "mnm") {
                NMGA ga(cfg.pop_size, cfg.gene_length, cfg.num_generations,
                        cfg.crossover_rate, cfg.mutation_rate, cfg.mutation_strength,
                        cfg.tournament_size, cfg.gene_min, cfg.gene_max, cfg.omp_threads, 1, trial_seed,
                        cfg.cuda_block_size, cfg.gpu_id);
                auto res = ga.run();
                best_fit = res.best_fitness;
                t = res.timings;
                used_cuda = res.used_cuda;
            } else if (app == "island") {
                IslandGA ga(cfg.pop_size, cfg.gene_length, cfg.num_generations,
                            cfg.crossover_rate, cfg.mutation_rate, cfg.mutation_strength,
                            cfg.tournament_size, cfg.gene_min, cfg.gene_max, cfg.num_islands, trial_seed,
                            cfg.cuda_block_size, cfg.gpu_id);
                auto res = ga.run();
                best_fit = res.best_fitness;
                t = res.timings;
                used_cuda = res.used_cuda;
            }

            std::cout << "Done! Total: " << std::fixed << std::setprecision(4) << t.total_time 
                      << "s, Best Fitness: " << std::setprecision(6) << best_fit << std::endl;

            // Output row to CSV
            csv << app << ","
                << trial << ","
                << trial_seed << ","
                << cfg.pop_size << ","
                << cfg.num_generations << ","
                << cfg.gene_length << ","
                << (app == "island" ? cfg.num_islands : 1) << ","
                << cfg.omp_threads << ","
                << cfg.cuda_block_size << ","
                << cfg.gpu_id << ","
                << (used_cuda ? "1" : "0") << ","
                << std::setprecision(8) << best_fit << ","
                << std::setprecision(6) << t.total_time << ","
                << t.init_time << ","
                << t.selection_time << ","
                << t.crossover_time << ","
                << t.mutation_time << ","
                << t.fitness_time << ","
                << t.gpu_kernel_time << ","
                << t.gpu_host_to_device_time << ","
                << t.gpu_device_to_host_time << ","
                << t.gpu_sync_time << ","
                << t.communication_time << "\n";
            csv.flush();
        }
    }

    csv.close();
    std::cout << "\n================================================================================" << std::endl;
    std::cout << "Benchmark complete. Results successfully saved to: " << cfg.csv_output << std::endl;
    std::cout << "================================================================================" << std::endl;
}

int main(int argc, char** argv) {
    BenchmarkConfig cfg;

    #ifdef _OPENMP
    cfg.omp_threads = omp_get_max_threads();
    #endif

    if (argc > 1) cfg.pop_size = std::stoi(argv[1]);
    if (argc > 2) cfg.gene_length = std::stoi(argv[2]);
    if (argc > 3) cfg.num_generations = std::stoi(argv[3]);
    if (argc > 4) cfg.num_trials = std::stoi(argv[4]);
    if (argc > 5) cfg.num_islands = std::stoi(argv[5]);
    if (argc > 6) cfg.omp_threads = std::stoi(argv[6]);
    if (argc > 7) cfg.csv_output = argv[7];

    std::cout << "=== Parallel Genetic Algorithm Research Benchmarking Suite ===" << std::endl;
    std::cout << "CUDA Available: " << (GPUFitnessEvaluator::isCudaAvailable() ? "Yes" : "No") << std::endl;
    if (GPUFitnessEvaluator::isCudaAvailable()) {
        std::cout << "GPU Device: " << GPUFitnessEvaluator::getDeviceName(cfg.gpu_id) << std::endl;
    }
    std::cout << "CPU Host Threads: " << cfg.omp_threads << std::endl;

    run_all_benchmarks(cfg);
    return 0;
}
