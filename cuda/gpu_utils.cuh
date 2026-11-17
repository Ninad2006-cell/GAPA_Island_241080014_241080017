#ifndef GPU_UTILS_CUH
#define GPU_UTILS_CUH

#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <stdexcept>
#include "../common/config.h"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while(0)

__global__ void rastrigin_kernel(const double* __restrict__ genes, 
                                double* __restrict__ fitness, 
                                int gene_length, 
                                int pop_size);

__global__ void sphere_kernel(const double* __restrict__ genes, 
                             double* __restrict__ fitness, 
                             int gene_length, 
                             int pop_size);

class GPUTimingTracker {
private:
    cudaEvent_t ev_h2d_start, ev_h2d_stop;
    cudaEvent_t ev_kernel_start, ev_kernel_stop;
    cudaEvent_t ev_d2h_start, ev_d2h_stop;
    cudaEvent_t ev_sync_start, ev_sync_stop;

public:
    double host_to_device_time = 0.0;
    double kernel_time = 0.0;
    double device_to_host_time = 0.0;
    double sync_time = 0.0;
    double total_gpu_time = 0.0;

    GPUTimingTracker() {
        CUDA_CHECK(cudaEventCreate(&ev_h2d_start));
        CUDA_CHECK(cudaEventCreate(&ev_h2d_stop));
        CUDA_CHECK(cudaEventCreate(&ev_kernel_start));
        CUDA_CHECK(cudaEventCreate(&ev_kernel_stop));
        CUDA_CHECK(cudaEventCreate(&ev_d2h_start));
        CUDA_CHECK(cudaEventCreate(&ev_d2h_stop));
        CUDA_CHECK(cudaEventCreate(&ev_sync_start));
        CUDA_CHECK(cudaEventCreate(&ev_sync_stop));
    }

    ~GPUTimingTracker() {
        cudaEventDestroy(ev_h2d_start);
        cudaEventDestroy(ev_h2d_stop);
        cudaEventDestroy(ev_kernel_start);
        cudaEventDestroy(ev_kernel_stop);
        cudaEventDestroy(ev_d2h_start);
        cudaEventDestroy(ev_d2h_stop);
        cudaEventDestroy(ev_sync_start);
        cudaEventDestroy(ev_sync_stop);
    }

    void reset() {
        host_to_device_time = 0.0;
        kernel_time = 0.0;
        device_to_host_time = 0.0;
        sync_time = 0.0;
        total_gpu_time = 0.0;
    }

    void startHostToDevice(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(ev_h2d_start, stream));
    }

    void stopHostToDevice(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(ev_h2d_stop, stream));
        CUDA_CHECK(cudaEventSynchronize(ev_h2d_stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev_h2d_start, ev_h2d_stop));
        host_to_device_time += (ms / 1000.0);
    }

    void startKernel(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(ev_kernel_start, stream));
    }

    void stopKernel(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(ev_kernel_stop, stream));
        CUDA_CHECK(cudaEventSynchronize(ev_kernel_stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev_kernel_start, ev_kernel_stop));
        kernel_time += (ms / 1000.0);
    }

    void startDeviceToHost(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(ev_d2h_start, stream));
    }

    void stopDeviceToHost(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(ev_d2h_stop, stream));
        CUDA_CHECK(cudaEventSynchronize(ev_d2h_stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev_d2h_start, ev_d2h_stop));
        device_to_host_time += (ms / 1000.0);
    }

    void startSync(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(ev_sync_start, stream));
    }

    void stopSync(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(ev_sync_stop, stream));
        CUDA_CHECK(cudaEventSynchronize(ev_sync_stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev_sync_start, ev_sync_stop));
        sync_time += (ms / 1000.0);
    }

    void computeTotal() {
        total_gpu_time = host_to_device_time + kernel_time + device_to_host_time + sync_time;
    }
};

// GPU Fitness Evaluator managing device allocations and launches
class GPUFitnessEvaluator {
private:
    double* d_genes;
    double* d_fitness;
    int allocated_pop_size;
    int gene_length;
    int gpu_id;
    int threads_per_block;
    bool initialized;

public:
    GPUFitnessEvaluator(int max_pop_size, int g_len, int device_id = DEFAULT_GPU_DEVICE_ID, 
                        int block_size = DEFAULT_CUDA_THREADS_PER_BLOCK)
        : d_genes(nullptr), d_fitness(nullptr), allocated_pop_size(max_pop_size),
          gene_length(g_len), gpu_id(device_id), threads_per_block(block_size), initialized(false) {
        
        int count = 0;
        cudaError_t err = cudaGetDeviceCount(&count);
        if (err != cudaSuccess || count <= 0) {
            std::cerr << "Warning: No CUDA device detected." << std::endl;
            return;
        }

        if (gpu_id >= count) {
            std::cerr << "Warning: Requested GPU " << gpu_id << " out of range (count=" << count << "). Defaulting to GPU 0." << std::endl;
            gpu_id = 0;
        }

        CUDA_CHECK(cudaSetDevice(gpu_id));
        CUDA_CHECK(cudaMalloc(&d_genes, static_cast<size_t>(allocated_pop_size) * gene_length * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_fitness, static_cast<size_t>(allocated_pop_size) * sizeof(double)));
        initialized = true;
    }

    ~GPUFitnessEvaluator() {
        if (initialized) {
            cudaSetDevice(gpu_id);
            if (d_genes) cudaFree(d_genes);
            if (d_fitness) cudaFree(d_fitness);
            d_genes = nullptr;
            d_fitness = nullptr;
            initialized = false;
        }
    }

    bool isInitialized() const {
        return initialized;
    }

    int getGpuId() const {
        return gpu_id;
    }

    void evaluate_rastrigin(const double* host_genes, double* host_fitness, 
                            int pop_size, GPUTimingTracker& tracker, cudaStream_t stream = 0) {
        if (!initialized) {
            throw std::runtime_error("GPUFitnessEvaluator: CUDA not initialized on device.");
        }
        if (pop_size > allocated_pop_size) {
            throw std::runtime_error("GPUFitnessEvaluator: pop_size exceeds allocated device buffer.");
        }

        CUDA_CHECK(cudaSetDevice(gpu_id));

        // 1. Host to Device transfer
        tracker.startHostToDevice(stream);
        CUDA_CHECK(cudaMemcpyAsync(d_genes, host_genes, 
                                  static_cast<size_t>(pop_size) * gene_length * sizeof(double), 
                                  cudaMemcpyHostToDevice, stream));
        tracker.stopHostToDevice(stream);

        // 2. Launch Kernel
        int blocks = (pop_size + threads_per_block - 1) / threads_per_block;
        tracker.startKernel(stream);
        rastrigin_kernel<<<blocks, threads_per_block, 0, stream>>>(d_genes, d_fitness, gene_length, pop_size);
        CUDA_CHECK(cudaGetLastError());
        tracker.stopKernel(stream);

        // 3. Device to Host transfer
        tracker.startDeviceToHost(stream);
        CUDA_CHECK(cudaMemcpyAsync(host_fitness, d_fitness, 
                                  static_cast<size_t>(pop_size) * sizeof(double), 
                                  cudaMemcpyDeviceToHost, stream));
        tracker.stopDeviceToHost(stream);

        // 4. Stream Synchronization
        tracker.startSync(stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        tracker.stopSync(stream);
    }

    static bool isCudaAvailable() {
        int count = 0;
        cudaError_t err = cudaGetDeviceCount(&count);
        return (err == cudaSuccess && count > 0);
    }

    static std::string getDeviceName(int device_id = 0) {
        int count = 0;
        cudaError_t err = cudaGetDeviceCount(&count);
        if (err != cudaSuccess || count <= 0 || device_id >= count) {
            return "None";
        }
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
        return std::string(prop.name);
    }

    static int getDeviceCount() {
        int count = 0;
        cudaError_t err = cudaGetDeviceCount(&count);
        return (err == cudaSuccess) ? count : 0;
    }
};

#endif 