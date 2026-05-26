#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <chrono>
#include <cmath>
#include "filter_utils.h"

extern "C" void apply_hdr(unsigned char* d_input, unsigned char* d_output, int width, int height, float exposure, float gamma);

// hdr calculation on cpu
void hdr_cpu(unsigned char* input, unsigned char* output, int width, int height, float exposure, float gamma) {
    for (int i = 0; i < width * height * 3; i++) {
        float pixel = (float)input[i] / 255.0f;
        pixel *= exposure;
        pixel = pixel / (1.0f + pixel);
        pixel = powf(pixel, 1.0f / gamma);
        output[i] = (unsigned char)(pixel * 255.0f);
    }
}

int main() {
    const int width = 1920;
    const int height = 1080;
    const size_t size = width * height * 3 * sizeof(unsigned char);
    const float exposure = 1.5f;
    const float gamma = 2.2f;

    std::vector<unsigned char> h_input(width * height * 3, 180); 
    std::vector<unsigned char> h_output_gpu(width * height * 3);
    std::vector<unsigned char> h_output_cpu(width * height * 3);

    // cpu time measurement
    auto start_cpu = std::chrono::high_resolution_clock::now();
    hdr_cpu(h_input.data(), h_output_cpu.data(), width, height, exposure, gamma);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_ms = end_cpu - start_cpu;

    unsigned char *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    cudaMemcpy(d_input, h_input.data(), size, cudaMemcpyHostToDevice);

    // gpu time measurement
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    apply_hdr(d_input, d_output, width, height, exposure, gamma);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_ms = 0;
    cudaEventElapsedTime(&gpu_ms, start, stop);

    // results
    std::cout << "--- HDR Tone Mapping Performance (1920x1080) ---" << std::endl;
    std::cout << "CPU Time: " << cpu_ms.count() << " ms" << std::endl;
    std::cout << "GPU Time: " << gpu_ms << " ms" << std::endl;
    std::cout << "Speedup: " << cpu_ms.count() / gpu_ms << "x" << std::endl;

    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}