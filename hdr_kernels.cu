#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

#define TILE_SIZE 16

__global__ void hdr_optimized_kernel(unsigned char* input, unsigned char* output, int width, int height, float exposure, float gamma) {
    // shared memory
    __shared__ unsigned char s_tile[TILE_SIZE * TILE_SIZE * 3];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int global_idx = (y * width + x) * 3;
    int local_idx = (ty * TILE_SIZE + tx) * 3;

    if (x < width && y < height) {
        s_tile[local_idx] = input[global_idx];
        s_tile[local_idx + 1] = input[global_idx + 1];
        s_tile[local_idx + 2] = input[global_idx + 2];
    }
    __syncthreads();

    // 2. calculation on shared memory
    if (x < width && y < height) {
        for (int i = 0; i < 3; i++) {
            float pixel = (float)s_tile[local_idx + i] / 255.0f;
            
            pixel *= exposure;
            pixel = pixel / (1.0f + pixel); // reinhard tone mapping
            pixel = powf(pixel, 1.0f / gamma);

            output[global_idx + i] = (unsigned char)(pixel * 255.0f);
        }
    }
}

extern "C" void apply_hdr(unsigned char* d_input, unsigned char* d_output, int width, int height, float exposure, float gamma) {
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((width + TILE_SIZE - 1) / TILE_SIZE, (height + TILE_SIZE - 1) / TILE_SIZE);
    hdr_optimized_kernel<<<blocks, threads>>>(d_input, d_output, width, height, exposure, gamma);
}