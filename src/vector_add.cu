#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// ==========================================
// 1. GPU 向量加法核函数 (Kernel)
// ==========================================
__global__ void vector_add_kernel(const float* A, const float* B, float* C, int n) {
    // 灵魂公式：计算当前线程在全显卡中的全局唯一索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 边界保护
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

// ==========================================
// 2. CPU 向量加法 (单核基准对照)
// ==========================================
void vector_add_cpu(const float* A, const float* B, float* C, int n) {
    for (int i = 0; i < n; i++) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    // 测试数据量: 10,000,000 个浮点数 (约 40 MB 数据)
    int n = 10000000;
    size_t bytes = n * sizeof(float);
    printf("====================================================\n");
    printf("🚀 CUDA 第 5 周实验: 10,000,000 规模向量加法性能基准\n");
    printf("数据大小: %.2f MB\n", (float)bytes / (1024 * 1024));
    printf("====================================================\n\n");

    // 1. 在主机 (CPU) 上分配内存并初始化
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C_cpu = (float*)malloc(bytes);
    float *h_C_gpu = (float*)malloc(bytes);

    for (int i = 0; i < n; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // 2. 测量 CPU 基准耗时
    cudaEvent_t cpu_start, cpu_end;
    cudaEventCreate(&cpu_start);
    cudaEventCreate(&cpu_end);
    
    cudaEventRecord(cpu_start);
    vector_add_cpu(h_A, h_B, h_C_cpu, n);
    cudaEventRecord(cpu_end);
    cudaEventSynchronize(cpu_end);
    
    float cpu_time_ms = 0;
    cudaEventElapsedTime(&cpu_time_ms, cpu_start, cpu_end);
    printf("[CPU 单核基准] 耗时: %.2f ms\n\n", cpu_time_ms);

    // 3. 在设备 (GPU) 上分配显存并将数据拷贝过去 (H2D)
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // 4. 探索不同 Block Size 对 GPU 性能的影响
    int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    int num_tests = sizeof(block_sizes) / sizeof(block_sizes[0]);

    printf("--- [GPU 实验: Block Size 对耗时与带宽的影响] ---\n");
    printf("| Block Size | Grid Size | 耗时 (ms) | 有效显存带宽 (GB/s) |\n");
    printf("|---|---|---|---|\n");

    cudaEvent_t gpu_start, gpu_end;
    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_end);

    for (int i = 0; i < num_tests; i++) {
        int block_size = block_sizes[i];
        int grid_size = (n + block_size - 1) / block_size;

        // 预热 3 次
        for (int w = 0; w < 3; w++) {
            vector_add_kernel<<<grid_size, block_size>>>(d_A, d_B, d_C, n);
        }
        cudaDeviceSynchronize();

        // 正式测量 10 次取平均值
        int iters = 10;
        cudaEventRecord(gpu_start);
        for (int it = 0; it < iters; it++) {
            vector_add_kernel<<<grid_size, block_size>>>(d_A, d_B, d_C, n);
        }
        cudaEventRecord(gpu_end);
        cudaEventSynchronize(gpu_end);

        float total_gpu_ms = 0;
        cudaEventElapsedTime(&total_gpu_ms, gpu_start, gpu_end);
        float avg_gpu_ms = total_gpu_ms / iters;

        // 计算有效带宽: 读 A、读 B、写 C，总搬运 3 * bytes 数据
        float bandwidth_gb_s = (3.0f * bytes) / (avg_gpu_ms / 1000.0f) / 1e9;

        printf("| %10d | %9d | %9.3f | %19.2f |\n", block_size, grid_size, avg_gpu_ms, bandwidth_gb_s);
    }

    // 5. 将结果从 GPU 拷回 CPU 并做正确性校验 (D2H)
    cudaMemcpy(h_C_gpu, d_C, bytes, cudaMemcpyDeviceToHost);

    float max_err = 0.0f;
    for (int i = 0; i < n; i++) {
        float err = fabsf(h_C_cpu[i] - h_C_gpu[i]);
        if (err > max_err) max_err = err;
    }
    printf("\n[正确性校验] CPU 与 GPU 结果最大误差: %.6f\n", max_err);
    if (max_err < 1e-5) {
        printf("✅ [校验通过] GPU 向量加法结果与 CPU 严格一致！\n");
    } else {
        printf("❌ [校验失败] 发现数值不匹配！\n");
    }

    // 释放内存与显存
    free(h_A); free(h_B); free(h_C_cpu); free(h_C_gpu);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(cpu_start); cudaEventDestroy(cpu_end);
    cudaEventDestroy(gpu_start); cudaEventDestroy(gpu_end);

    return 0;
}