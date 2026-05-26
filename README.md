# 🎥 Real-Time HDR Tone Mapping with CUDA

This project implements a high-performance **HDR Tone Mapping** filter designed for real-time video processing. By utilizing NVIDIA CUDA, the implementation achieves significant speedup compared to traditional CPU processing, enabling real-time frame rates for Full HD (1080p) streams.

## 🚀 Performance Results
The following benchmarks were conducted on an **NVIDIA GeForce GTX 1070** using a **1920x1080** resolution.

| Implementation | Execution Time (ms) | Speedup | FPS Capability |
| :--- | :--- | :--- | :--- |
| Sequential CPU | 244.569 ms | 1.0x | ~4 FPS |
| **Optimized GPU (Shared Memory)** | **1.733 ms** | **141.07x** | **~577 FPS** |

---

## 🛠️ Optimization & Technical Details

### 1. Global Tone Mapping Operator
We implemented the **Reinhard Tone Mapping** operator, which intelligently compresses the luminance range. This ensures that details in both extremely bright and dark regions are preserved, simulating High Dynamic Range.
- **Formula:** $L_{out} = \frac{L_{in}}{1 + L_{in}}$

### 2. Shared Memory Utilization
To meet the real-time requirements, the kernel uses **Shared Memory Tiling**. 
- Each 16x16 pixel block is first loaded into high-speed on-chip Shared Memory.
- This reduces the number of expensive Global Memory transactions, allowing the GTX 1070 to process the frame in under 2ms.

### 3. Real-Time Parameters
The filter framework supports dynamic adjustments for:
- **Exposure:** Controls the overall brightness scaling before tone mapping.
- **Gamma Correction:** Corrects the non-linear brightness perception of the human eye (standard $\gamma=2.2$).

### 4. Color Space Handling
The implementation processes RGB data by normalizing pixel values to a $[0.0, 1.0]$ float range, applying the HDR operator, and performing gamma correction before converting back to 8-bit integers.

---

## 💻 Compilation & Usage

Ensure you have the CUDA Toolkit installed. This version uses a synthetic stream generator for performance benchmarking (OpenCV independent).

```bash
nvcc main.cpp filter_utils.cpp hdr_kernels.cu -o hdr_test
./hdr_test
