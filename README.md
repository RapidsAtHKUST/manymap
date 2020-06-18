# manymap

This repository contains the source code of the paper "Accelerating Long Read Alignment on Three Processors", by Zonghao Feng, Shuang Qiu, Lipeng Wang, and Qiong Luo.

### Micro Benchmarks

Evaluation of the base-level alignment kernel. Run `make` to compile the following executables:

`ksw_{sse2, sse41, avx2, avx512}`: The original kernel in minimap2.

`cpu_{sse2, sse41, avx2, avx512}`: CPU version of the optimized kernel.

`knl`: KNL (Intel Xeon Phi processor) version.

`gpu`: GPU version. Variants: single block / multiple blocks with cooperative groups, global memory / shared memory (applicable to small data), score only / complete path

### Macro Benchmarks

Evaluation of end-to-end runtime. Run `make` to compile. To use the MCDRAM on KNL, add `numactl -m 1` before the run command.

### Acknowledgment

This project is based on [minimap2](https://github.com/lh3/minimap2) (version 2.16).
