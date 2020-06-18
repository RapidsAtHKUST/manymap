#include <string.h>
#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <sys/time.h>
// #include <omp.h>
// #include <cooperative_groups.h>
#include "micro.h"

// using namespace cooperative_groups;

#define MCH 2
#define MIS -4
#define M1 4
#define SCN -1

#define WIDTH 32
#define SHIFT 5

#define NSTREAM 128
// #define FLAG 9

// #define MAXLEN 15000
// #define MAXCIGAR 16384 // 15000
#define MAXMEM 500000 // 15000, 9

__global__ void ksw_extd2_kernel(int qlen, const uint8_t *query, int tlen, const uint8_t *target,
							int8_t q, int8_t e, int8_t q2, int8_t e2, int w, int zdrop, ksw_extz_t *ez) {
	extern __shared__ int8_t smem[];
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int elementskip = blockDim.x * gridDim.x;
	int tlen_ = (tlen + WIDTH - 1) / WIDTH, qlen_ = (qlen + WIDTH - 1) / WIDTH;
	int last_H0_t = 0, H0 = 0;
	int t, r;

	// grid_group g = this_grid();

	int8_t *u, *v, *x, *y, *x2, *y2;
	uint8_t *qr;
	const uint8_t *sf;

	int long_thres = e != e2? (q2 - q) / (e - e2) - 1 : 0;
	if (q2 + e2 + long_thres * e2 > q + e + long_thres * e)
		++long_thres;
	int long_diff = long_thres * (e - e2) - (q2 - q) - e2;

	u = smem;
	v = u + tlen_ * WIDTH, x = v + qlen_ * WIDTH, y = x + qlen_ * WIDTH;
	x2 = y + tlen_ * WIDTH, y2 = x2 + qlen_ * WIDTH, qr = (uint8_t*)(y2 + tlen_ * WIDTH);
	sf = target;

	for (t = tid; t < tlen_ * WIDTH; t += elementskip) {
		u[t] = y[t] = -q - e;
		y2[t] = -q2 - e2;
	}
	for (t = tid; t < qlen_ * WIDTH; t += elementskip) {
		v[t] = x[t] = -q -e;
		x2[t] = -q2 - e2;
	}
	for (t = tid; t < qlen; t += elementskip) qr[t] = query[qlen - 1 - t];

	for (r = 0; r < qlen + tlen - 1; ++r) {
		int st = 0, en = tlen - 1;
		uint8_t *qrr = qr + (qlen - 1 - r);
		int8_t *v8 = v + qlen - r - 1, *x8 = x + qlen - r - 1, *x28 = x2 + qlen - r - 1;

		if (st < r - qlen + 1) st = r - qlen + 1;
		if (en > r) en = r;
		if (st < (r-w+1)>>1) st = (r-w+1)>>1; // take the ceil
		if (en > (r+w)>>1) en = (r+w)>>1; // take the floor

		if (st == 0 && tid == 0) {
			v8[0] = r == 0? -q - e : r < long_thres? -e : r == long_thres? long_diff : -e2;
		}
		if (en >= r && tid == 0) {
			((int8_t*)y)[r] = -q - e, ((int8_t*)y2)[r] = -q2 - e2;
			u[r] = r == 0? -q - e : r < long_thres? -e : r == long_thres? long_diff : -e2;
		}

		// g.sync();
		__syncthreads();

		for (t = st + tid; t <= en; t += elementskip) {
			int8_t z, a, b, a2, b2, vt, ut;
			int8_t st = sf[t], qt = qrr[t];

			z = (st == qt) ? MCH : MIS;
			z = ((st == M1) || (qt == M1)) ? SCN : z;
			ut = u[t];
			
			vt = v8[t];

			a = x8[t] + vt;
			b = y[t] + ut;
			a2 = x28[t] + vt;
			b2 = y2[t] + ut;

			z = MAX(z, MAX(a, MAX(b, MAX(a2, b2))));
			z = MIN(z, MCH);
			u[t] = z - vt;
			v8[t] = z - ut;
			x8[t] = MAX(a - z + q, 0) - q - e;
			y[t] = MAX(b - z + q, 0) - q - e;
			x28[t] = MAX(a2 - z + q2, 0) - q2 - e2;
			y2[t] = MAX(b2 - z + q2, 0) - q2 - e2;
		}

		// g.sync();
		__syncthreads();

		if (tid == 0) {
			if (r > 0) {
				if (last_H0_t >= st && last_H0_t <= en && last_H0_t + 1 >= st && last_H0_t + 1 <= en) {
					int32_t d0 = v8[last_H0_t];
					int32_t d1 = u[last_H0_t + 1];
					if (d0 > d1) H0 += d0;
					else H0 += d1, ++last_H0_t;
				} else if (last_H0_t >= st && last_H0_t <= en) {
					H0 += v8[last_H0_t];
				} else {
					++last_H0_t, H0 += u[last_H0_t];
				}
			} else H0 = v8[0] - q - e, last_H0_t = 0;
			// ksw_apply_zdrop_gpu(ez, 1, H0, r, last_H0_t, zdrop, e2);
			if (r == qlen + tlen - 2 && en == tlen - 1)
				ez->score = H0;
		}
	}
}

void ksw_extd2_gpu(int nkernel, int nblock, int nthread, int qlen, uint8_t *query, int tlen, uint8_t *target, int8_t m, int8_t *mat)
{	
	cudaStream_t *streams = (cudaStream_t*)malloc(NSTREAM * sizeof(cudaStream_t));
	// cudaEvent_t startEvent, stopEvent;
	// float ms;
	// CHECK(cudaEventCreate(&startEvent));
	// CHECK(cudaEventCreate(&stopEvent));

	uint8_t *d_query, *d_target;
	// int8_t *d_mat;
	ksw_extz_t *d_ez;
	// uint32_t *d_cigar;
	// void *d_gm;
	// int *d_last_H0_t, *d_H0;

	ksw_extz_t *ez;
	// uint32_t **h_cigar;
	// int *h_m_cigar;

	CHECK(cudaMallocHost(&ez, nkernel * sizeof(ksw_extz_t)));
	// CHECK(cudaMallocHost((void**)&h_cigar, nkernel * sizeof(uint32_t*)));
	// CHECK(cudaMallocHost((void**)&h_m_cigar, nkernel * sizeof(int)));
	memset(ez, 0, nkernel * sizeof(ksw_extz_t));

	for (int i = 0; i < NSTREAM; ++i) {
		CHECK(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
	}

	CHECK(cudaMalloc(&d_query, MAXLEN * sizeof(uint8_t) * NSTREAM)); // max: 12123
	CHECK(cudaMalloc(&d_target, MAXLEN * sizeof(uint8_t) * NSTREAM)); // max: 12883
	// CHECK(cudaMalloc(&d_mat, m * m * sizeof(int8_t)));
	// CHECK(cudaMalloc(&d_mat, m * m * sizeof(int8_t) * NSTREAM));
	CHECK(cudaMalloc(&d_ez, sizeof(ksw_extz_t) * nkernel)); // stream -> kernel
	// CHECK(cudaMalloc(&d_cigar, MAXCIGAR * sizeof(uint32_t) * nkernel)); // stream -> kernel
	// CHECK(cudaMalloc(&d_gm, (size_t)MAXMEM * NSTREAM));
	// CHECK(cudaMalloc(&d_last_H0_t, sizeof(int) * NSTREAM));
	// CHECK(cudaMalloc(&d_H0, sizeof(int) * NSTREAM));

	// CHECK(cudaMemcpy(d_mat, mat, m * m * sizeof(int8_t), cudaMemcpyHostToDevice));

	for (int i = 0; i < nkernel; ++i) {
		ksw_reset_extz(&ez[i]);
		// ez[i].cigar = &d_cigar[i * MAXCIGAR];
	}

	double mm_realtime0 = realtime();
/*
	for (int i = 0; i < nkernel; ++i) {
		// CHECK(cudaMemcpyAsync(&d_mat[i * m * m], mat, m * m * sizeof(int8_t), cudaMemcpyHostToDevice, streams[i]));
		CHECK(cudaMemcpyAsync(&d_query[i * MAXLEN], query, qlen * sizeof(uint8_t), cudaMemcpyHostToDevice, streams[i]));
		CHECK(cudaMemcpyAsync(&d_target[i * MAXLEN], target, tlen * sizeof(uint8_t), cudaMemcpyHostToDevice, streams[i]));
		CHECK(cudaMemcpyAsync(&d_ez[i], &ez[i], sizeof(ksw_extz_t), cudaMemcpyHostToDevice, streams[i])); // TODO: only m_cigar is necessary	
	}
	fprintf(stderr, "[cudaMemcpyAsync] Timestamp: %.3f\n", realtime() - mm_realtime0);
*/
	int q = 4, e = 2, q2 = 24, e2 = 1, w = -1, zdrop = 400;
	if (w < 0) w = tlen > qlen? tlen : qlen;
	size_t smem_size = tlen * 8;

	// CHECK(cudaEventRecord(startEvent, 0));
/*
	for (int i = 0; i < nkernel; ++i) {
		void *gm = (void*)((int8_t*)d_gm + i * MAXMEM);
		uint8_t *dqi = &d_query[i * MAXLEN];
		uint8_t *dti = &d_target[i * MAXLEN];
		ksw_extz_t *dezi = &d_ez[i];

		void *args[] = {&gm, &qlen, &dqi, &tlen, &dti, &q, &e, &q2, &e2, &w, &zdrop, &dezi};
		cudaLaunchKernel((void*)ksw_extd2_kernel, nblock, nthread, args, 0, streams[i]);
		// CHECK(cudaDeviceSynchronize());
	}
	fprintf(stderr, "[kernel] Timestamp: %.3f\n", realtime() - mm_realtime0);
*/
	for (int i = 0; i < nkernel; ++i) {
		int sid = i % NSTREAM;
		CHECK(cudaMemcpyAsync(&d_query[sid * MAXLEN], query, qlen * sizeof(uint8_t), cudaMemcpyHostToDevice, streams[sid]));
		CHECK(cudaMemcpyAsync(&d_target[sid * MAXLEN], target, tlen * sizeof(uint8_t), cudaMemcpyHostToDevice, streams[sid]));
		CHECK(cudaMemcpyAsync(&d_ez[i], &ez[i], sizeof(ksw_extz_t), cudaMemcpyHostToDevice, streams[sid]));
		ksw_extd2_kernel<<<nblock, nthread, smem_size, streams[sid]>>>(qlen, &d_query[sid * MAXLEN], tlen, &d_target[sid * MAXLEN], q, e, q2, e2, w, zdrop, &d_ez[i]);
		CHECK(cudaMemcpyAsync(&ez[i], &d_ez[i], sizeof(ksw_extz_t), cudaMemcpyDeviceToHost, streams[sid]));
	}
/*
	for (int i = 0; i < nkernel; ++i) {
		CHECK(cudaMemcpyAsync(&ez[i], &d_ez[i], sizeof(ksw_extz_t), cudaMemcpyDeviceToHost, streams[i]));
	}
	fprintf(stderr, "[cudaMemcpyAsync] Timestamp: %.3f\n", realtime() - mm_realtime0);
*/
	CHECK(cudaDeviceSynchronize());
	// CHECK(cudaEventRecord(stopEvent, 0));
	// CHECK(cudaEventSynchronize(stopEvent));
	// CHECK(cudaEventElapsedTime(&ms, startEvent, stopEvent));

	// fprintf(stderr, "[cudaDeviceSynchronize] Timestamp: %.3f\n", realtime() - mm_realtime0);
/*
	for (int i = 0; i < nkernel; ++i) {
		printf("max: %u, zdropped: %u, max_q: %d, max_t: %d, mqe: %d, mqe_t: %d, mte: %d, mte_q: %d, score: %d, m_cigar: %d, n_cigar: %d, reach_end: %d\n",
			ez[i].max, ez[i].zdropped, ez[i].max_q, ez[i].max_t, ez[i].mqe, ez[i].mqe_t, ez[i].mte, ez[i].mte_q, ez[i].score, ez[i].m_cigar, ez[i].n_cigar, ez[i].reach_end);
	}
*/
	double curtime = realtime();
	// fprintf(stderr, "[M::%s] Real time: %.3f sec; CPU: %.3f sec; Peak RSS: %.3f GB\n", __func__, realtime() - mm_realtime0, cputime(), peakrss() / 1024.0 / 1024.0 / 1024.0);
	fprintf(stderr, "%.3f GCUPS, %.3f seconds, %.3f GB\n", (double)nkernel * qlen * tlen / (curtime - mm_realtime0) / 1000000000, curtime - mm_realtime0, peakrss() / 1024.0 / 1024.0 / 1024.0);
	// CHECK(cudaDeviceReset());
}

int main(int argc, char **argv) {
	int nkernel = atoi(argv[1]);
	int nblock = atoi(argv[2]);
	int nthread = atoi(argv[3]);

	FILE * fp;
	fp = fopen(argv[4], "r");

	int qlen, tlen;
	uint8_t *query, *target;
	CHECK(cudaMallocHost(&query, MAXLEN * sizeof(uint8_t)));
	CHECK(cudaMallocHost(&target, MAXLEN * sizeof(uint8_t)));
	char tmps[128];

	fscanf(fp, "%s%s", tmps, query);
	fscanf(fp, "%s%s", tmps, target);
	qlen = strlen((char*)query);
	tlen = strlen((char*)target);
	for (int i = 0; i < qlen; ++i) {
		query[i] = seq_nt4_table[query[i]];
	}
	for (int i = 0; i < tlen; ++i) {
		target[i] = seq_nt4_table[target[i]];
	}
	fclose(fp);

	int8_t *mat;
	CHECK(cudaMallocHost(&mat, 25 * sizeof(int8_t)));
	ksw_gen_simple_mat(5, mat, 2, 4, 1);

	// CHECK(cudaDeviceReset());
	ksw_extd2_gpu(nkernel, nblock, nthread, qlen, query, tlen, target, 5, mat);
	// CHECK(cudaDeviceSynchronize());

	return 0;
}
