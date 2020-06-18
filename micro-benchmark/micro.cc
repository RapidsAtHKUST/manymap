#include <string.h>
#include <stdio.h>
#include <assert.h>
#include <pthread.h>
#include "micro.h"

typedef struct {
	int nseq, qlen, tlen, flag;
	uint8_t *query;
	uint8_t *target;
	int8_t *mat;
	ksw_extz_t *ez;
} tdata;

void *ksw_dispatch(void *data) {
	tdata *d = (tdata*)data;
	for (int i = 0; i < d->nseq; ++i) {
#if defined(__AVX512BW__)
	ksw_extd2_avx512(NULL, d->qlen, d->query, d->tlen, d->target, 5, d->mat, 4, 2, 24, 1, MAX(d->qlen, d->tlen), 400, -1, d->flag, d->ez + i);
#elif defined(__AVX2__)
	ksw_extd2_avx2(NULL, d->qlen, d->query, d->tlen, d->target, 5, d->mat, 4, 2, 24, 1, MAX(d->qlen, d->tlen), 400, -1, d->flag, d->ez + i);
#elif defined(__SSE4_1__)
	ksw_extd2_sse41(NULL, d->qlen, d->query, d->tlen, d->target, 5, d->mat, 4, 2, 24, 1, MAX(d->qlen, d->tlen), 400, -1, d->flag, d->ez + i);
#elif defined(__SSE2__)
	ksw_extd2_sse2(NULL, d->qlen, d->query, d->tlen, d->target, 5, d->mat, 4, 2, 24, 1, MAX(d->qlen, d->tlen), 400, -1, d->flag, d->ez + i);
#endif
	}
	return NULL;
}

void ksw_extd2(int nthread, int nseq, int qlen, uint8_t *query, int tlen, uint8_t *target, int8_t *mat, int flag) {
	pthread_t *tid;
	tdata *td;
	tid = (pthread_t*)malloc(nthread * sizeof(pthread_t));
	td = (tdata*)malloc(nthread * sizeof(tdata));

	ksw_extz_t *ez;
	ez = (ksw_extz_t*)malloc(nseq * sizeof(ksw_extz_t));
	memset(ez, 0, nseq * sizeof(ksw_extz_t));
	for (int i = 0; i < nseq; ++i) {
		ksw_reset_extz(&ez[i]);
	}

	for (int i = 0; i < nthread; ++i) {
		td[i].nseq = nseq / nthread;
		td[i].qlen = qlen;
		td[i].tlen = tlen;
		td[i].query = query;
		td[i].target = target;
		td[i].mat = mat;
		td[i].flag = flag;
		td[i].ez = &ez[i * nseq / nthread];
	}

	double mm_realtime0 = realtime();
	for (int i = 0; i < nthread; ++i) {
		pthread_create(&tid[i], NULL, ksw_dispatch, &td[i]);
	}

	for (int i = 0; i < nthread; ++i) {
		pthread_join(tid[i], NULL);
	}

	// for (int i = 0; i < nthread; ++i) {
	// 	printf("max: %u, zdropped: %u, max_q: %d, max_t: %d, mqe: %d, mqe_t: %d, mte: %d, mte_q: %d, score: %d, m_cigar: %d, n_cigar: %d, reach_end: %d\n",
	// 		ez[i].max, ez[i].zdropped, ez[i].max_q, ez[i].max_t, ez[i].mqe, ez[i].mqe_t, ez[i].mte, ez[i].mte_q, ez[i].score, ez[i].m_cigar, ez[i].n_cigar, ez[i].reach_end);
	// }

	// printf("[M::%s] Real time: %.3f sec; CPU: %.3f sec; Peak RSS: %.3f GB\n", __func__, realtime() - mm_realtime0, cputime(), peakrss() / 1024.0 / 1024.0 / 1024.0);
	double curtime = realtime();
	printf("%.3f GCUPS, %.3f seconds, %.3f GB\n", (double)nseq * qlen * tlen / (curtime - mm_realtime0) / 1000000000, curtime - mm_realtime0, peakrss() / 1024.0 / 1024.0 / 1024.0);
}

int main(int argc, char **argv) {
	int nthread = atoi(argv[1]);
	int nseq = atoi(argv[2]);
	int flag = atoi(argv[3]);

	FILE * fp;
	fp = fopen(argv[4], "r");

	int qlen, tlen;
	uint8_t *query, *target;
	query = (uint8_t*)malloc(MAXLEN * sizeof(uint8_t));
	target = (uint8_t*)malloc(MAXLEN * sizeof(uint8_t));
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
/*
	for (int i = 0; i < qlen; ++i) {
		printf("%d", query[i]);
	}
	printf("\n");
	for (int i = 0; i < tlen; ++i) {
		printf("%d", target[i]);
	}
	printf("\n");

	return 0;
*/

	int8_t *mat;
	mat = (int8_t*)malloc(25 * sizeof(int8_t));
	ksw_gen_simple_mat(5, mat, 2, 4, 1);

	ksw_extd2(nthread, nseq, qlen, query, tlen, target, mat, flag);

	return 0;
}
