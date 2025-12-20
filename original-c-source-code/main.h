#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "graphic.h"

#define MAX 1000
#define NMAX 10

extern int endf;
extern int in, ot, hd, hd2;
extern double ot_in[NMAX+1][MAX+1], ot_ot[NMAX+1][MAX+1], del_ot[NMAX+1][MAX+1][2];
extern double indata_input[MAX+1], indata_tch[MAX+1];
extern double w_ot_ot[NMAX+1][MAX+1][MAX+1];
extern double ow[MAX+1];
extern double alpha, beta;
extern double u0, u1;
extern double inival1, inival2;
extern int f[15];
extern int all;
extern double err, res;
extern int count, t_loop;

extern double sgn(double x);
extern double rnd();
extern double sigmf(double u);

extern void neuro_param_input();
extern void neuro_output_calc(double* indata_input);
extern void neuro_teach_calc(double* indata_tch);
extern void neuro_weight_calc();
extern void neuro_init(int in0, int ot0, int hd0, int hd20);
extern void neuro_calc(double* input, double* teach);
extern void neuro_output_write(int wr, double* indata_tch);
extern int neuro_weight_write(int loop, int wr, int pa);
