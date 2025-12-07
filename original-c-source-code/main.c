#include "main.h"

double g_indata_input[MAX+1][MAX+1], g_indata_tch[MAX+1][MAX+1];

double sgn(double x) { return (x > 0.0) ? 1.0 : ((x == 0.0) ? 0.0 : -1.0); }
double sigmf(double u) { return(1 / (1 + exp((double)(-2 * u / u0)))); }

int in, ot, hd, hd2;
double ot_in[NMAX+1][MAX+1], ot_ot[NMAX+1][MAX+1], del_ot[NMAX+1][MAX+1][2];
double indata_input[MAX+1], indata_tch[MAX+1];
double w_ot_ot[NMAX+1][MAX+1][MAX+1];
double ow[MAX+1];
double alpha, beta;
double u0, u1;
double inival1, inival2;
int f[15];
int all;
double err, res;
int count, t_loop;
int pat[MAX+1];
int times;
int loop, loopl, wr;
int pa, po;

extern void teach_input();

void inputi(char* s, int* i, int def)
{
  char buf[128];

  printf("%s", s);
  if (gets(buf) && *buf) *i = atoi(buf);
  else *i = def;
}

void inputf(char* s, double* f, double def)
{
  char buf[128];

  printf("%s", s);
  if (gets(buf) && *buf) *f = atof(buf);
  else *f = def;
}

double rnd()
{
  return ((rand() % 10000) / 10000.0);
}

void locate(int x, int y)
{
  putchar(27);
  printf("[%d;%dH",y+1,x+1);
}

void cls()
{
  putchar(27);
  printf("[2J");
  locate(0,0);
}

main()
{
  int seed;

  init();
  cls();

  inputi("seed ? ", &seed, 1);
  srand(seed);
  inputi("in ?(4) ", &in, 4);
  inputi("pa ?(16) ", &pa, 16);
  inputi("ot ?(1) ", &ot, 1);
  in *= 2;

  teach_input();

  inputi("hd ?(8) ", &hd, 8); 
  inputi("hd2 ?(0) ", &hd2, 0); 
  
  inputi("data write(0-3) ", &wr, 0); 
  inputi("write position ? ", &po, 0); 

  neuro_param_input();

  neuro_init(in, ot, hd, hd2);
  cls();
  XClearWindow(d, w);

  loop = 0; 
  box(50, 40, times+50, 250); 
  for (;;) {
    loop++;
    if (wr == 3)
      locate(0, 0);
    for (loopl = 0; loopl <= pa-1; loopl++) {
      switch (wr) {
      case 0:
      case 1:
	locate(0, loopl + po);
	break;
      case 2:
	locate(loopl / (30-po) * 20, loopl % (30 - po) + po);
	break;
      }
      neuro_calc(g_indata_input[loopl], g_indata_tch[loopl]);
      neuro_output_write(wr, g_indata_tch[loopl]);
    }
    if (neuro_weight_write(loop, wr, pa)) {
      break;
    }
  }
}


