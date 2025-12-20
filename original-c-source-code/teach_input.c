#include "main.h"

extern double g_indata_input[MAX+1][MAX+1], g_indata_tch[MAX+1][MAX+1];
extern int in, ot, hd, hd2;
extern int pat[MAX+1];
extern int times;
extern int loop, loopl, wr;
extern int pa, po;

extern void inputi(char* s, int* i, int def);
extern void inputf(char* s, double* f, double def);

void teach_input()
{
  int inp, k, l, m, n, loop1;
  int check[MAX+1];
  double inival;

  inputi("input random ?(0) ", &inp, 0);
  for (k = 0; k <= ot-1; k++) {
    inputi("input pattern 0-random 1-parity 2-mirror 3-manual 4-real random 5-only one ?(1) ", &m, 1);
    pat[k] = m;
  }
  for (k = 0; k <= pa-1; k++) {
    for (l = 0; l <= in / 2 - 1; l++) {
      if ( inp == 0 ) {
	if ( (k & (1 << l)) )
	  g_indata_input[k][l] = 1.0;
	else
	  g_indata_input[k][l] = 0.0;
      }
      else
	g_indata_input[k][l] = rnd(); 
    }
    for (m = 0; m <= pa-1; m++)
      check[m] = -1; 
    for (n = 0; n <= ot-1; n++) {
      if ( pat[n] == 0 )
	if ( rnd() > 0.5 )
	  g_indata_tch[k][n] = 1.0; 
	else
	  g_indata_tch[k][n] = 0.0; 
      else if ( pat[n] == 4 )
	g_indata_tch[k][n] = rnd(); 
      else if ( pat[n] == 1 ) {
	m=0; 
	for (l = 0; l <= in/2-1; l++)
	  if ( g_indata_input[k][l]>0.5 ) m++;
	if ( m % 2 == 1 )
	  g_indata_tch[k][n] = 1.0; 
	else
	  g_indata_tch[k][n] = 0.0; 
      }
      else if ( pat[n] == 2 ) {
	m = 0; 
	for (l = 0; l <= in/4-1; l++)
	  if ( g_indata_input[k][l] != g_indata_input[k][in/2-1-l] ) m = 1;
	if ( m == 1 )
	  g_indata_tch[k][n] = 0.0; 
	else
	  g_indata_tch[k][n] = 1.0; 
      }
      else if ( pat[n] == 3 ) {
	for (l = 0; l <= in/2-1; l++)
	  printf("%4.2f ", g_indata_input[k][l]); 
	printf("ot:%2d", n); 
	putchar('\n'); 
	inputf("tch ? ", &inival, 0.0); 
	g_indata_tch[k][n] = inival; 
      }
      else if ( pat[n] == 5 ) {
	for (l = 0; l <= pa-1; l++) {
	  g_indata_tch[l][n] = 0.0; 
	}
	do {
	  loop1 = (int)(rnd() * pa); 
	} while (!(check[loop1] == -1)); 
	check[loop1] = 1; 
	g_indata_tch[loop1][n] = 1.0; 
      }
    }
  }
}

