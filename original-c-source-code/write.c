#include "main.h"

extern void locate(int, int);

void neuro_output_write(int wr, double* indata_tch)
{
  int k;

  switch (wr) {
  case 0:
  case 1:
    printf("in:"); 
    for (k = 1; k <= in/2; k++)
      printf("%4.2f ", ot_in[0][k*2]); 
    printf("-> "); 
    printf("%7.5f, %4.2f ", ot_in[0][in+2], indata_tch[0]); 
    printf("hd: "); 
    for (k = in+3; k <= in+6; k++)
      if ( k <= all+1 )
	printf("%7.4f ", ot_ot[0][k]); 
    break;
  case 2:
    printf("%1d", (int)(indata_tch[0] * 9.999)); 
    printf(":"); 
    for (k = in+2; k <= all+1; k++) {
      printf("%1d", (int)(fabs(ot_ot[0][k])*9.999)); 
      if ( k == in+2 )
	printf(" "); 
    }
    break;
  case 3:
    printf("%1d", (int)(indata_tch[0] * 9.999)); 
    printf(":"); 
    for (k = in+2; k <= all+1; k++) {
      printf("%1d", (int)(fabs(ot_ot[0][k])*9.999)); 
      if ( k == in+2 ) {
	printf(" "); 
	break;
      }
    }
    break;
  }
  if (wr < 3)
    putchar('\n');
}

int neuro_weight_write(int loop, int wr, int pa)
{
  int k, l, xp, yp;

  if (wr == 0) {
    printf("th+   th-   in1+  in1-  in2+  in2-  ...\n"); 
    for (k = in+2; k <= all+1; k++) {
      for (l = 0; l <= all+1; l++)
	printf("%5.2f ", w_ot_ot[0][k][l]); 
      putchar('\n');
    }
  }

  locate(0, 29); 

  printf("err:%3d count:%d\n", count, loop); 
  fflush(stdout);
  yp = (int)(250 - 200 * err /pa / ot);
  xp = loop + 50; 
  if (loop > 10000 || err < 0.1) {
    locate(0,0);
    printf("   %d   \n", loop);
    return 1;
  }
  line(xp, 250, xp, yp); 
  flush();
  err = 0;
  count = 0;
  return 0;
}
