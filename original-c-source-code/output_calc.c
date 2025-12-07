#include "main.h"

void neuro_calc(double* indata_input, double* indata_tch)
{
  neuro_output_calc(indata_input);
  neuro_teach_calc(indata_tch);
  neuro_weight_calc();
}

void neuro_output_calc(double* indata_input)
{
  int k, t, m, n;
  double inival;

  for (n = 0; n < ot; n++) {
    for (k = 2; k <= in+1; k++)
      ot_in[n][k] = indata_input[(int)(k/2)-1];
    if (f[6])
      for (k = in+2; k <= all+1; k++)
	ot_in[n][k] = 0; 
    for (t = 1; t <= t_loop; t++) {
      for (k = in+2; k <= all+1; k++) {
	inival = 0; 
	for (m = 0; m <= all+1; m++)
	  inival += w_ot_ot[n][k][m]*ot_in[n][m]; 
	ot_ot[n][k] = sigmf(inival);
      }
      for (k = in+2; k <= all+1; k++)
	ot_in[n][k] = ot_ot[n][k]; 
    }
  }
}

