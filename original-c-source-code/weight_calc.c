#include "main.h"

void neuro_weight_calc()
{
  int k, m, n;
  double del;

  for (n = 0; n < ot; n++) {
    for (k = in+2; k <= all+1; k++) {
      for (m = 0; m <= all+1; m++) {
	if ( w_ot_ot[n][k][m] != 0 ) {
	  del = alpha*ot_in[n][m]; 
	  del *= fabs(ot_ot[n][k]); 
	  del *= (1 - fabs(ot_ot[n][k])); 
	  if ( f[10] == 1 )
	    w_ot_ot[n][k][m] += del * ow[k] * (del_ot[n][k][0] - del_ot[n][k][1]); 
	  else
	    if ( ow[m] > 0 )
	      w_ot_ot[n][k][m] += del * del_ot[n][k][0] * ow[m] * ow[k];
	    else
	      w_ot_ot[n][k][m] += del * del_ot[n][k][1] * ow[m] * ow[k];
	}
      }
    }
  }
}

