#include "main.h"
  
void neuro_init(int in0, int ot0, int hd0, int hd20)
{
  int	k, l, n;

  in = in0;
  ot = ot0;
  hd = hd0 + hd20;
  hd2 = hd20;
  all = in + 1 + hd;

  for (n = 0; n < ot; n++) {
    for (k = 0; k <= all+1; k++) ow[k] = ((k+1) % 2) * 2 - 1; 
    ow[in+2] = 1; 
    for (k = in+2; k <= all+1; k++) {
      for (l = 0; l <= all+1; l++) {
	if ( l < 2 ) w_ot_ot[n][k][l] = inival2 * rnd();
	if ( l > 1 ) w_ot_ot[n][k][l] = inival1 * rnd(); 
	if ( k > all+1-hd2 && l < in+2 && l >= 2 ) w_ot_ot[n][k][l] =0; 
	if ( f[6] == 1 && k != l && k > in+2 && l > in+1 ) w_ot_ot[n][k][l] = 0; 
	if ( f[6] == 1 && k > in+1 && l > in+1 && l < in+3 ) w_ot_ot[n][k][l] = 0; 
	if ( f[7] == 1 && l >= 2 && l < in+2 && k >= in+2 && k < in+3 ) w_ot_ot[n][k][l] = 0; 
	if ( k > all+1-hd2 && l >= in+3 ) w_ot_ot[n][k][l] = inival1 * rnd(); 
	if ( k == l )
	  if ( f[3] == 1 )
	    w_ot_ot[n][k][l] = 0; 
	  else
	    w_ot_ot[n][k][l] = inival1 * rnd(); 
	if ( f[11] == 0 && l < in+2 && (l % 2) == 1 )
	  w_ot_ot[n][k][l] = 0; 
	w_ot_ot[n][k][l] *= ow[l] * ow[k];
      }
    }
    ot_in[n][0] = beta;
    ot_in[n][1] = beta;
  }

  count = 0;
  err = 0;
}

