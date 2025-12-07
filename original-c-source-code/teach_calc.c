#include "main.h"

void neuro_teach_calc(double* indata_tch)
{
  double inival1, inival2, wkb;
  int l, k, n;

  for (l = 0; l <= ot-1; l++) {
    wkb = indata_tch[l] - ot_ot[l][in+2]; 
    err += fabs(wkb); 
    if ( fabs(wkb) > 0.5 ) count++;

    if ( wkb > 0 ) {
      del_ot[l][in+2][0] = wkb; 
      del_ot[l][in+2][1] = 0; 
    }
    else {
      del_ot[l][in+2][0] = 0; 
      del_ot[l][in+2][1] = -wkb; 
    }

    inival1 = del_ot[l][in+2][0]; 
    inival2 = del_ot[l][in+2][1]; 

    for (k = in+3; k <= all+1; k++) {
      del_ot[l][k][0] = inival1*u1; 
      del_ot[l][k][1] = inival2*u1; 
    }
  }
}

