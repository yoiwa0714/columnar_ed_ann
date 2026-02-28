#include "main.h"

extern void inputi(char* s, int* i, int def);
extern void inputf(char* s, double* f, double def);

void neuro_param_input()
{
  inputi("t_loop ?(2) ", &t_loop, 2); 
  inputf("weight initial value ?(1.0) ", &inival1, 1.0); 
  inputf("threshold initial value ?(1.0) ", &inival2, 1.0); 
  inputi("multi layer flag ?(1) ", &f[7], 1); 
  inputi("weight decrement ?(0) ", &f[10], 0); 
  inputi("loop cut flag ?(1) ", &f[6], 1); 
  inputi("self loop cut flag ?(1) ", &f[3], 1); 
  inputi("- input flag ?(1) ", &f[11], 1); 
  inputf("u0 ?(0.4) ", &u0, 0.4); 
  inputf("u1 ?(1.0) ", &u1, 1.0); 
  inputf("alpha ?(0.8) ", &alpha, 0.8); 
  inputf("beta ?(0.8) ", &beta, 0.8); 
  inputf("res ?(0.0) ", &res, 0.0); 
}

