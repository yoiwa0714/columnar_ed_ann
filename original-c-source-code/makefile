OPT = -L/usr/X11R6/lib -g
CC = gcc
LIB = -lX11 -lm
NEURO = graphic.o neuro.o main.o output_calc.o param_input.o teach_calc.o weight_calc.o write.o teach_input.o

new : $(NEURO)
	$(CC) $(OPT) -o new $(NEURO) $(LIB)

