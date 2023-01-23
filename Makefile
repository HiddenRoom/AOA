CC=gcc
CFLAGS=-O0 -g3 -ggdb3 -Wall -Wextra -Wpedantic
LINKFLAGS=-lm

srcs: src/matrix.c src/neuralNetwork.c src/include/matrix.h src/include/neuralNetwork.h
	$(CC) $(CFLAGS) src/matrix.c -c -o matrix.o $(LINKFLAGS)
	$(CC) $(CFLAGS) src/neuralNetwork.c -c -o neuralNetwork.o $(LINKFLAGS)

test: srcs src/test.c
	$(CC) $(CFLAGS) matrix.o neuralNetwork.o src/test.c -o test.out $(LINKFLAGS)

clean:
	rm *.o
	rm *.out
