CC = gcc
CFLAGS = -Wall -O2 -march=native
all: main

.PHONY: main
main: main.c
	$(CC) $(CFLAGS) main.c -o a.out -lm

.PHONY: clean
clean:
	rm a.out

