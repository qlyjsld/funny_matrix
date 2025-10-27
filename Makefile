CC = gcc
CFLAGS = -Wall -g -mavx2
all: main

.PHONY: main
main: main.c
	$(CC) $(CFLAGS) main.c -o a.out

.PHONY: clean
clean:
	rm a.out

