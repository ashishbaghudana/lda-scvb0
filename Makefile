CC = gcc
CFLAGS = -Wall -fopenmp -lm -std=c11
DEBUG = -DDEBUG
LDLIBS =
OBJECTS = dataset.o util.o scvb0.o

all: lda

scvb0.o: scvb0.c scvb0.h
	$(CC) $(CFLAGS) -c scvb0.c

dataset.o: dataset.c dataset.h
	$(CC) $(CFLAGS) -c dataset.c

util.o: util.c util.h
	$(CC) $(CFLAGS) -c util.c

lda: $(OBJECTS)
	$(CC) -o lda $(CFLAGS) $(LDLIBS) $(OBJECTS)

debug:
	$(CC) $(CFLAGS) $(DEBUG) scvb0.c dataset.c util.c -o lda-debug

clean:
	rm -f *.o *~ lda
