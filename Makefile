.SILENT:

CC = mpicc
CFLAGS = -O3 -Wall
OUT = mpisort

mpisort: mpisort.o
	$(CC) $(CFLAGS) -o mpisort mpisort.o -lm

mpisort.o: mpisort.c
	$(CC) $(CFLAGS) -c mpisort.c -o mpisort.o

clean:
	rm -f mpisort mpisort.o

run: $(OUT)
	for k in 0 1 2 3 4; do \
		for p in 1 2 4 8; do \
			mpirun -np $$p $(OUT) $$k; \
		done; \
		echo ""; \
	done;
	
