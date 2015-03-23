
all: seq


OPTFLAGS  = -g
INCFLAGS = -I.
CFLAGS = $(OPTFLAGS) $(INCFLAGS)
LDFLAGS = $(OPTFLAGS)
LIBS =

CPP = g++

.c.o:
	$(CC) $(CFLAGS) -c $<

H_FILES = kde.h
SEQ_SRC = kde_seq.cc seq_main.cc file_io.cc wtime.cc

SEQ_OBJ = $(SEQ_SRC:%.c=%.o)

seq: seq_main
seq_main: $(SEQ_OBJ) $(H_FILES)
	$(CPP) $(LDFLAGS) -o seq_main $(SEQ_OBJ) $(LIBS)

clean:
	rm -rf *.o seq_main
