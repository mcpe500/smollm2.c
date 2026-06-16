# Makefile for smollm2.c — minimal C inference engine reading Ollama GGUF

CC      = gcc
CFLAGS  = -std=c99 -O2 -march=native -Wall -Wextra
LDFLAGS = -lm

SRC = \
    src/gguf.c \
    src/main.c

OBJ = $(SRC:.c=.o)

TARGET = smollm2

.PHONY: all clean inspect

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJ) $(TARGET)

inspect: $(TARGET)
	./$(TARGET) --inspect
