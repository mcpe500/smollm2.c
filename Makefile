# Makefile for smollm2.c — minimal C inference engine reading Ollama GGUF

CC      = gcc
CFLAGS  = -std=c99 -O2 -march=native -Wall -Wextra
LDFLAGS = -lm -lncurses

SRC = \
    src/gguf.c \
    src/tokenizer.c \
    src/forward.c \
    src/sampling.c \
    src/tui.c \
    src/web.c \
    src/data.c \
    src/gguf_write.c \
    src/backward.c \
    src/hw_probe.c \
    src/train.c \
    src/studio.c \
    src/main.c

OBJ = $(SRC:.c=.o)

TARGET = smollm2

.PHONY: all clean inspect studio-smoke studio-test

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJ) $(TARGET)

inspect: $(TARGET)
	./$(TARGET) --inspect

studio-smoke: $(TARGET)
	python3 eval/studio_smoke.py

studio-test: $(TARGET)
	python3 eval/studio_smoke.py
	python3 eval/attn_matrix_test.py
	python3 eval/heavy_test.py
	python3 eval/parity.py
