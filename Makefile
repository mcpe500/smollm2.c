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
    src/attn_registry.c \
    src/resolve_model.c \
    src/studio.c \
    src/main.c

OBJ = $(SRC:.c=.o)

TARGET = smollm2

MODEL ?= models/smollm2-135m-f16.gguf
PORT  ?= 8082

.PHONY: all clean inspect studio studio-smoke studio-test \
        studio-train studio-bench studio-data studio-merge studio-clean

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJ) $(TARGET)

inspect: $(TARGET)
	./$(TARGET) --inspect

# Launch studio WebUI (default port 8082). Override: make studio PORT=9090
studio: $(TARGET)
	./$(TARGET) studio web --port $(PORT) --model $(MODEL)

studio-smoke: $(TARGET)
	python3 eval/studio_smoke.py

studio-test: $(TARGET)
	python3 eval/studio_smoke.py
	python3 eval/attn_matrix_test.py
	python3 eval/heavy_test.py
	python3 eval/parity.py

studio-bench:
	python3 eval/attn_bench.py

studio-data: $(TARGET)
	./$(TARGET) studio data-build --in $(IN) --out $(OUT) --model $(MODEL) $(ARGS)

studio-train: $(TARGET)
	./$(TARGET) studio train --data $(DATA) --model $(MODEL) \
	    --mode $(or $(MODE),lora) --rank $(or $(RANK),8) \
	    --out-dir $(or $(OUT_DIR),adapters) $(ARGS)

studio-merge: $(TARGET)
	./$(TARGET) studio merge --base $(BASE) --adapter $(ADAPTER) --out $(OUT)

studio-clean:
	rm -rf adapters/ packed/ /tmp/studio_*.bin
