# Makefile for smollm2.c - Decode-first SmolLM2 inference engine

CC = gcc
CFLAGS = -std=c99 -Wall -Wextra -O2 -march=native
CFLAGS += -DNDEBUG -Iinclude

# Target: smollm2-cli (CLI tool)
TARGET = smollm2-cli

# Source directory
SRC = src/

# Core runtime sources
SRC_CORE = $(SRC)smollm2.c \
           $(SRC)sm2_model.c \
           $(SRC)sm2_tokenizer.c \
           $(SRC)sm2_rmsnorm.c \
           $(SRC)sm2_rope.c \
           $(SRC)sm2_mlp.c \
           $(SRC)sm2_sampling.c \
           $(SRC)sm2_matmul_ref.c \
           $(SRC)sm2_context.c

# Decode layer (smollm2dl)
SRC_DECODE = $(SRC)decode/sm2dl_decode.c \
             $(SRC)decode/sm2dl_flash_decode.c \
             $(SRC)decode/sm2dl_paged_attention.c \
             $(SRC)decode/sm2dl_kv_quant.c \
             $(SRC)decode/sm2dl_batch_decode.c \
             $(SRC)decode/sm2dl_speculative.c

# Attention kernels
SRC_ATTENTION = $(SRC)attention/sm2_attn_prefill.c \
                $(SRC)attention/sm2_attn_flash_prefill.c \
                $(SRC)attention/sm2_attn_paged.c

# KV cache
SRC_KV = $(SRC)kv/sm2_kv_pool.c \
         $(SRC)kv/sm2_kv_page.c \
         $(SRC)kv/sm2_kv_quant_q8.c \
         $(SRC)kv/sm2_kv_quant_q4.c \
         $(SRC)kv/sm2_kv_turbo2.c

# Quantization
SRC_QUANT = $(SRC)quant/sm2_q8.c \
            $(SRC)quant/sm2_q4.c \
            $(SRC)quant/sm2_q4k.c \
            $(SRC)quant/sm2_q5k.c

# Backend
SRC_BACKEND = $(SRC)backend/sm2_backend_ref.c

# Server daemon (optional, Phase 6)
SRC_SERVER = $(SRC)server/smollm2d.c \
             $(SRC)server/sm2_http.c \
             $(SRC)server/sm2_sse.c \
             $(SRC)server/sm2_scheduler.c \
             $(SRC)server/sm2_metrics.c

# DFlash (Phase 8b)
SRC_DFLASH = $(SRC)dflash/sm2_dflash.c \
             $(SRC)dflash/sm2_dflash_model.c \
             $(SRC)dflash/sm2_dflash_verify.c

# All sources for full build
ALL_SRC = $(SRC_CORE) $(SRC_DECODE) $(SRC_ATTENTION) $(SRC_KV) $(SRC_QUANT) $(SRC_BACKEND)

# Object files
OBJ_DIR = obj
OBJS = $(patsubst $(SRC)%.c,$(OBJ_DIR)/%.o,$(ALL_SRC))

# Libraries
LIBM = -lm

.PHONY: all clean dirs

all: dirs $(TARGET)

dirs:
	mkdir -p $(OBJ_DIR)/decode $(OBJ_DIR)/attention $(OBJ_DIR)/kv
	mkdir -p $(OBJ_DIR)/quant $(OBJ_DIR)/backend $(OBJ_DIR)/server
	mkdir -p $(OBJ_DIR)/dflash

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LIBM)
	@echo "Built: $@"

# Pattern rules
$(OBJ_DIR)/%.o: $(SRC)%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -MMD -MP -c $< -o $@

clean:
	rm -rf $(OBJ_DIR) *.o $(TARGET) smollm2d

# Check syntax without full build
check:
	@for f in $(ALL_SRC); do \
		echo "Checking $$f..."; \
		$(CC) $(CFLAGS) -fsyntax-only -Iinclude $$f 2>&1 || true; \
	done

# Generate help
help:
	@echo "smollm2.c build system"
	@echo ""
	@echo "Targets:"
	@echo "  all      - Build smollm2-cli (default)"
	@echo "  clean    - Remove build artifacts"
	@echo "  check    - Syntax check all sources"
	@echo "  help     - Show this help"
	@echo ""
	@echo "To build:"
	@echo "  make"
	@echo ""
	@echo "To build with debug:"
	@echo "  make DEBUG=1"
	@echo ""
	@echo "To build server daemon:"
	@echo "  make smollm2d"

-include $(OBJS:.o=.d)