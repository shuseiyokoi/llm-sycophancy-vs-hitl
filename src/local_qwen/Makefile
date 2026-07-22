# Local open-weight models served with llama.cpp (OpenAI-compatible API on localhost)
#
# Prereqs:
#   - llama.cpp (mac: `brew install llama.cpp`; HPC: build with -DGGML_CUDA=ON)
#
# Usage:
#   make download        # fetch all GGUF models (~29GB total)
#   make serve           # serve Qwen2.5-7B-Instruct on :8080
#   make serve-qwen3     # serve Qwen3-8B on :8080
#   make serve-llama31   # serve Llama-3.1-8B-Instruct on :8080
#   make serve-llama32   # serve Llama-3.2-3B-Instruct on :8080
#   make serve-gemma2    # serve Gemma-2-9B-it on :8080
#   make serve-gemma3    # serve Gemma-3-12B-it on :8080
#
# Note: `python call_qwen.py` starts/stops these servers by itself,
# so the serve targets are only needed for manual testing.

LLAMA_SERVER ?= llama-server
PORT ?= 8080
CTX ?= 8192
NGL ?= 99

MODELS_DIR := models
QWEN25  := $(MODELS_DIR)/Qwen2.5-7B-Instruct-Q4_K_M.gguf
QWEN3   := $(MODELS_DIR)/Qwen3-8B-Q4_K_M.gguf
LLAMA31 := $(MODELS_DIR)/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
LLAMA32 := $(MODELS_DIR)/Llama-3.2-3B-Instruct-Q4_K_M.gguf
GEMMA2  := $(MODELS_DIR)/gemma-2-9b-it-Q4_K_M.gguf
GEMMA3  := $(MODELS_DIR)/google_gemma-3-12b-it-Q4_K_M.gguf

download:
	mkdir -p $(MODELS_DIR)
	curl -L -C - -o $(QWEN25) https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf
	curl -L -C - -o $(QWEN3) https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf
	curl -L -C - -o $(LLAMA31) https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
	curl -L -C - -o $(LLAMA32) https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf
	curl -L -C - -o $(GEMMA2) https://huggingface.co/bartowski/gemma-2-9b-it-GGUF/resolve/main/gemma-2-9b-it-Q4_K_M.gguf
	curl -L -C - -o $(GEMMA3) https://huggingface.co/bartowski/google_gemma-3-12b-it-GGUF/resolve/main/google_gemma-3-12b-it-Q4_K_M.gguf

serve:
	$(LLAMA_SERVER) -m $(QWEN25) --port $(PORT) -c $(CTX) -ngl $(NGL)

serve-qwen3:
	$(LLAMA_SERVER) -m $(QWEN3) --port $(PORT) -c $(CTX) -ngl $(NGL) --reasoning-budget 0

serve-llama31:
	$(LLAMA_SERVER) -m $(LLAMA31) --port $(PORT) -c $(CTX) -ngl $(NGL)

serve-llama32:
	$(LLAMA_SERVER) -m $(LLAMA32) --port $(PORT) -c $(CTX) -ngl $(NGL)

serve-gemma2:
	$(LLAMA_SERVER) -m $(GEMMA2) --port $(PORT) -c $(CTX) -ngl $(NGL)

serve-gemma3:
	$(LLAMA_SERVER) -m $(GEMMA3) --port $(PORT) -c $(CTX) -ngl $(NGL)
