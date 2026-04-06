
# last verify on docker image:
# nvcr.io/nvidia/nemo:26.02.01

NSTEP := 10
torchrun_intra = torchrun --standalone --nproc-per-node
ngpu ?= $(shell nvidia-smi -L | wc -l)
MASTER_ADDR ?= localhost
MASTER_PORT ?= 29700
master ?= $(MASTER_ADDR)
port ?= $(MASTER_PORT)
dbg ?= 0

dep-install:
	pip install os

login-hf:
	if ! git config --global --get credential.helper | grep --color=auto -q store; then \
		echo "Warning: git config 'credential.helper' is not set to 'store'."; \
		echo "Executing ... git config --global credential.helper store"; \
		git config --global credential.helper store; \
	fi
	hf auth login

_bench_intranode:
	DBG_ATTACH=$(dbg) $(torchrun_intra) $(ngpu) run_script.py \
		-m $(family) -mr $(model) -ng $(ngpu) -g $(gputype) -c $(recipe) --max_steps $(NSTEP)

# Step Time: 5.90s , GPU utilization: 1142.7 MODEL_TFLOP/s/GPU , Peak Mem: 136.4GB
B16-b200-bf16-llama3_8b:
	$(MAKE) _bench_intranode family=llama model=llama3_8b gputype=b200 recipe=bf16

# Step Time: 4.23s , GPU utilization: 1595.5 MODEL_TFLOP/s/GPU , Peak Mem: 113.0GB
B08-b200-fp8-cs-llama3_8b:
	$(MAKE) _bench_intranode family=llama model=llama3_8b gputype=b200 recipe=fp8_cs

# Step Time: 4.34s , GPU utilization: 1554.5 MODEL_TFLOP/s/GPU , Peak Mem: 124.4GB
B18-b200-mxfp8-llama3_8b:
	$(MAKE) _bench_intranode family=llama model=llama3_8b gputype=b200 recipe=fp8_mx

# Step Time: 3.44s , GPU utilization: 1961.8 MODEL_TFLOP/s/GPU , Peak Mem: 171.6GB
B04-b200-nvfp4-llama3_8b:
	$(MAKE) _bench_intranode family=llama model=llama3_8b gputype=b200 recipe=nvfp4

# # Qwen3 30B A3B  8xB200, MXFP8
# 2-b200-mxfp8-qwen3-30b-a3b:
# 	$(MAKE) _bench_intranode family=qwen model=qwen3_30b_a3b gputype=b200 recipe=fp8_mx

# # Qwen3 30B A3B  8xB200, NVFP4
# # NOTE: QWEN3_30B_A3B_PRETRAIN_CONFIG_B200_NVFP4_V1 does not exist upstream yet
# b200-nvfp4-qwen3-30b-a3b:
# 	$(MAKE) _bench_intranode family=qwen model=qwen3_30b_a3b gputype=b200 recipe=nvfp4
# 		-m qwen -mr qwen3_30b_a3b -ng 8 -g b200 -c nvfp4 --max_steps $(NSTEP)

# already precalculated.
# Step Time : 24.14s GPU utilization: 279.5MODEL_TFLOP/s/GPU

# TPS = (GBS × L) / train_step_timing
# TPS/gpu = TPS/n_gpus
# L: sequence length; GBS: global batch size; n_gpus: number of gpus used
# e.g. 
#  [2026-04-05 21:39:57] iteration        8/      25 | consumed samples:         1024 
# | elapsed time per iteration (ms): 4225.9 | learning rate: 2.400000E-04 | global batch size:   128 
# | lm loss: 7.466920E-02 | loss scale: 1.0 | grad norm: 0.890 | number of skipped iterations:   0 | number of nan iterations:   0 |

# # TPS/gpu = (128*8192)/4.2259/8 = 31016 tps/gpu

# B200 4500 FP8-TFLOPS	9000 FP4-TFLOPS


