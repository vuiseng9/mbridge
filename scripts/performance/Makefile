
# last verify on docker image:
# nvcr.io/nvidia/nemo:26.04
# goto /opt/Megatron-Bridge/
# git remote add vs https://github.com/vuiseng9/mbridge
# git fetch vs
# git checkout vs/r0.4.0-perf -b r0.4.0-perf

NSTEP := 10
torchrun_intra = torchrun --standalone --nproc-per-node
ngpu ?= $(shell nvidia-smi -L | wc -l)
MASTER_ADDR ?= localhost
MASTER_PORT ?= 29700
master ?= $(MASTER_ADDR)
port ?= $(MASTER_PORT)
dbg ?= 0
xargs ?= 

PROFILE_DIR ?= profiles
PROFILE_MAX_STEPS ?= 10
PROFILE_START ?= 5
PROFILE_STOP ?= 7
PROFILE_RANKS ?= [0]
PROFILE_TRACE ?= cuda,nvtx,osrt
PROFILE_SAMPLE ?= none
PROFILE_RECORD_SHAPES ?= false
PROFILE_GPU_METRICS ?= 0
PROFILE_GPU_METRICS_DEVICES ?= all
PROFILE_GPU_METRICS_FREQ ?=
PROFILE_OUTPUT ?= $(PROFILE_DIR)/$(family)-$(model)-$(gputype)-$(recipe)
PROFILE_NSYS_ARGS ?=
PROFILE_HYDRA_OVERRIDES ?=
PROFILE_GPU_METRICS_ARGS = $(if $(filter 1,$(PROFILE_GPU_METRICS)),--gpu-metrics-devices=$(PROFILE_GPU_METRICS_DEVICES),) $(if $(PROFILE_GPU_METRICS_FREQ),--gpu-metrics-frequency=$(PROFILE_GPU_METRICS_FREQ),)


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

_bench_intranode_megatron_fsdp:
	DBG_ATTACH=$(dbg) $(torchrun_intra) $(ngpu) run_script.py \
		-m $(family) -mr $(model) -ng $(ngpu) -g $(gputype) -c $(recipe) --max_steps $(NSTEP) --use_megatron_fsdp 1 --cuda_graph_impl none 


_profile_bench_intranode:
	mkdir -p $(PROFILE_DIR); \
	DBG_ATTACH=$(dbg) nsys profile \
		-s $(PROFILE_SAMPLE) \
		-t $(PROFILE_TRACE) \
		-o $(PROFILE_OUTPUT) \
		--force-overwrite true \
		--capture-range=cudaProfilerApi \
		--capture-range-end=stop \
		$(PROFILE_GPU_METRICS_ARGS) \
		$(PROFILE_NSYS_ARGS) \
		$(torchrun_intra) $(ngpu) run_script.py \
			-m $(family) -mr $(model) -ng $(ngpu) -g $(gputype) -c $(recipe) --max_steps $(PROFILE_MAX_STEPS) \
			profiling.use_nsys_profiler=true \
			profiling.profile_step_start=$(PROFILE_START) \
			profiling.profile_step_end=$(PROFILE_STOP) \
			"profiling.profile_ranks=$(PROFILE_RANKS)" \
			profiling.record_shapes=$(PROFILE_RECORD_SHAPES) \
			$(PROFILE_HYDRA_OVERRIDES);


# on A100 - Step Time: 32.59s , GPU utilization: 207.0 MODEL_TFLOP/s/GPU , Peak Mem: 64.3GB
B16-h100-bf16-llama3_8b:
	$(MAKE) _bench_intranode family=llama model=llama3_8b gputype=h100 recipe=bf16

# on A100 - Step Time: 7.04s , GPU utilization: 119.8 MODEL_TFLOP/s/GPU , Peak Mem: 59.5GB (gb16)
# 			Step Time: 11.91s , GPU utilization: 70.8 MODEL_TFLOP/s/GPU , Peak Mem: 42.5GB (add -ol offload 16)
# INFO:utils.overrides:DP: 8; TP: 1; PP: 1; CP: 1; VP: 1
# INFO:megatron.core.distributed.fsdp.src.megatron_fsdp.param_and_grad_buffer:Number of FSDP Parameter Groups: 34
# original cp:2, sl:8k, gb:512, too long to show progress with mb=1
# -ol 16, half of total tx layers.
mfsdp-B16-h100-bf16-llama3_8b:
	DBG_ATTACH=$(dbg) $(torchrun_intra) $(ngpu) run_script.py \
		-m llama -mr llama3_8b -ng $(ngpu) -g h100 -c bf16 --max_steps $(NSTEP) \
		--use_megatron_fsdp 1 -cp 1 -gb 16 -ol 16
# --nccl_ub 1 error
	
profile-mfsdp-B16-h100-bf16-llama3_8b:
	mkdir -p profiles/$@; \
	DBG_ATTACH=$(dbg) nsys profile \
		-s $(PROFILE_SAMPLE) \
		-t $(PROFILE_TRACE) \
		-o profiles/$@ \
		--force-overwrite true \
		--capture-range=cudaProfilerApi \
		--capture-range-end=stop \
		$(PROFILE_GPU_METRICS_ARGS) \
		$(PROFILE_NSYS_ARGS) \
		$(torchrun_intra) $(ngpu) run_script.py \
			-m llama -mr llama3_8b -ng $(ngpu) -g h100 -c bf16 \
			--use_megatron_fsdp 1 -cp 1 -gb 16 -ol 16 \
			--max_steps $(PROFILE_MAX_STEPS) \
			profiling.use_nsys_profiler=true \
			profiling.profile_step_start=$(PROFILE_START) \
			profiling.profile_step_end=$(PROFILE_STOP) \
			"profiling.profile_ranks=$(PROFILE_RANKS)" \
			profiling.record_shapes=$(PROFILE_RECORD_SHAPES) \
			$(PROFILE_HYDRA_OVERRIDES);

# Step Time: 5.90s , GPU utilization: 1142.7 MODEL_TFLOP/s/GPU , Peak Mem: 136.4GB
B16-b200-bf16-llama3_8b:
	$(MAKE) _bench_intranode family=llama model=llama3_8b gputype=b200 recipe=bf16

# Step Time: 6.23s , GPU utilization: 1082.6 MODEL_TFLOP/s/GPU , Peak Mem: 99.0GB
B16-mfsdp-b200-bf16-llama3_8b:
	$(MAKE) _bench_intranode_megatron_fsdp family=llama model=llama3_8b gputype=b200 recipe=bf16

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


# for attr in dir(cfg.ddp):
#     if '__' not in attr:
#         print(f"{attr}={getattr(cfg.ddp, attr)}")

