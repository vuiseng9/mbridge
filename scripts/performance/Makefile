
# last verify on docker image:
# nvcr.io/nvidia/nemo:26.04
# goto /opt/Megatron-Bridge/
# git remote add vs https://github.com/vuiseng9/mbridge
# git fetch vs
# git checkout vs/r0.4.0-olmoe -b r0.4.0-olmoe

# do pull aligned megatron: git submodule update --init --recursive
# make login-hf
# how to debug? add dbg=1. e.g. make target dbg=1, 
#	launch dbg-attach cfg when "#### Waiting for debugger attach..." is printed.

NSTEP := 10
torchrun_intra = torchrun --standalone --nproc-per-node
ngpu ?= $(shell nvidia-smi -L | wc -l)
MASTER_ADDR ?= localhost
MASTER_PORT ?= 29700
master ?= $(MASTER_ADDR)
port ?= $(MASTER_PORT)
dbg ?= 0
xargs ?= 
CFG_VER ?= V2
benchcfg ?= 0
ENVVARS ?=
PROFILE_DIR ?= profiles
PROFILE_MAX_STEPS ?= 20
PROFILE_START ?= 16
PROFILE_STOP ?= 18
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
	DBG_ATTACH=$(dbg) BENCH_CFG=$(benchcfg) $(ENVVARS) $(torchrun_intra) $(ngpu) run_script.py \
		-m $(family) -mr $(recipe) -cv $(CFG_VER) -ng $(ngpu) -g $(gputype) -c $(compute) --max_steps $(NSTEP) 
		
# on 8xH100, step 10, 12.631 secs, 10376.77 tps/gpu, 534.096 model_tflops/gpu, 68.01 gb/rank0
101-h100-bf16-llama3_8b:
	$(MAKE) _bench_intranode family=llama recipe=llama3_8b gputype=h100 compute=bf16

_h100-bf16-olmoe:
	$(MAKE) _bench_intranode family=olmoe recipe=olmoe_1b_7b CFG_VER=V1 gputype=h100 compute=bf16 NSTEP=30

# closest to paper setup, mbs=4, fsdp only
b0-olmoe-megatron-fsdp:
	$(MAKE) _h100-bf16-olmoe benchcfg=0

b1-olmoe-ep8:
	$(MAKE) _h100-bf16-olmoe benchcfg=1

b2-olmoe-ep8-act-ckpt:
	$(MAKE) _h100-bf16-olmoe benchcfg=2

b3-olmoe-ep8-act-ckpt-offload:
	$(MAKE) _h100-bf16-olmoe benchcfg=3 ENVVARS="NVTE_CPU_OFFLOAD_V1=1"


_profile_intranode:
	mkdir -p $(PROFILE_DIR); \
	NVTE_CPU_OFFLOAD_V1=1 \
	DBG_ATTACH=$(dbg) BENCH_CFG=$(benchcfg) $(ENVVARS) nsys profile \
		-s $(PROFILE_SAMPLE) \
		-t $(PROFILE_TRACE) \
		-o $(PROFILE_DIR)/$(rptname) \
		--force-overwrite true \
		--capture-range=cudaProfilerApi \
		--capture-range-end=stop \
		$(PROFILE_GPU_METRICS_ARGS) \
		$(PROFILE_NSYS_ARGS) \
		$(torchrun_intra) $(ngpu) run_script.py \
			-m $(family) -mr $(recipe) -cv $(CFG_VER) -ng $(ngpu) -g $(gputype) -c $(compute) --max_steps $(PROFILE_MAX_STEPS) \
			profiling.use_nsys_profiler=true \
			profiling.profile_step_start=$(PROFILE_START) \
			profiling.profile_step_end=$(PROFILE_STOP) \
			"profiling.profile_ranks=$(PROFILE_RANKS)" \
			profiling.record_shapes=$(PROFILE_RECORD_SHAPES) \
			$(PROFILE_HYDRA_OVERRIDES);

prof-b1-olmoe-ep8:
	$(MAKE) _profile_intranode family=olmoe recipe=olmoe_1b_7b CFG_VER=V1 gputype=h100 compute=bf16 benchcfg=1 rptname=$@



test-dispatcher:
	$(torchrun_intra) $(ngpu) -m pytest \
		../../3rdparty/Megatron-LM/tests/unit_tests/transformer/moe/test_a2a_token_dispatcher.py::TestAlltoAllDispatcher  
# 		../../3rdparty/Megatron-LM/tests/unit_tests/transformer/moe/test_a2a_token_dispatcher.py::TestAlltoAllDispatcher::test_forward_backward  

# parallel_maps = [(1, 8), (8, 1), (4, 2), (1, 1)]
# parallel_maps = [(1, 2)] # (tp, ep)

# cd /opt/Megatron-Bridge/3rdparty/Megatron-LM/tests/test_utils/python_scripts/
# python download_unit_tests_dataset.py 
#  mv assets/* /opt/data/.

# test-ag-flex:
# 	python -m torch.distributed.run --nproc_per_node=$(ngpu) -m pytest ../../3rdparty/Megatron-LM/tests/unit_tests/transformer/moe/test_token_dispatcher.py::TestAllgatherDispatcher::test_forward_backward  

# allgather and flex (deepep, hybridep, need nvlink)
# /opt/Megatron-Bridge/3rdparty/Megatron-LM/tests/unit_tests/transformer/moe/test_token_dispatcher.py
# L365


# moe_permute_fusion - dispatch+permute to expert major
# moe_router_fusion - with aux loss

dl-test-assets:
	python /opt/Megatron-Bridge/3rdparty/Megatron-LM/tests/test_utils/python_scripts/download_unit_tests_dataset.py && mkdir -p /opt/data/ && mv assets/* /opt/data/.

symm-mem-dispatcher-test:
	$(torchrun_intra) $(ngpu) -m pytest \
		../../3rdparty/Megatron-LM/tests/unit_tests/transformer/moe/test_token_dispatcher.py::TestSymmMemDispatcher::test_forward_backward 