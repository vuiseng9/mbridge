
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

# on 8xH100, step 10, 12.631 secs, 10376.77 tps/gpu, 534.096 model_tflops/gpu, 68.01 gb/rank0
101-h100-bf16-llama3_8b:
	$(MAKE) _bench_intranode family=llama model=llama3_8b gputype=h100 recipe=bf16

