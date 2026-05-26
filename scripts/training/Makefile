NSTEP := 10
torchrun_intra = torchrun --standalone --nproc-per-node
ngpu ?= $(shell nvidia-smi -L | wc -l)
MASTER_ADDR ?= localhost
MASTER_PORT ?= 29700
master ?= $(MASTER_ADDR)
port ?= $(MASTER_PORT)
dbg ?= 0
xargs ?= 

llama32_1b_pretrain:
	rm -rf nemo_experiments/
	DBG_ATTACH=$(dbg) $(torchrun_intra) $(ngpu) run_recipe.py \
		--recipe llama32_1b_pretrain_config --dataset llm-pretrain-mock 
		
olmoe_pretrain:
	rm -rf nemo_experiments/
	DBG_ATTACH=$(dbg) $(torchrun_intra) $(ngpu) run_recipe.py \
		--recipe olmoe_7b_pretrain_config --dataset llm-pretrain-mock

# notes: 
# 	* pull aligned megatron: git submodule update --init --recursive
#	* llama32_1b rquires hf login
#	* these are just smoke tests, not for performance studies. Above has been tested runnable on 8xh100.
# 	* there is no print out when it is running. 
#	* cfg.train.train_iters is set to 10, this affects mock dataset creation as well.
#	* how to debug? add dbg=1. e.g. make olmoe_pretrain dbg=1, 
#		launch dbg-attach cfg when "#### Waiting for debugger attach..." is printed.