NSTEP := 10
torchrun_intra = torchrun --standalone --nproc-per-node
ngpu ?= $(shell nvidia-smi -L | wc -l)
MASTER_ADDR ?= localhost
MASTER_PORT ?= 29700
master ?= $(MASTER_ADDR)
port ?= $(MASTER_PORT)
dbg ?= 0
xargs ?= 

# expect OOM on 80GBx8
llama_pretrain:
	rm -rf nemo_experiments/
	DBG_ATTACH=$(dbg) $(torchrun_intra) $(ngpu) run_recipe.py \
		--recipe llama32_1b_pretrain_config --dataset llm-pretrain-mock 
		
olmoe_pretrain:
	rm -rf nemo_experiments/
	DBG_ATTACH=$(dbg) $(torchrun_intra) $(ngpu) run_recipe.py \
		--recipe olmoe_7b_pretrain_config --dataset llm-pretrain-mock

# note: there is no print out when it is running. 