SHELL := /bin/bash

APP ?= pytorch_cuda_ops
IMAGE ?= $(APP):dev

RUN = docker run --gpus all --rm -it \
	-v $(PWD):/workspace \
	-v $(HOME)/.cache/pip:/root/.cache/pip \
	-w /workspace --name $(APP)


.PHONY: build run lab test-gpu

build:
	docker build -t $(IMAGE) .

run:
	$(RUN) $(IMAGE) bash

lab:
	$(RUN) -p 8888:8888 $(IMAGE) jupyter-lab --ip=0.0.0.0 --no-browser --allow-root

test-gpu:
	$(RUN) $(IMAGE) python -c "import torch; print('Torch GPU:', torch.cuda.is_available())"
