SHELL := /bin/bash

IMAGE ?= gpu-ai-base
APP   ?= pytorch-cuda-ops

RUN = docker run --gpus all --rm -it \
	-v $(PWD):/workspace \
	-v $(HOME)/.cache/pip:/root/.cache/pip \
	-w /workspace --name $(APP)

.PHONY: build run lab test-gpu

build:
	docker build -t $(APP):dev .

run:
	$(RUN) $(IMAGE) bash

lab:
	$(RUN) -p 8888:8888 $(IMAGE) jupyter-lab --ip=0.0.0.0 --no-browser --allow-root

test-gpu:
	$(RUN) $(IMAGE) python -c "import torch; print("Torch GPU:", torch.cuda.is_available())

test-tf:
	docker run --gpus all --rm -it \
	-v $(PWD):/workspace -w /workspace \
	tensorflow/tensorflow:2.15.0-gpu \
	python -c "import tensorflow as tf; print('TF GPUs:', tf.config.list_physical_devices('GPU'))"
