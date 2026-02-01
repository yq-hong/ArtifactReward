#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
    --host 127.0.0.1 \
    --trust-remote-code \
    --served-model-name Qwen2.5-VL-7B-Instruct \
    --gpu-memory-utilization 0.8 \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 1 \
    --limit-mm-per-prompt image=2 \
    --port 8090
