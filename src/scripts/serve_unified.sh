#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

vllm serve CodeGoat24/UnifiedReward-qwen-7b \
    --host 127.0.0.1 \
    --trust-remote-code \
    --served-model-name UnifiedReward \
    --gpu-memory-utilization 0.8 \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 1 \
    --limit-mm-per-prompt image=2 \
    --port 8080
