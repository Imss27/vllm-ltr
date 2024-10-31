#!/bin/bash

# Run benchmark with mlfq-quant0.03-thres10 schedule
echo "=========================================================="
echo "Running benchmark with schedule type: mlfq-quant0.03-thres10"
CUDA_VISIBLE_DEVICES=0 python benchmark_throughput_original.py --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl --model meta-llama/Meta-Llama-3-8B-Instruct --num-prompts 2000 --schedule-type mlfq-quant0.03-thres10 --enable-chunked-prefill --enforce-eager --swap-space 200 --dir BURST


# Run benchmark with PO schedule
echo "=========================================================="
echo "Running benchmark with schedule type: PO"
CUDA_VISIBLE_DEVICES=0 python benchmark_throughput_original.py --dataset PO-gen-llama3-8b-sharegpt-test-t1-s0-8192.jsonl --model meta-llama/Meta-Llama-3-8B-Instruct --num-prompts 2000 --schedule-type PO  --enable-chunked-prefill --enforce-eager --swap-space 100 --dir BURST


# Run benchmark with opt-xxx schedule
echo "=========================================================="
echo "Running benchmark with schedule type: opt-xxx"
CUDA_VISIBLE_DEVICES=0 python benchmark_throughput_original.py --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl --model meta-llama/Meta-Llama-3-8B-Instruct --num-prompts 2000 --schedule-type opt-xxx  --enable-chunked-prefill --enforce-eager --swap-space 16 --prefill-predictor-model-config MODEL/results/opt-125m-llama3-8b-sharegpt-score-trainbucket10-b32/usage_config.jsonjson --dir BURST


# Run benchmark with fcfs schedule
echo "=========================================================="
echo "Running benchmark with schedule type: fcfs"
CUDA_VISIBLE_DEVICES=0 python benchmark_throughput_original.py --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl --model meta-llama/Meta-Llama-3-8B-Instruct --num-prompts 2000 --schedule-type fcfs  --enable-chunked-prefill --enforce-eager --swap-space 16 --dir BURST


# Run benchmark with tpt-class10-xxx schedule
echo "=========================================================="
echo "Running benchmark with schedule type: tpt-class10-xxx"
CUDA_VISIBLE_DEVICES=0 python benchmark_throughput_original.py --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl --model meta-llama/Meta-Llama-3-8B-Instruct --num-prompts 2000 --schedule-type tpt-class10-xxx  --enable-chunked-prefill --enforce-eager --swap-space 100 --prefill-predictor-model-config MODEL/results/opt-125m-llama3-8b-sharegpt-class-trainbucket820-b32/usage_config.json --dir BURST

