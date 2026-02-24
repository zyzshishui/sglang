set -e

NODES=(mia1-p02-g23 mia1-p02-g46 mia1-p02-g05 mia1-p02-g45)
HEAD_NODE="${NODES[0]}"

MODEL_PATH="${MODEL_PATH:-/workspace/DeepSeek-R1-0528}"

DOCKER_IMAGE="${DOCKER_IMAGE:-lmsysorg/sglang:v0.5.8.post1-rocm700-mi35x}"
WORKSPACE_HOST="${WORKSPACE_HOST:-/it-share-2/data/yuzhzhou}"
HF_CACHE="${HF_CACHE:-/data/yuzhzhou/cache/huggingface}"
TORCH_CACHE="${TORCH_CACHE:-/data/yuzhzhou/cache/torch}"
PIP_CACHE="${PIP_CACHE:-/data/yuzhzhou/cache/pip}"

echo "=========================================="
echo "Launching DeepSeek R1 on 4 nodes (Docker)"
echo "Head node: $HEAD_NODE"
echo "Model: $MODEL_PATH"
echo "Image: $DOCKER_IMAGE"
echo "=========================================="

run_docker() {
  local rank=$1
  local node=$2
  echo "docker run -itd --network=host --privileged --device=/dev/kfd --device=/dev/dri \\
  --ipc=host --shm-size 16G --group-add video --cap-add=SYS_PTRACE \\
  --security-opt seccomp=unconfined \\
  -v ${WORKSPACE_HOST}:/workspace \\
  -w /workspace/sglang \\
  -v ${HF_CACHE}:/root/.cache/huggingface \\
  -v ${TORCH_CACHE}:/root/.cache/torch \\
  -v ${PIP_CACHE}:/root/.cache/pip \\
  -e HF_HOME=/root/.cache/huggingface \\
  -e TRANSFORMERS_CACHE=/root/.cache/huggingface \\
  -e HF_DATASETS_CACHE=/root/.cache/huggingface/datasets \\
  -e SGLANG_USE_AITER=1 \\
  -e RCCL_MSCCL_ENABLE=0 \\
  -e ROCM_QUICK_REDUCE_QUANTIZATION=INT4 \\
  -e GLOO_SOCKET_IFNAME=eno0 \\
  -e NCCL_SOCKET_IFNAME=eno0 \\
  --name sglang_r1_node${rank} \\
  $DOCKER_IMAGE \\
  python3 -m sglang.launch_server \\
    --model-path $MODEL_PATH \\
    --tp 8 \\
    --ep 4 \\
    --dist-init-addr 10.24.112.167:20000 \\
    --nnodes 4 \\
    --node-rank $rank \\
    --trust-remote-code \\
    --host 0.0.0.0 \\
    --port 30000 \\
    --mem-fraction-static 0.85 \\
    --chunked-prefill-size 196608 \\
    --num-continuous-decode-steps 4 \\
    --max-prefill-tokens 196608 \\
    --cuda-graph-max-bs 512 \\
    --attention-backend aiter"
}

for i in "${!NODES[@]}"; do
  node="${NODES[$i]}"
  rank=$i
  log_file="/tmp/sglang_node${rank}_$(date +%Y%m%d_%H%M%S).log"

  echo "[INFO] Starting node $rank ($node) -> log: $log_file"

  ssh "$node" "
    if docker ps -a --format '{{.Names}}' | grep -qx sglang_r1_node${rank}; then
      echo '[INFO] Removing old container sglang_r1_node${rank}'
      docker rm -f sglang_r1_node${rank}
    fi
    $(run_docker $rank $node)
  " 2>&1 | tee "$log_file" &

  sleep 2
done

wait
echo "[INFO] All nodes launched. Check logs in /tmp/sglang_node*.log"
echo "[INFO] API endpoint: http://${HEAD_NODE}:30000"
