set -e

NODES=(mia1-p02-g23 mia1-p02-g46 mia1-p02-g05 mia1-p02-g45)
HEAD_NODE="${NODES[0]}"
NNODES=${#NODES[@]}

MODEL_PATH="${MODEL_PATH:-/workspace/DeepSeek-R1-0528}"

DOCKER_IMAGE="${DOCKER_IMAGE:-lmsysorg/sglang:v0.5.8.post1-rocm700-mi35x}"
WORKSPACE_HOST="${WORKSPACE_HOST:-/it-share-2/data/yuzhzhou}"
HF_CACHE="${HF_CACHE:-/data/yuzhzhou/cache/huggingface}"
TORCH_CACHE="${TORCH_CACHE:-/data/yuzhzhou/cache/torch}"
PIP_CACHE="${PIP_CACHE:-/data/yuzhzhou/cache/pip}"
AITER_JIT_CACHE="${AITER_JIT_CACHE:-${WORKSPACE_HOST}/cache/aiter_jit}"

API_URL="http://localhost:30000"

# Run a command on the head node inside the node-0 container
run_on_head() {
  ssh "${HEAD_NODE}" "docker exec sglang_r1_node0 bash -c '$*'"
}

# ==================== helpers ====================

run_docker() {
  local rank=$1
  echo "docker run -itd --network=host --privileged --device=/dev/kfd --device=/dev/dri \
  --device=/dev/infiniband \
  --ipc=host --shm-size 16G --group-add video --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -v ${WORKSPACE_HOST}:/workspace \
  -w /workspace/sglang \
  -v ${HF_CACHE}:/root/.cache/huggingface \
  -v ${TORCH_CACHE}:/root/.cache/torch \
  -v ${PIP_CACHE}:/root/.cache/pip \
  -v ${AITER_JIT_CACHE}:/sgl-workspace/aiter/aiter/jit \
  -v /usr/lib/x86_64-linux-gnu/libionic.so.1.0.54.0-149.g3304be71:/usr/lib/x86_64-linux-gnu/libionic.so.1.0.54.0-149.g3304be71:ro \
  -v /usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so:/usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so:ro \
  -v /etc/libibverbs.d/ionic.driver:/etc/libibverbs.d/ionic.driver:ro \
  -v ${WORKSPACE_HOST}/rccl-net-plugin-rocm700:/opt/rocm/lib/rccl-net-plugin:ro \
  -v /boot/config-\$(uname -r):/boot/config-\$(uname -r):ro \
  -e LD_LIBRARY_PATH=/opt/rocm/lib/rccl-net-plugin:/opt/rocm/lib \
  -e HF_HOME=/root/.cache/huggingface \
  -e TRANSFORMERS_CACHE=/root/.cache/huggingface \
  -e HF_DATASETS_CACHE=/root/.cache/huggingface/datasets \
  -e SGLANG_USE_AITER=1 \
  -e RCCL_MSCCL_ENABLE=0 \
  -e ROCM_QUICK_REDUCE_QUANTIZATION=INT4 \
  -e HSA_FORCE_FINE_GRAIN_PCIE=1 \
  -e HSA_NO_SCRATCH_RECLAIM=1 \
  -e GLOO_SOCKET_IFNAME=eno0 \
  -e NCCL_NET_PLUGIN=anp \
  -e NCCL_SOCKET_IFNAME=eno0 \
  -e NCCL_NET_GDR_LEVEL=3 \
  -e NCCL_DMABUF_ENABLE=0 \
  -e NCCL_GDR_FLUSH_DISABLE=1 \
  -e NCCL_MAX_P2P_CHANNELS=56 \
  -e NET_OPTIONAL_RECV_COMPLETION=1 \
  -e NCCL_IB_USE_INLINE=1 \
  -e RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING=0 \
  -e NCCL_IB_GID_INDEX=1 \
  -e NCCL_IB_TC=104 \
  -e NCCL_IB_FIFO_TC=192 \
  -e NCCL_IGNORE_CPU_AFFINITY=1 \
  -e NCCL_IB_QPS_PER_CONNECTION=1 \
  -e UCX_NET_DEVICES=eno0 \
  -e NCCL_DEBUG=WARN \
  -e WANDB_KEY='cd411df8b73eb3f5c1ae1220cc1ec4e3c9d1f86e' \
  -e WANDB_API_KEY='cd411df8b73eb3f5c1ae1220cc1ec4e3c9d1f86e' \
  --name sglang_r1_node${rank} \
  $DOCKER_IMAGE \
  python3 -m sglang.launch_server \
    --model-path $MODEL_PATH \
    --tp 8 \
    --dp-size $NNODES \
    --dist-init-addr 10.24.112.167:20000 \
    --nnodes $NNODES \
    --node-rank $rank \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 30000 \
    --mem-fraction-static 0.8 \
    --disable-radix-cache \
    --chunked-prefill-size 196608 \
    --num-continuous-decode-steps 4 \
    --max-prefill-tokens 196608 \
    --enable-dp-attention \
    --cuda-graph-max-bs 256 \
    --attention-backend aiter"
}

# ==================== start ====================
do_start() {
  echo "=========================================="
  echo "Starting DeepSeek R1 on ${NNODES} nodes (TP8 EP8)"
  echo "Head node: $HEAD_NODE"
  echo "Model: $MODEL_PATH"
  echo "Image: $DOCKER_IMAGE"
  echo "=========================================="

  if ssh "${HEAD_NODE}" "[ ! -f ${AITER_JIT_CACHE}/.initialized ]"; then
    echo "[INFO] Initializing aiter JIT cache from container image..."
    ssh "${HEAD_NODE}" "
      mkdir -p ${AITER_JIT_CACHE}
      docker run --rm \
        -v ${AITER_JIT_CACHE}:/mnt/aiter_jit_cache \
        ${DOCKER_IMAGE} \
        bash -c 'cp -a /sgl-workspace/aiter/aiter/jit/. /mnt/aiter_jit_cache/ && touch /mnt/aiter_jit_cache/.initialized'
    "
    echo "[INFO] aiter JIT cache initialized at ${AITER_JIT_CACHE}"
  else
    echo "[INFO] aiter JIT cache already initialized, skipping."
  fi

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
      echo '[INFO] Creating new container sglang_r1_node${rank}'
      $(run_docker $rank)
    " 2>&1 | tee "$log_file" &

    sleep 2
  done

  wait
  echo "[INFO] All containers launched. Check logs in /tmp/sglang_node*.log"
  echo "[INFO] API endpoint: $API_URL"

  echo "[INFO] Waiting for server to be ready..."
  for i in $(seq 1 120); do
    if ssh "${HEAD_NODE}" "curl -sf ${API_URL}/health" > /dev/null 2>&1; then
      echo "[INFO] Server is ready!"
      echo "[INFO] You can now run: bash multinodes.sh submit"
      return 0
    fi
    if [ "$i" -eq 120 ]; then
      echo "[ERROR] Server did not become ready within 10 minutes. Check logs."
      exit 1
    fi
    sleep 5
  done
}

# ==================== submit ====================
do_submit() {
  echo "[INFO] Checking server health on ${HEAD_NODE}..."
  if ! ssh "${HEAD_NODE}" "curl -sf ${API_URL}/health" > /dev/null 2>&1; then
    echo "[ERROR] Server is not running. Run 'bash multinodes.sh start' first."
    exit 1
  fi
  echo "[INFO] Server is healthy."

  echo "=========================================="
  echo "[INFO] Running benchmark on ${HEAD_NODE}"
  echo "=========================================="

  ssh "${HEAD_NODE}" "docker exec sglang_r1_node0 python3 -m sglang.bench_serving \
    --backend sglang \
    --base-url ${API_URL} \
    --dataset-name random \
    --random-input-len 1024 \
    --random-output-len 1024 \
    --max-concurrency ${MAX_CONCURRENCY:-256} \
    --num-prompts ${NUM_PROMPTS:-640} \
    --warmup-requests ${WARMUP_REQUESTS:-128} \
    --port 30000" 2>&1

  echo "=========================================="
  echo "[DONE] Benchmark complete!"
  echo "=========================================="
}

# ==================== stop ====================
do_stop() {
  echo "[INFO] Stopping all sglang containers on ${NNODES} nodes..."

  for i in "${!NODES[@]}"; do
    node="${NODES[$i]}"
    rank=$i
    echo "[INFO] Stopping node $rank ($node)..."
    ssh "$node" "
      if docker ps -a --format '{{.Names}}' | grep -qx sglang_r1_node${rank}; then
        docker rm -f sglang_r1_node${rank}
        echo '[INFO] sglang_r1_node${rank} removed'
      else
        echo '[INFO] sglang_r1_node${rank} not found, skip'
      fi
    " &
  done

  wait
  echo "[DONE] All containers stopped and removed."
}

# ==================== status ====================
do_status() {
  echo "[INFO] Checking status on ${NNODES} nodes..."
  echo ""

  for i in "${!NODES[@]}"; do
    node="${NODES[$i]}"
    rank=$i
    status=$(ssh "$node" "docker ps --filter name=sglang_r1_node${rank} --format '{{.Status}}' 2>/dev/null" || echo "unreachable")
    if [ -z "$status" ]; then
      status="not running"
    fi
    echo "  Node $rank ($node): $status"
  done

  echo ""
  if ssh "${HEAD_NODE}" "curl -sf ${API_URL}/health" > /dev/null 2>&1; then
    echo "[INFO] API is healthy (checked from ${HEAD_NODE})."
  else
    echo "[WARN] API is not responding (checked from ${HEAD_NODE})."
  fi
}

# ==================== logs ====================
do_logs() {
  local target_rank="${1:-0}"
  local target_node="${NODES[$target_rank]}"
  echo "[INFO] Tailing logs from node $target_rank ($target_node)..."
  ssh "$target_node" "docker logs -f --tail 100 sglang_r1_node${target_rank}"
}

# ==================== main ====================
CMD="${1:-}"
shift || true

case "$CMD" in
  start)  do_start ;;
  submit) do_submit ;;
  stop)   do_stop ;;
  status) do_status ;;
  logs)   do_logs "$@" ;;
  *)
    echo "Usage: bash multinodes.sh {start|submit|stop|status|logs [rank]}"
    echo ""
    echo "  start   - Launch sglang on all ${NNODES} nodes, wait until ready"
    echo "  submit  - Run bench_serving benchmark against the running server"
    echo "  stop    - Stop and remove all containers"
    echo "  status  - Check container and API health"
    echo "  logs    - Tail container logs (default: node 0, or specify rank)"
    exit 1
    ;;
esac
