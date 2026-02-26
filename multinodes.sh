set -e

NODES=(mia1-p02-g23 mia1-p02-g46) # mia1-p02-g05 mia1-p02-g45
HEAD_NODE="${NODES[0]}"
NNODES=${#NODES[@]}

MODEL_PATH="${MODEL_PATH:-/workspace/DeepSeek-R1-0528}"

DOCKER_IMAGE="${DOCKER_IMAGE:-lmsysorg/sglang:v0.5.8.post1-rocm700-mi35x}"
WORKSPACE_HOST="${WORKSPACE_HOST:-/it-share-2/data/yuzhzhou}"
HF_CACHE="${HF_CACHE:-/data/yuzhzhou/cache/huggingface}"
TORCH_CACHE="${TORCH_CACHE:-/data/yuzhzhou/cache/torch}"
PIP_CACHE="${PIP_CACHE:-/data/yuzhzhou/cache/pip}"
AITER_JIT_CACHE="${AITER_JIT_CACHE:-${WORKSPACE_HOST}/cache/aiter_jit}"

SERVER_PORT=30001
ROUTER_PORT=30000
API_URL="http://localhost:${ROUTER_PORT}"

run_on_head() {
  ssh "${HEAD_NODE}" "docker exec sglang_r1_server0 bash -c '$*'"
}

get_node_ip() {
  ssh "$1" "hostname -I | awk '{print \$1}'"
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
  -e SGLANG_USE_ROCM700A=1 \
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
  --name sglang_r1_server${rank} \
  $DOCKER_IMAGE \
  python3 -m sglang.launch_server \
    --model-path $MODEL_PATH \
    --tp 8 \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port ${SERVER_PORT} \
    --mem-fraction-static 0.8 \
    --disable-radix-cache \
    --chunked-prefill-size 196608 \
    --num-continuous-decode-steps 4 \
    --max-prefill-tokens 196608 \
    --cuda-graph-max-bs 256 \
    --attention-backend aiter "
}

run_router() {
  local worker_urls="$1"
  echo "docker run -itd --network=host \
  --name sglang_r1_router \
  $DOCKER_IMAGE \
  python3 -m sglang_router.launch_router \
    --worker-urls ${worker_urls} \
    --policy round_robin \
    --host 0.0.0.0 \
    --port ${ROUTER_PORT}"
}

# ==================== start ====================
do_start() {
  echo "=========================================="
  echo "Head node: $HEAD_NODE"
  echo "Model: $MODEL_PATH"
  echo "Image: $DOCKER_IMAGE"
  echo "Architecture: Router + ${NNODES} independent TP8 servers"
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

  echo "[INFO] Resolving node IP addresses..."
  WORKER_URLS=""
  for i in "${!NODES[@]}"; do
    node="${NODES[$i]}"
    ip=$(get_node_ip "$node")
    echo "  Server $i ($node): $ip"
    WORKER_URLS="${WORKER_URLS} http://${ip}:${SERVER_PORT}"
  done
  WORKER_URLS="${WORKER_URLS# }"

  for i in "${!NODES[@]}"; do
    node="${NODES[$i]}"
    log_file="/tmp/sglang_server${i}_$(date +%Y%m%d_%H%M%S).log"

    echo "[INFO] Starting server $i ($node) -> log: $log_file"

    ssh "$node" "
      if docker ps -a --format '{{.Names}}' | grep -qx sglang_r1_server${i}; then
        echo '[INFO] Removing old container sglang_r1_server${i}'
        docker rm -f sglang_r1_server${i}
      fi
      echo '[INFO] Creating new container sglang_r1_server${i}'
      $(run_docker $i)
    " 2>&1 | tee "$log_file" &

    sleep 2
  done

  wait
  echo "[INFO] All server containers launched."

  echo "[INFO] Waiting for servers to be ready..."
  for i in "${!NODES[@]}"; do
    node="${NODES[$i]}"
    echo "[INFO] Waiting for server $i ($node)..."
    for attempt in $(seq 1 240); do
      if ssh "$node" "curl -sf http://localhost:${SERVER_PORT}/health" > /dev/null 2>&1; then
        echo "[INFO] Server $i ($node) is ready!"
        break
      fi
      if [ "$attempt" -eq 240 ]; then
        echo "[ERROR] Server $i did not become ready within 20 minutes."
        exit 1
      fi
      sleep 5
    done
  done

  echo "[INFO] Starting router on ${HEAD_NODE} -> workers: ${WORKER_URLS}"
  ssh "${HEAD_NODE}" "
    if docker ps -a --format '{{.Names}}' | grep -qx sglang_r1_router; then
      echo '[INFO] Removing old router container'
      docker rm -f sglang_r1_router
    fi
    $(run_router "$WORKER_URLS")
  "

  echo "[INFO] Waiting for router to be ready..."
  for attempt in $(seq 1 60); do
    if ssh "${HEAD_NODE}" "curl -sf ${API_URL}/health" > /dev/null 2>&1; then
      echo "[INFO] Router is ready!"
      echo "[INFO] API endpoint: $API_URL"
      echo "[INFO] You can now run: bash multinodes.sh submit"
      return 0
    fi
    if [ "$attempt" -eq 60 ]; then
      echo "[ERROR] Router did not become ready within 5 minutes."
      exit 1
    fi
    sleep 5
  done
}

# ==================== submit ====================
do_submit() {
  echo "[INFO] Checking router health on ${HEAD_NODE}..."
  if ! ssh "${HEAD_NODE}" "curl -sf ${API_URL}/health" > /dev/null 2>&1; then
    echo "[ERROR] Router is not running. Run 'bash multinodes.sh start' first."
    exit 1
  fi
  echo "[INFO] Router is healthy."

  echo "=========================================="
  echo "[INFO] Running benchmark on ${HEAD_NODE}"
  echo "=========================================="

  ssh "${HEAD_NODE}" "docker exec sglang_r1_server0 python3 -m sglang.bench_serving \
    --backend sglang \
    --base-url ${API_URL} \
    --dataset-name random \
    --random-input-len 1024 \
    --random-output-len 1024 \
    --max-concurrency ${MAX_CONCURRENCY:-256} \
    --num-prompts ${NUM_PROMPTS:-640} \
    --warmup-requests ${WARMUP_REQUESTS:-128} \
    --port ${ROUTER_PORT}" 2>&1

  echo "=========================================="
  echo "[DONE] Benchmark complete!"
  echo "=========================================="
}

# ==================== stop ====================
do_stop() {
  echo "[INFO] Stopping all sglang containers..."

  echo "[INFO] Stopping router on ${HEAD_NODE}..."
  ssh "${HEAD_NODE}" "
    if docker ps -a --format '{{.Names}}' | grep -qx sglang_r1_router; then
      docker rm -f sglang_r1_router
      echo '[INFO] sglang_r1_router removed'
    else
      echo '[INFO] sglang_r1_router not found, skip'
    fi
  " &

  for i in "${!NODES[@]}"; do
    node="${NODES[$i]}"
    echo "[INFO] Stopping server $i ($node)..."
    ssh "$node" "
      if docker ps -a --format '{{.Names}}' | grep -qx sglang_r1_server${i}; then
        docker rm -f sglang_r1_server${i}
        echo '[INFO] sglang_r1_server${i} removed'
      else
        echo '[INFO] sglang_r1_server${i} not found, skip'
      fi
    " &
  done

  wait
  echo "[DONE] All containers stopped and removed."
}

# ==================== status ====================
do_status() {
  echo "[INFO] Checking status..."
  echo ""

  router_status=$(ssh "${HEAD_NODE}" "docker ps --filter name=sglang_r1_router --format '{{.Status}}' 2>/dev/null" || echo "unreachable")
  if [ -z "$router_status" ]; then
    router_status="not running"
  fi
  echo "  Router (${HEAD_NODE}): $router_status"

  for i in "${!NODES[@]}"; do
    node="${NODES[$i]}"
    status=$(ssh "$node" "docker ps --filter name=sglang_r1_server${i} --format '{{.Status}}' 2>/dev/null" || echo "unreachable")
    if [ -z "$status" ]; then
      status="not running"
    fi
    echo "  Server $i ($node): $status"
  done

  echo ""
  if ssh "${HEAD_NODE}" "curl -sf ${API_URL}/health" > /dev/null 2>&1; then
    echo "[INFO] Router API is healthy."
  else
    echo "[WARN] Router API is not responding."
  fi

  for i in "${!NODES[@]}"; do
    node="${NODES[$i]}"
    if ssh "$node" "curl -sf http://localhost:${SERVER_PORT}/health" > /dev/null 2>&1; then
      echo "[INFO] Server $i ($node) is healthy."
    else
      echo "[WARN] Server $i ($node) is not responding."
    fi
  done
}

# ==================== logs ====================
do_logs() {
  local target="${1:-router}"
  if [ "$target" = "router" ]; then
    echo "[INFO] Tailing logs from router (${HEAD_NODE})..."
    ssh "${HEAD_NODE}" "docker logs -f --tail 100 sglang_r1_router"
  else
    local target_node="${NODES[$target]}"
    echo "[INFO] Tailing logs from server $target ($target_node)..."
    ssh "$target_node" "docker logs -f --tail 100 sglang_r1_server${target}"
  fi
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
    echo "Usage: bash multinodes.sh {start|submit|stop|status|logs [router|0|1]}"
    echo ""
    echo "  start   - Launch ${NNODES} independent TP8 servers + router, wait until ready"
    echo "  submit  - Run bench_serving benchmark via the router"
    echo "  stop    - Stop and remove all containers (router + servers)"
    echo "  status  - Check container and API health"
    echo "  logs    - Tail container logs (default: router, or specify 0/1 for servers)"
    exit 1
    ;;
esac
