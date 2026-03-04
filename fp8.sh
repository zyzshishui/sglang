set -e

# ==================== 1P2D Disaggregated Topology ====================
# Prefill: 1 node  (TP8, EP1, no DPA)  — 8 GPUs
# Decode:  2 nodes (TP8/EP8/DP8, DPA=True) — 16 GPUs, each node is 1 worker
# Aligned with InferenceX dsr1-fp8-mi355x-sglang-disagg "Middle of curve"
# Concurrency: 1536 (peak throughput ~2319 tok/s/gpu)
PREFILL_NODE="mia1-p02-g23"
DECODE_NODES=(mia1-p02-g46 mia1-p02-g10)

PREFILL_IP="${PREFILL_IP:-10.24.112.167}"
DECODE_IPS=("${DECODE_IP_0:-10.24.112.172}" "${DECODE_IP_1:-10.24.112.141}")

# ==================== model & environment ====================
MODEL_PATH="${MODEL_PATH:-/workspace/DeepSeek-R1-0528}"
DOCKER_IMAGE="${DOCKER_IMAGE:-lmsysorg/sglang:v0.5.9-rocm720-mi35x}"
WORKSPACE_HOST="${WORKSPACE_HOST:-/it-share-2/data/yuzhzhou}"
HF_CACHE="${HF_CACHE:-/data/yuzhzhou/cache/huggingface}"
TORCH_CACHE="${TORCH_CACHE:-/data/yuzhzhou/cache/torch}"
PIP_CACHE="${PIP_CACHE:-/data/yuzhzhou/cache/pip}"

# ==================== ports ====================
PREFILL_PORT=8000
DECODE_PORT=8000
ROUTER_PORT=30000
API_URL="http://localhost:${ROUTER_PORT}"

# IB devices for MoRI (mia1-* cluster)
IBDEVICES="${IBDEVICES:-rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7}"
DISAGG_DECODE_TP="${DISAGG_DECODE_TP:-8}"
DISAGG_DECODE_DP="${DISAGG_DECODE_DP:-8}"

# ==================== helpers ====================

run_on_head() {
  ssh "${PREFILL_NODE}" "docker exec sglang_r1_prefill bash -c '$*'"
}

docker_base_cmd() {
  local container_name=$1
  echo "docker run -itd --network=host --privileged --device=/dev/kfd --device=/dev/dri \
  --ipc=host --shm-size 128G --group-add video --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -v ${WORKSPACE_HOST}:/workspace \
  -w /workspace \
  -v ${HF_CACHE}:/root/.cache/huggingface \
  -v ${TORCH_CACHE}:/root/.cache/torch \
  -v ${PIP_CACHE}:/root/.cache/pip \
  -e HF_HOME=/root/.cache/huggingface \
  -e TRANSFORMERS_CACHE=/root/.cache/huggingface \
  -e HF_DATASETS_CACHE=/root/.cache/huggingface/datasets \
  -e WANDB_KEY='cd411df8b73eb3f5c1ae1220cc1ec4e3c9d1f86e' \
  -e WANDB_API_KEY='cd411df8b73eb3f5c1ae1220cc1ec4e3c9d1f86e' \
  -e NCCL_IB_HCA=${IBDEVICES} \
  -e DISAGG_DECODE_TP=${DISAGG_DECODE_TP} \
  -e DISAGG_DECODE_DP=${DISAGG_DECODE_DP} \
  -e GLOO_SOCKET_IFNAME=eno0 \
  -e NCCL_SOCKET_IFNAME=eno0 \
  -e PYTHONPATH=/workspace/sglang/python \
  --name ${container_name} \
  $DOCKER_IMAGE"
}

prefill_decode_shape_flags_cmd() {
  cat <<'EOF'
  DISAGG_DECODE_SHAPE_FLAGS=""
  if python3 -m sglang.launch_server --help 2>/dev/null | grep -q -- '--disaggregation-decode-tp'; then
    DISAGG_DECODE_SHAPE_FLAGS="--disaggregation-decode-tp ${DISAGG_DECODE_TP} --disaggregation-decode-dp ${DISAGG_DECODE_DP}"
  else
    echo "[WARN] launch_server does not support --disaggregation-decode-tp/dp; skipping these flags."
  fi
EOF
}

# Prefill: TP8, no EP, no DPA
# models.yaml DeepSeek-R1-0528 no_dp prefill config:
#   max_running_requests=128, chunked_prefill_size=262144, cuda_graph_bs=1-128
#   disable_radix_cache=true, mem_fraction_static=0.8
# Because P(no DPA) != D(DPA), need --disaggregation-decode-tp/dp
run_docker_prefill() {
  echo "$(docker_base_cmd sglang_r1_prefill) \
  bash -c ' \
  export SGLANG_USE_AITER=1; \
  export SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=16384; \
  export MORI_MAX_DISPATCH_TOKENS_PREFILL=16384; \
  export MORI_MAX_DISPATCH_TOKENS_DECODE=256; \
  export SGLANG_MORI_DISPATCH_INTER_KERNEL_SWITCH_THRESHOLD=512; \
  export MORI_SHMEM_MODE=ISOLATION; \
  export SGLANG_MORI_FP8_DISP=True; \
  export MORI_EP_LAUNCH_CONFIG_MODE=AUTO; \
  export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200; \
  export MORI_RDMA_TC=104; \
  $(prefill_decode_shape_flags_cmd); \
  python3 -m sglang.launch_server \
    --model-path $MODEL_PATH \
    --disaggregation-mode prefill \
    --disaggregation-transfer-backend mori \
    --disaggregation-ib-device ${IBDEVICES} \
    --tp 8 \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port ${PREFILL_PORT} \
    --mem-fraction-static 0.80 \
    --max-running-requests 128 \
    --chunked-prefill-size 262144 \
    --cuda-graph-bs $(echo {1..128}) \
    --disable-radix-cache \
    --attention-backend aiter \
    --kv-cache-dtype fp8_e4m3 \
    --ep-dispatch-algorithm fake \
    --load-balance-method round_robin \
    \${DISAGG_DECODE_SHAPE_FLAGS} \
    --decode-log-interval 1000 \
    --log-level warning \
    --watchdog-timeout 3600 \
    --log-level-http warning'"
}

# Decode: TP8/EP8/DP8, DPA=True
# models.yaml DeepSeek-R1-0528 dp decode config:
#   max_running_requests = cuda_graph_bs_dp_end(160) * tp(8) = 1280
#   chunked_prefill_size = MORI_MAX_DISPATCH_TOKENS_DECODE(256) * tp(8) = 2048
#   cuda_graph_bs = 1-160
# dp_flags: --moe-a2a-backend mori --deepep-mode normal
#           --enable-dp-attention --moe-dense-tp-size 1 --enable-dp-lm-head
run_docker_decode() {
  local rank=$1
  echo "$(docker_base_cmd sglang_r1_decode${rank}) \
  bash -c ' \
  SGLANG_USE_AITER=1 \
  SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256 \
  MORI_MAX_DISPATCH_TOKENS_PREFILL=16384 \
  MORI_MAX_DISPATCH_TOKENS_DECODE=256 \
  SGLANG_MORI_DISPATCH_INTER_KERNEL_SWITCH_THRESHOLD=512 \
  MORI_SHMEM_MODE=ISOLATION \
  SGLANG_MORI_FP8_DISP=True \
  MORI_EP_LAUNCH_CONFIG_MODE=AUTO \
  SGLANG_DISAGGREGATION_WAITING_TIMEOUT=1200 \
  MORI_RDMA_TC=104 \
  python3 -m sglang.launch_server \
    --model-path $MODEL_PATH \
    --disaggregation-mode decode \
    --disaggregation-transfer-backend mori \
    --disaggregation-ib-device ${IBDEVICES} \
    --tp 8 \
    --ep 8 \
    --dp-size 8 \
    --deepep-mode normal \
    --enable-dp-attention \
    --moe-dense-tp-size 1 \
    --enable-dp-lm-head \
    --moe-a2a-backend mori \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port ${DECODE_PORT} \
    --mem-fraction-static 0.85 \
    --max-running-requests 1280 \
    --chunked-prefill-size 2048 \
    --cuda-graph-bs $(echo {1..160}) \
    --attention-backend aiter \
    --kv-cache-dtype fp8_e4m3 \
    --ep-dispatch-algorithm fake \
    --load-balance-method round_robin \
    --decode-log-interval 1000 \
    --log-level warning \
    --watchdog-timeout 3600 \
    --log-level-http warning'"
}

run_docker_router() {
  local decode_args=""
  for ip in "${DECODE_IPS[@]}"; do
    decode_args+=" --decode http://${ip}:${DECODE_PORT}"
  done

  echo "$(docker_base_cmd sglang_r1_router) \
  python3 -m sglang_router.launch_router \
    --pd-disaggregation \
    --prefill http://${PREFILL_IP}:${PREFILL_PORT} \
    ${decode_args} \
    --host 0.0.0.0 \
    --port ${ROUTER_PORT} \
    --policy random \
    --prefill-policy random \
    --decode-policy random"
}

# ==================== start ====================
do_start() {
  echo "=========================================="
  echo "Starting DeepSeek R1 0528 FP8 — 1P2D Disaggregated"
  echo "  Prefill: ${PREFILL_NODE} (TP8, EP1, no DPA)"
  echo "  Decode:  ${DECODE_NODES[*]} (TP8/EP8/DP8, DPA=True)"
  echo "  Model:   $MODEL_PATH"
  echo "  Image:   $DOCKER_IMAGE"
  echo "  Backend: mori"
  echo "=========================================="

  # --- Launch prefill server ---
  echo "[INFO] Starting prefill on ${PREFILL_NODE}..."
  ssh "$PREFILL_NODE" "
    if docker ps -a --format '{{.Names}}' | grep -qx sglang_r1_prefill; then
      docker rm -f sglang_r1_prefill
    fi
    $(run_docker_prefill)
  " &

  sleep 2

  # --- Launch decode servers ---
  for i in "${!DECODE_NODES[@]}"; do
    node="${DECODE_NODES[$i]}"
    echo "[INFO] Starting decode $i on ${node}..."
    ssh "$node" "
      if docker ps -a --format '{{.Names}}' | grep -qx sglang_r1_decode${i}; then
        docker rm -f sglang_r1_decode${i}
      fi
      $(run_docker_decode $i)
    " &
    sleep 2
  done

  wait
  echo "[INFO] All server containers launched."

  # --- Wait for prefill server ---
  echo "[INFO] Waiting for prefill server to be ready..."
  for i in $(seq 1 180); do
    if ssh "${PREFILL_NODE}" "curl -sf http://localhost:${PREFILL_PORT}/health" > /dev/null 2>&1; then
      echo "[INFO] Prefill server is ready!"
      break
    fi
    if [ "$i" -eq 180 ]; then
      echo "[ERROR] Prefill server did not become ready within 15 minutes."
            exit 1
    fi
    sleep 5
  done

  # --- Wait for decode servers ---
  for i in "${!DECODE_NODES[@]}"; do
    echo "[INFO] Waiting for decode server $i (${DECODE_NODES[$i]}) to be ready..."
    for j in $(seq 1 180); do
      if ssh "${DECODE_NODES[$i]}" "curl -sf http://localhost:${DECODE_PORT}/health" > /dev/null 2>&1; then
        echo "[INFO] Decode server $i is ready!"
        break
      fi
      if [ "$j" -eq 180 ]; then
        echo "[ERROR] Decode server $i did not become ready within 15 minutes."
        exit 1
      fi
      sleep 5
    done
  done

  # --- Launch router ---
  echo "[INFO] Starting router on ${PREFILL_NODE}..."
  ssh "$PREFILL_NODE" "
    if docker ps -a --format '{{.Names}}' | grep -qx sglang_r1_router; then
      docker rm -f sglang_r1_router
    fi
    $(run_docker_router)
  "

  echo "[INFO] Waiting for router to be ready..."
  for i in $(seq 1 60); do
    if ssh "${PREFILL_NODE}" "curl -sf http://localhost:${ROUTER_PORT}/health" > /dev/null 2>&1; then
      echo "[INFO] Router is ready!"
      echo "[INFO] API endpoint: http://${PREFILL_NODE}:${ROUTER_PORT}"
      echo "[INFO] You can now run: bash fp8.sh submit"
      return 0
    fi
    if [ "$i" -eq 60 ]; then
      echo "[ERROR] Router did not become ready within 5 minutes."
      exit 1
    fi
    sleep 5
  done
}

# ==================== submit ====================
do_submit() {
  echo "[INFO] Checking router health on ${PREFILL_NODE}..."
  if ! ssh "${PREFILL_NODE}" "curl -sf http://localhost:${ROUTER_PORT}/health" > /dev/null 2>&1; then
    echo "[ERROR] Router is not running. Run 'bash fp8.sh start' first."
    exit 1
  fi
  echo "[INFO] Router is healthy."

  local conc="${MAX_CONCURRENCY:-1536}"
  local num_prompts=$((conc * 10))
  local warmup=$((conc * 2))

  echo "=========================================="
  echo "[INFO] Running benchmark (ISL=1024, OSL=1024, conc=${conc})"
  echo "=========================================="

  ssh "${PREFILL_NODE}" "docker exec sglang_r1_prefill python3 -m sglang.bench_serving \
    --backend sglang \
    --base-url http://localhost:${ROUTER_PORT} \
    --dataset-name random \
    --random-input-len 1024 \
    --random-output-len 1024 \
    --random-range-ratio 0.8 \
    --max-concurrency ${conc} \
    --num-prompts ${num_prompts} \
    --warmup-requests ${warmup} \
    --port ${ROUTER_PORT}" 2>&1

  echo "=========================================="
  echo "[DONE] Benchmark complete!"
  echo "=========================================="
}

# ==================== stop ====================
do_stop() {
  echo "[INFO] Stopping all containers..."

  ssh "$PREFILL_NODE" "
    docker rm -f sglang_r1_router 2>/dev/null && echo '[INFO] router removed' || echo '[INFO] router not found'
  " &

  ssh "$PREFILL_NODE" "
    docker rm -f sglang_r1_prefill 2>/dev/null && echo '[INFO] prefill removed' || echo '[INFO] prefill not found'
  " &

  for i in "${!DECODE_NODES[@]}"; do
    node="${DECODE_NODES[$i]}"
    ssh "$node" "
      docker rm -f sglang_r1_decode${i} 2>/dev/null && echo '[INFO] decode${i} removed' || echo '[INFO] decode${i} not found'
    " &
  done

  wait
  echo "[DONE] All containers stopped and removed."
}

# ==================== status ====================
do_status() {
  echo "[INFO] Checking status..."
  echo ""

  status=$(ssh "$PREFILL_NODE" "docker ps --filter name=sglang_r1_prefill --format '{{.Status}}' 2>/dev/null" || echo "unreachable")
  [ -z "$status" ] && status="not running"
  echo "  Prefill  (${PREFILL_NODE}): $status"

  for i in "${!DECODE_NODES[@]}"; do
    status=$(ssh "${DECODE_NODES[$i]}" "docker ps --filter name=sglang_r1_decode${i} --format '{{.Status}}' 2>/dev/null" || echo "unreachable")
    [ -z "$status" ] && status="not running"
    echo "  Decode$i (${DECODE_NODES[$i]}): $status"
  done

  status=$(ssh "$PREFILL_NODE" "docker ps --filter name=sglang_r1_router --format '{{.Status}}' 2>/dev/null" || echo "unreachable")
  [ -z "$status" ] && status="not running"
  echo "  Router   (${PREFILL_NODE}): $status"

  echo ""
  if ssh "${PREFILL_NODE}" "curl -sf http://localhost:${ROUTER_PORT}/health" > /dev/null 2>&1; then
    echo "[INFO] Router API is healthy."
  else
    echo "[WARN] Router API is not responding."
  fi
}

# ==================== logs ====================
do_logs() {
  local target="${1:-prefill}"
  case "$target" in
    prefill)
      echo "[INFO] Tailing prefill logs from ${PREFILL_NODE}..."
      ssh "$PREFILL_NODE" "docker logs -f --tail 100 sglang_r1_prefill"
      ;;
    decode0|decode1)
      local idx="${target#decode}"
      echo "[INFO] Tailing decode${idx} logs from ${DECODE_NODES[$idx]}..."
      ssh "${DECODE_NODES[$idx]}" "docker logs -f --tail 100 sglang_r1_decode${idx}"
      ;;
    router)
      echo "[INFO] Tailing router logs from ${PREFILL_NODE}..."
      ssh "$PREFILL_NODE" "docker logs -f --tail 100 sglang_r1_router"
      ;;
    *)
      echo "Usage: bash fp8.sh logs {prefill|decode0|decode1|router}"
      exit 1
      ;;
  esac
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
    echo "Usage: bash fp8.sh {start|submit|stop|status|logs [target]}"
    echo ""
    echo "  start   - Launch 1P2D disaggregated serving, wait until ready"
    echo "  submit  - Run bench_serving benchmark (ISL=1024, OSL=1024)"
    echo "  stop    - Stop and remove all containers"
    echo "  status  - Check container and API health"
    echo "  logs    - Tail logs (prefill|decode0|decode1|router)"
    exit 1
    ;;
esac
