#!/bin/bash
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"

GRN='\033[0;32m'; YLW='\033[1;33m'; RED='\033[0;31m'; DIM='\033[2m'; NC='\033[0m'
log()  { echo -e "${GRN}▸${NC} $1"; }
warn() { echo -e "${YLW}⚠${NC}  $1"; }
die()  { echo -e "${RED}✗${NC}  $1"; exit 1; }

echo ""
echo -e "  ${GRN}◈ OpenClaw Gateway${NC}"
echo -e "  ${DIM}─────────────────────────────${NC}"
echo ""

command -v docker >/dev/null 2>&1 || die "docker not found"

PORT="${PORT:-8080}"
OLLAMA_MODEL="${OLLAMA_MODEL:-nomic-embed-text}"
SKIP_PULL="${SKIP_PULL:-0}"

for arg in "$@"; do
  case $arg in
    --port=*)      PORT="${arg#*=}" ;;
    --no-pull)     SKIP_PULL=1 ;;
    --stop)
      log "Stopping all services..."
      docker compose -f "$ROOT/docker-compose.yml" down
      exit 0 ;;
    --help)
      echo "Usage: ./start.sh [options]"
      echo "  --port=N     Gateway port (default: 8080)"
      echo "  --no-pull    Skip Ollama model pull"
      echo "  --stop       Stop all services"
      exit 0 ;;
  esac
done

# Expose HOST_HOME cho docker-compose volume mount
export HOST_HOME="$HOME"
export PORT="$PORT"

# ── 1. Build + start ──────────────────────────────
log "Building gateway image..."
docker compose -f "$ROOT/docker-compose.yml" build --quiet gateway

log "Starting services..."
docker compose -f "$ROOT/docker-compose.yml" up -d

# ── 2. Wait for ChromaDB ──────────────────────────
log "Waiting for ChromaDB..."
for i in $(seq 1 30); do
  docker exec chromadb curl -sf http://localhost:8000/api/v1/heartbeat >/dev/null 2>&1 && break
  sleep 1
done
docker exec chromadb curl -sf http://localhost:8000/api/v1/heartbeat >/dev/null 2>&1 \
  || warn "ChromaDB not ready yet — will retry on first request"

# ── 3. Pull embed model ───────────────────────────
if [ "$SKIP_PULL" = "0" ]; then
  log "Waiting for Ollama..."
  for i in $(seq 1 30); do
    docker exec ollama curl -sf http://localhost:11434/api/tags >/dev/null 2>&1 && break
    sleep 1
  done

  if docker exec ollama curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
    MODELS=$(docker exec ollama ollama list 2>/dev/null | awk 'NR>1{print $1}')
    if echo "$MODELS" | grep -q "$OLLAMA_MODEL"; then
      log "Embed model '$OLLAMA_MODEL' already pulled"
    else
      log "Pulling embed model '$OLLAMA_MODEL'..."
      docker exec ollama ollama pull "$OLLAMA_MODEL"
    fi
  else
    warn "Ollama not responding — skipping model pull"
  fi
fi

# ── 4. Wait for gateway ───────────────────────────
log "Waiting for gateway..."
for i in $(seq 1 30); do
  curl -sf http://localhost:${PORT}/health >/dev/null 2>&1 && break
  sleep 1
done

echo ""
if curl -sf http://localhost:${PORT}/health >/dev/null 2>&1; then
  BOOTSTRAPPED=$(curl -sf http://localhost:${PORT}/health | python3 -c "import sys,json; print(json.load(sys.stdin).get('bootstrapped','?'))")
  echo -e "  ${GRN}✓ Gateway ready${NC}  →  http://localhost:${PORT}"
  if [ "$BOOTSTRAPPED" = "False" ]; then
    echo -e "  ${YLW}⚠ Not configured yet — open browser to complete setup${NC}"
  else
    echo -e "  ${DIM}Bootstrapped: true${NC}"
  fi
else
  warn "Gateway may still be starting — check: docker logs gateway"
fi
echo ""
