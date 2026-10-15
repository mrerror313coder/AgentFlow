#!/usr/bin/env bash
# One-command DigitalOcean Droplet deploy for AgentFlow.
# Creates an Ubuntu 22.04 2 GB Droplet, installs Python 3.11 and Ollama,
# pulls llama3, and starts the API plus the Streamlit dashboard.
#
# Usage:
#   export REPO_URL=https://github.com/mrerror313coder/AgentFlow.git
#   bash deploy_droplet.sh
#
# Requires an authenticated doctl CLI and at least one SSH key on the account.

set -euo pipefail

DROPLET_NAME="${DROPLET_NAME:-agentflow}"
REGION="${REGION:-nyc3}"
SIZE="${SIZE:-s-1vcpu-2gb}"
IMAGE="${IMAGE:-ubuntu-22-04-x64}"
REPO_URL="${REPO_URL:-https://github.com/mrerror313coder/AgentFlow.git}"
OLLAMA_MODEL="${OLLAMA_MODEL:-llama3}"

log() {
  printf '\n==> %s\n' "$1"
}

if ! command -v doctl >/dev/null 2>&1; then
  echo "doctl is not installed. Install it from https://docs.digitalocean.com/reference/doctl/how-to/install/"
  exit 1
fi

if ! doctl account get >/dev/null 2>&1; then
  echo "doctl is not authenticated. Run: doctl auth init"
  exit 1
fi

SSH_KEY_ID="$(doctl compute ssh-key list --format ID --no-header | head -n 1 || true)"
if [ -z "${SSH_KEY_ID}" ]; then
  echo "No SSH keys found on this DigitalOcean account. Add one with: doctl compute ssh-key import"
  exit 1
fi

existing_ip="$(doctl compute droplet list --format Name,PublicIPv4 --no-header | awk -v name="$DROPLET_NAME" '$1 == name { print $2; exit }')"
if [ -n "${existing_ip}" ]; then
  log "Droplet ${DROPLET_NAME} already exists at ${existing_ip}"
  IP="${existing_ip}"
else
  log "Creating Droplet ${DROPLET_NAME} (${SIZE}, ${IMAGE}, ${REGION})"
  doctl compute droplet create "${DROPLET_NAME}" \
    --region "${REGION}" \
    --size "${SIZE}" \
    --image "${IMAGE}" \
    --ssh-keys "${SSH_KEY_ID}" \
    --wait
  IP="$(doctl compute droplet list --format Name,PublicIPv4 --no-header | awk -v name="$DROPLET_NAME" '$1 == name { print $2; exit }')"
fi

if [ -z "${IP}" ]; then
  echo "Could not determine the Droplet public IP."
  exit 1
fi

log "Waiting for SSH on ${IP}"
ready=0
for _attempt in $(seq 1 30); do
  if ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new -o BatchMode=yes "root@${IP}" "echo ready" >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 10
done
if [ "${ready}" -ne 1 ]; then
  echo "SSH did not become ready on ${IP}."
  exit 1
fi

log "Installing Python 3.11, Ollama, and AgentFlow"
ssh -o StrictHostKeyChecking=accept-new "root@${IP}" \
  "REPO_URL=$(printf '%q' "${REPO_URL}") OLLAMA_MODEL=$(printf '%q' "${OLLAMA_MODEL}") bash -s" <<'REMOTE'
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y software-properties-common ca-certificates curl git ufw
add-apt-repository -y universe
apt-get update
if ! apt-get install -y python3.11 python3.11-venv python3-pip; then
  add-apt-repository -y ppa:deadsnakes/ppa
  apt-get update
  apt-get install -y python3.11 python3.11-venv python3-pip
fi

if ! swapon --show | grep -q '/swapfile'; then
  fallocate -l 4G /swapfile || dd if=/dev/zero of=/swapfile bs=1M count=4096
  chmod 600 /swapfile
  mkswap /swapfile
  swapon /swapfile
  grep -q '/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi

if ! command -v ollama >/dev/null 2>&1; then
  curl -fsSL https://ollama.com/install.sh | sh
fi
systemctl enable ollama
systemctl start ollama
sleep 3
ollama pull "${OLLAMA_MODEL}"

mkdir -p /opt
if [ -d /opt/agentflow/.git ]; then
  git -C /opt/agentflow pull --ff-only
else
  rm -rf /opt/agentflow
  git clone "${REPO_URL}" /opt/agentflow
fi

python3.11 -m venv /opt/agentflow/.venv
/opt/agentflow/.venv/bin/pip install --upgrade pip
/opt/agentflow/.venv/bin/pip install -r /opt/agentflow/requirements.txt
mkdir -p /opt/agentflow/exports

ufw allow OpenSSH
ufw allow 8000/tcp
ufw allow 8501/tcp
ufw --force enable

pkill -f "uvicorn backend.main:app" || true
pkill -f "streamlit run app.py" || true

cd /opt/agentflow
export OLLAMA_HOST=http://127.0.0.1:11434
nohup /opt/agentflow/.venv/bin/uvicorn backend.main:app --host 0.0.0.0 --port 8000 \
  > /var/log/agentflow-api.log 2>&1 &
nohup /opt/agentflow/.venv/bin/streamlit run app.py \
  --server.port 8501 \
  --server.address 0.0.0.0 \
  --server.headless true \
  > /var/log/agentflow-ui.log 2>&1 &
REMOTE

cat <<EOF

AgentFlow is deploying on the Droplet.

Droplet IP:  ${IP}
API:         http://${IP}:8000
API docs:    http://${IP}:8000/docs
Dashboard:   http://${IP}:8501
Ollama:      http://127.0.0.1:11434 (bound to the Droplet, not the public internet)

Logs:
  ssh root@${IP} tail -f /var/log/agentflow-api.log
  ssh root@${IP} tail -f /var/log/agentflow-ui.log

Give the processes about 20 seconds to finish installing packages before opening the URLs.
llama3 needs the 4 GB swap file created above; a larger Droplet is more comfortable for the model.
EOF
