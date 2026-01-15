#!/bin/bash

# ==========================================
# 1. Environment Configuration (Edit here)
# ==========================================

# [Network Settings]

NETWORK_IF="10.0.0.0/24"       # Thunderbolt Bridge Subnet

# [Host: Local Machine (M3 Pro)]
HOST_NAME="localhost"          # Host Address
HOST_SLOTS="1"                 # Number of processes/slots (usually 1 for single GPU)
HOST_PYTHON="/Users/a0000/local_A/.venv/bin/python"
HOST_DIR="/Users/a0000/local_A/TB4_Bridge"

# [Node: Remote Machine (M4 Mac Mini)]
NODE_IP="10.0.0.2"             # Remote Node IP
NODE_USER="playlab"            # Remote User Account Name
NODE_SLOTS="1"                 # Number of processes/slots
NODE_PYTHON="/Users/playlab/local_A/.venv/bin/python"
NODE_DIR="/Users/playlab/local_A/TB4_Bridge"

# ==========================================
# 2. Execution Logic
# ==========================================

SCRIPT_NAME=$1
shift # Remove the first argument (filename) so remaining args are passed to the script

if [ -z "$SCRIPT_NAME" ]; then
  echo "Usage: ./run_cluster.sh [python_script.py] [args...]"
  exit 1
fi

echo "Launching Cluster..."
echo " - Host: $HOST_NAME ($HOST_SLOTS process)"
echo " - Node: $NODE_USER@$NODE_IP ($NODE_SLOTS process)"
echo " - Network: $NETWORK_IF"
echo "---------------------------------------------------"

# Execute MPI command
mpirun \
    --mca btl_tcp_if_include $NETWORK_IF \
    -tag-output \
    -x DYLD_LIBRARY_PATH=/opt/homebrew/lib:$DYLD_LIBRARY_PATH \
    -x PATH \
    -x OMPI_MCA_btl_tcp_if_include=$NETWORK_IF \
    -np $HOST_SLOTS -H $HOST_NAME:$HOST_SLOTS \
    $HOST_PYTHON $HOST_DIR/$SCRIPT_NAME "$@" \
    : \
    -np $NODE_SLOTS -H $NODE_USER@$NODE_IP:$NODE_SLOTS \
    $NODE_PYTHON $NODE_DIR/$SCRIPT_NAME "$@"