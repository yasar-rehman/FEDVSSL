#!/usr/bin/env bash
CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}
export NCCL_IB_DISABLE=1
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train_net.py --cfg $CONFIG --launcher pytorch ${@:3}
#python -m $(dirname "$0")/train_net.py --cfg $CONFIG ${@:3}
