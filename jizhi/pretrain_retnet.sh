#! /bin/bash

#########################
#  Project Related args
#########################
PROJ_DIR=/apdcephfs/share_300000800/user/swcho

CODE_DIR=${PROJ_DIR}/code/lit-GPT
DATA_PATH=${PROJ_DIR}/data
OUTPUT_DIR=${PROJ_DIR}/output
HF_DIR=${PROJ_DIR}/huggingface_models

EXP_NAME="retnet_3b_redpajama_sample"

mkdir -p ${OUTPUT_DIR}/${EXP_NAME}

#########################
#  Debug args
#########################
export PYTORCH_JIT=0
#########################
#  NCCL and CUDA args
#########################
export CUDA_DEVICE_MAX_CONNECTIONS=1
NET_TYPE="low"
if [[ "${NET_TYPE}" = "low" ]]; then
    export NCCL_SOCKET_IFNAME=eth1
    export NCCL_IB_GID_INDEX=3
    export NCCL_IB_HCA=mlx5_2:1,mlx5_2:1
    export NCCL_IB_SL=3
    export NCCL_CHECK_DISABLE=1
    export NCCL_P2P_DISABLE=0
    export NCCL_LL_THRESHOLD=16384
    export NCCL_IB_CUDA_SUPPORT=1
else
    export NCCL_IB_GID_INDEX=3
    export NCCL_IB_SL=3
    export NCCL_CHECK_DISABLE=1
    export NCCL_P2P_DISABLE=0
    export NCCL_IB_DISABLE=0
    export NCCL_LL_THRESHOLD=16384
    export NCCL_IB_CUDA_SUPPORT=1
    export NCCL_SOCKET_IFNAME=bond1
    export UCX_NET_DEVICES=bond1
    export NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6
    export NCCL_COLLNET_ENABLE=0
    export SHARP_COLL_ENABLE_SAT=0
    export NCCL_NET_GDR_LEVEL=2
    export NCCL_IB_QPS_PER_CONNECTION=4
    export NCCL_IB_TC=160
    export NCCL_PXN_DISABLE=1
fi

GPUS_PER_NODE=8
export MASTER_ADDR=${CHIEF_IP}
export MASTER_PORT=10900
NNODES=${HOST_NUM}
NODE_RANK=${INDEX}
export WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))

export HF_DATASETS_CACHE=${PROJ_DIR}/.cache/
export TRANSFORMERS_CACHE=${PROJ_DIR}/.cache/

DATA_ARGS="--train_data_dir ${DATA_PATH}/lit-redpajama-sample \
--out_dir ${OUTPUT_DIR} \
--hf_dir ${HF_DIR}"

TRAIN_ARGS="--exp_name ${EXP_NAME} \
--model_name retnet_3b \
--save_interval 1000 \
--eval_interval 1000 \
--eval_iters 1 \
--log_interval 2 \
--micro_batch_size 2 \
--batch_size 2 \
--devices ${GPUS_PER_NODE} \
--num_nodes ${HOST_NUM}"

ALL_ARGS="${TRAIN_ARGS} \
          ${DATA_ARGS}"

echo ${ALL_ARGS}

CMD="python3 ${CODE_DIR}/pretrain/retnet_trainer_fabric.py ${ALL_ARGS}"
echo $CMD

eval ${CMD} 2>&1 | tee -a ${OUTPUT_DIR}/${EXP_NAME}/log_node_${INDEX}.txt

# mirrors.tencent.com/seattle-nlu/litgpt:v6.1
# mirrors.tencent.com/ai-lab-seattle/retnet:v911