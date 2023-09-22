#! /bin/bash

#########################
#  Project Related args
#########################
PROJ_DIR=/apdcephfs/share_300000800/user/swcho

CODE_DIR=${PROJ_DIR}/code/lit-GPT
DATA_PATH=${PROJ_DIR}/data
HF_DIR=${PROJ_DIR}/huggingface_models
OUTPUT_DIR=${PROJ_DIR}/output

EXP_NAME="retnet_3b_2k_64GPU"

OUTPUT_EXP_DIR=${OUTPUT_DIR}/${EXP_NAME}
OUTPUT_CODE_DIR=${OUTPUT_EXP_DIR}/code

if [[ "${INDEX}" == "0" ]]; then
  mkdir -p $OUTPUT_EXP_DIR
  rm -rf $OUTPUT_CODE_DIR
  cp -r $CODE_DIR $OUTPUT_CODE_DIR
else
  sleep 30
fi

#########################
#  Debug args
#########################
export PYTORCH_JIT=0
#########################
#  NCCL and CUDA args
#########################
export CUDA_DEVICE_MAX_CONNECTIONS=1
NET_TYPE="high"
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

LAUNCH_ARGS="
             --devices ${GPUS_PER_NODE} \
             --num_nodes ${HOST_NUM} \
             --node_rank $NODE_RANK \
             --main_address $MASTER_ADDR \
             --main_port $MASTER_PORT \
             "
#             --accelerator cuda \
#              --strategy fsdp \
#              --precision bf16-mixed \

DATA_ARGS="--out_dir ${OUTPUT_EXP_DIR} \
--hf_dir ${HF_DIR}"

TRAIN_ARGS="--exp_name ${EXP_NAME} \
--model_name retnet_3b \
--max_iters 200000 \
--warmup_iters 3000 \
--save_interval 500 \
--eval_interval 1000 \
--log_interval 1 \
--eval_iters 20 \
--micro_batch_size 4 \
--batch_size 32 \
--learning_rate 3e-4 \
--train_data_dir /apdcephfs/share_300000800/user/swcho/data/pretrain_retnet \
--val_data_dir /apdcephfs/share_300000800/user/swcho/data/pretrain_retnet \
--prefix PCS-merged-360G \
"
#--devices ${GPUS_PER_NODE} \
#--num_nodes ${HOST_NUM}"

ALL_ARGS="${TRAIN_ARGS} \
          ${DATA_ARGS}"

echo ${ALL_ARGS}
echo ${LAUNCH_ARGS}

CMD="lightning run model ${LAUNCH_ARGS} \
${OUTPUT_CODE_DIR}}/pretrain/retnet_trainer_fabric.py ${ALL_ARGS}"
echo $CMD

rm ${OUTPUT_EXP_DIR}/log_node_${INDEX}.txt
eval ${CMD} 2>&1 | tee -a ${OUTPUT_EXP_DIR}/log_node_${INDEX}.txt

# mirrors.tencent.com/seattle-nlu/litgpt:v6.1
# mirrors.tencent.com/ai-lab-seattle/retnet:v911