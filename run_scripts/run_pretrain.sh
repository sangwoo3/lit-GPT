python pretrain/retnet_trainer_fabric.py \
--devices 8 \
--exp_name retnet_3b_redpajama_sample \
--train_data_dir data/lit-redpajama-sample \
#--val_data_dir data/lit-redpajama-sample-val \


lightning run model --devices 4 --num_nodes 1 pretrain/retnet_trainer_fabric_dev.py \
--exp_name retnet_3b_redpajama_sample --train_data_dir data/lit-redpajama-sample \
--val_data_dir data/lit-redpajama-sample --model_name retnet_medium --save_interval 25 --eval_interval 25 \
--max_iters 500 --eval_iters 10 --warmup_iters 50 --log_interval 1 --batch_size 4 --micro_batch_size 2