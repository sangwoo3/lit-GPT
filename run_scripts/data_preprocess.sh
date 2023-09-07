# on P40
python scripts/prepare_redpajama_retnet.py \
--source_path data/RedPajama-Data-1T-Sample \
--checkpoint_dir /data2/swcho_data/.cache/huggingface/transformers/models--meta-llama--Llama-2-7b-hf/snapshots/6fdf2e60f86ff2481f2241aaee459f85b5b0bbb9/ \
--destination_path data/lit-redpajama-sample \
--sample True