from argparse import ArgumentParser


def arg_loader():
    parser = ArgumentParser()
    # Model
    parser.add_argument("--model_name", type=str, choices=['retnet_medium', 'retnet_xl', 'retnet_3b',
                                                           'retnet_7b', 'retnet_13b', 'retnet_65b'],
                        default='retnet_3b')
    parser.add_argument("--exp_name", type=str, default='pile-c4-stack')
    parser.add_argument("--save_interval", type=int, default=2500, help='based on steps (effective BS)')
    parser.add_argument("--eval_interval", type=int, default=1250, help='based on steps (effective BS)')
    parser.add_argument("--log_interval", type=int, default=160, help='based on interations')
    parser.add_argument("--eval_iters", type=int, default=1, help="validation iteration")

    # Hyper-parameters
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=64, help='effective BS with grad accum')
    parser.add_argument("--micro_batch_size", type=int, default=4, help='batch per each GPU')
    parser.add_argument("--max_iters", type=int, default=400000,
                        help="num_epochs * (epoch_size // micro_batch_size) // devices")
    parser.add_argument("--warmup_iters", type=int, default=600, help="1.5% of 400K iterations")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--decay_lr", action="store_false")
    parser.add_argument("--min_lr", type=float, default=6e-5)
    parser.add_argument("--precision", type=str)

    parser.add_argument("--gradient_clip_val", type=float, default=1.0)  # avoid fp16 nan
    # parser.add_argument("--devices", type=int, default=1)
    # parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=1)

    parser.add_argument("--train_data_dir", type=str,
                        default="/apdcephfs/share_300000800/user/swcho/data/pretrain_retnet")
    parser.add_argument("--val_data_dir", type=str)
    parser.add_argument("--prefix", type=str, default="PCS-merged-360G")
    parser.add_argument("--out_dir", type=str, default="out")
    parser.add_argument("--hf_dir", type=str)

    parser.add_argument("--block_size", type=int, default=2048)

    parser.add_argument("--resume", action="store_true")

    parser.add_argument("--seed", type=int, default=6060)

    args = parser.parse_args()
    return args


def retnet_base_architecture(args):
    # backward compatibility for older model checkpoints
    if hasattr(args, "no_tie_adaptive_proj"):
        # previous models defined --no-tie-adaptive-proj, so use the existence of
        # that option to determine if this is an "old" model checkpoint
        args.no_decoder_final_norm = True  # old models always set this to True
        if args.no_tie_adaptive_proj is False:
            args.tie_adaptive_proj = True
    if hasattr(args, "decoder_final_norm"):
        args.no_decoder_final_norm = not args.decoder_final_norm

    args.dropout = getattr(args, "dropout", 0.0)

    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_retention_heads = getattr(args, "decoder_retention_heads", 2)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.adaptive_softmax_factor = getattr(args, "adaptive_softmax_factor", 4)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.activation_fn = getattr(args, "activation_fn", "gelu")

    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0)
    args.decoder_layers_to_keep = getattr(args, "decoder_layers_to_keep", None)

    args.base_layers = getattr(args, "base_layers", 0)
    args.base_sublayers = getattr(args, "base_sublayers", 1)
    args.base_shuffle = getattr(args, "base_shuffle", False)

    args.add_bos_token = getattr(args, "add_bos_token", False)
    args.no_token_positional_embeddings = getattr(
            args, "no_token_positional_embeddings", False
    )
    args.share_decoder_input_output_embed = getattr(
            args, "share_decoder_input_output_embed", False
    )
    args.character_embeddings = getattr(args, "character_embeddings", False)

    args.decoder_output_dim = getattr(
            args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.chunkwise_recurrent = getattr(args, "chunkwise_recurrent", False)
    args.recurrent_chunk_size = getattr(args, "recurrent_chunk_size", 512)

    # Model training is not stable without this
    args.decoder_normalize_before = True
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", False)

    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.adaptive_input_factor = getattr(args, "adaptive_input_factor", 4)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", None)

    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", False)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.checkpoint_activations = getattr(args, "checkpoint_activations", False)
    args.offload_activations = getattr(args, "offload_activations", False)
    if args.offload_activations:
        args.checkpoint_activations = True


def retnet_medium(args):
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 2048)
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    args.decoder_retention_heads = getattr(args, "decoder_retention_heads", 4)
    retnet_base_architecture(args)


def retnet_xl(args):
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 2048)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_layers = getattr(args, "decoder_layers", 24)
    retnet_base_architecture(args)


def retnet_3b(args):
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 2560)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 5120)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 10)
    args.decoder_layers = getattr(args, "decoder_layers", 32)
    retnet_base_architecture(args)


def retnet_7b(args):
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 4096)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 8192)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.decoder_layers = getattr(args, "decoder_layers", 32)
    retnet_base_architecture(args)


def retnet_13b(args):
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 5120)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 10240)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 20)
    args.decoder_layers = getattr(args, "decoder_layers", 40)
    retnet_base_architecture(args)


def retnet_65b(args):
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 8192)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 16384)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 32)
    args.decoder_layers = getattr(args, "decoder_layers", 64)
    retnet_base_architecture(args)