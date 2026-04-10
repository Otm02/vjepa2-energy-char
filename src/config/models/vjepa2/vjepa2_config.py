from src.config.util.base_config import _BaseConfig, _Arg

config_name = "vjepa2"


class ModelConfig(_BaseConfig):
    """Configuration for the V-JEPA2 model."""
    
    def __init__(self):
        super().__init__()
        self._arg_model_name = _Arg(
            type=str,
            help="Name of the ViT model (e.g., vit_huge, vit_large).",
            default="vit_huge"
        )
        self._arg_pred_depth = _Arg(
            type=int,
            help="Depth of the predictor network.",
            default=12
        )
        self._arg_pred_embed_dim = _Arg(
            type=int,
            help="Embedding dimension of the predictor.",
            default=384
        )
        self._arg_crop_size = _Arg(
            type=int,
            help="Spatial crop size for video frames.",
            default=224
        )
        self._arg_patch_size = _Arg(
            type=int,
            help="Patch size for the ViT.",
            default=16
        )
        self._arg_num_frames = _Arg(
            type=int,
            help="Number of frames per video clip.",
            default=16
        )
        self._arg_tubelet_size = _Arg(
            type=int,
            help="Temporal tubelet size.",
            default=2
        )
        self._arg_num_clips = _Arg(
            type=int,
            help="Number of clips per video.",
            default=1
        )
        self._arg_sampling_rate = _Arg(
            type=int,
            help="Frame sampling rate.",
            default=4
        )
        self._arg_num_workers = _Arg(
            type=int,
            help="Number of dataloader worker processes. Defaults to 0 for the current integration.",
            default=0
        )
        self._arg_use_sdpa = _Arg(
            type=bool,
            help="Whether to use scaled dot-product attention.",
            default=True
        )
        self._arg_dtype = _Arg(
            type=str,
            help="Data type for mixed precision (bfloat16, float16, float32).",
            default="bfloat16"
        )
        self._arg_epochs = _Arg(
            type=int,
            help="Number of training epochs.",
            default=2
        )
        self._arg_warmup = _Arg(
            type=int,
            help="Number of warmup epochs.",
            default=1
        )
        self._arg_lr = _Arg(
            type=float,
            help="Peak learning rate.",
            default=0.000625
        )
        self._arg_weight_decay = _Arg(
            type=float,
            help="Weight decay.",
            default=0.04
        )
        self._arg_clip_grad = _Arg(
            type=float,
            help="Gradient clipping value.",
            default=10.0
        )
        self._arg_loss_exp = _Arg(
            type=float,
            help="Loss exponent (1.0 = L1, 2.0 = L2).",
            default=1.0
        )
        self._arg_reg_coeff = _Arg(
            type=float,
            help="Regularization coefficient.",
            default=0.0
        )
        self._arg_ema_start = _Arg(
            type=float,
            help="EMA momentum start value.",
            default=0.998
        )
        self._arg_ema_end = _Arg(
            type=float,
            help="EMA momentum end value.",
            default=1.0
        )
        self._arg_ipe = _Arg(
            type=int,
            help="Iterations per epoch (for scheduler).",
            default=300
        )
        self._arg_max_runtime_minutes = _Arg(
            type=float,
            help="Stop after this many minutes of wall-clock training time. Use 0 to disable.",
            default=0.0
        )
        self._arg_max_steps = _Arg(
            type=int,
            help="Stop after this many optimizer-update steps. Use 0 to disable.",
            default=0
        )
