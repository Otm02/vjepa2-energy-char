"""
V-JEPA2 Model Initialization
==============================
Initializes V-JEPA2 model components and returns a VJepa2Trainer.

Based on the MilaBench vjepa benchmark:
https://github.com/mila-iqia/milabench/blob/master/benchmarks/vjepa/main.py
"""

import copy
import logging
import os
import sys
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision  # Pre-import: avoids inspect crash during sys.modules swap

import src.config as config
import src.trainer as trainer
import src.trainer.stats as trainer_stats

logger = logging.getLogger(__name__)

# Add the jepa library to the path
JEPA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jepa")
if JEPA_DIR not in sys.path:
    sys.path.insert(0, JEPA_DIR)


def get_dtype(dtype_str: str) -> torch.dtype:
    """Convert string dtype to torch.dtype."""
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return dtype_map.get(dtype_str, torch.bfloat16)


def vjepa2_init(
    conf: config.Config, dataset: data.Dataset
) -> Tuple[trainer.Trainer, Optional[Dict[str, Any]]]:
    """
    Initialize V-JEPA2 model and trainer.
    
    Parameters
    ----------
    conf : config.Config
        Configuration object with model and training parameters.
    dataset : data.Dataset
        The placeholder dataset (contains csv_path).
    
    Returns
    -------
    Tuple[trainer.Trainer, Optional[Dict[str, Any]]]
        The VJepa2Trainer and optional additional info.
    """
    # ------------------------------------------------------------------
    # Import jepa components
    # ------------------------------------------------------------------
    # The jepa submodule has its own internal 'src' package whose modules
    # do things like `from src.models.utils.patch_embed import ...`.
    # That collides with the *project's* top-level 'src' package which
    # is already on sys.path and cached in sys.modules.  To work around
    # this we must:
    #   1. Save & remove every 'src' / 'src.*' entry from sys.modules
    #   2. Temporarily remove the project root from sys.path so that
    #      Python can no longer find the project's src/ directory
    #   3. Import AND USE all jepa modules while the swap is active
    #      (jepa code does lazy imports like `from src.datasets...`
    #       inside function bodies, so the swap must stay active
    #       through init_data() and any other jepa function calls)
    #   4. Restore sys.path and sys.modules to their original state
    # ------------------------------------------------------------------

    # Compute the project root (three levels up from this file)
    _project_root = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
    )

    # 1) Temporarily remove the project's src.* from sys.modules
    _saved_modules = {}
    for key in list(sys.modules.keys()):
        if key == "src" or key.startswith("src."):
            _saved_modules[key] = sys.modules.pop(key)

    # 2) Temporarily remove project root from sys.path so only JEPA_DIR's src/ is visible
    _saved_path = list(sys.path)
    sys.path = [JEPA_DIR] + [
        p for p in sys.path
        if os.path.normpath(p) != _project_root and p != JEPA_DIR
    ]

    jepa_worker_modules: Dict[str, Any] = {}

    try:
        # 3) Now 'src' resolves to the jepa submodule's own src/
        #    Keep the swap active for ALL jepa-dependent code because
        #    jepa functions do lazy imports (e.g. init_data -> from src.datasets.video_dataset ...)
        from src.models.vision_transformer import vit_huge, vit_large, vit_base
        from src.models.predictor import vit_predictor
        from src.masks.random_tube import MaskCollator as TubeMaskCollator
        from src.masks.multiblock3d import MaskCollator as MB3DMaskCollator
        from src.datasets.data_manager import init_data
        from src.utils.schedulers import WarmupCosineSchedule, CosineWDSchedule
        from app.vjepa.transforms import make_transforms

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mcfg = conf.model_configs.vjepa2

        # --- Model parameters ---
        model_name_str = mcfg.model_name  # e.g., "vit_huge"
        pred_depth = mcfg.pred_depth
        pred_embed_dim = mcfg.pred_embed_dim
        patch_size = mcfg.patch_size
        num_frames = mcfg.num_frames
        tubelet_size = mcfg.tubelet_size
        crop_size = mcfg.crop_size
        dtype = get_dtype(mcfg.dtype)
        mixed_precision = mcfg.dtype != "float32"
        use_sdpa = mcfg.use_sdpa

        # --- Select ViT model ---
        model_map = {
            "vit_huge": vit_huge,
            "vit_large": vit_large,
            "vit_base": vit_base,
        }
        vit_fn = model_map.get(model_name_str, vit_huge)

        # --- Initialize encoder ---
        encoder = vit_fn(
            img_size=crop_size,
            patch_size=patch_size,
            num_frames=num_frames,
            tubelet_size=tubelet_size,
            uniform_power=True,
            use_sdpa=use_sdpa,
        )
        encoder = encoder.to(device)
        embed_dim = encoder.embed_dim
        num_heads = encoder.num_heads

        # --- Initialize predictor ---
        predictor = vit_predictor(
            img_size=crop_size,
            use_mask_tokens=True,
            patch_size=patch_size,
            num_frames=num_frames,
            tubelet_size=tubelet_size,
            embed_dim=embed_dim,
            predictor_embed_dim=pred_embed_dim,
            depth=pred_depth,
            num_heads=num_heads,
            uniform_power=True,
            num_mask_tokens=2,  # 2 mask types (from config)
            zero_init_mask_tokens=True,
            use_sdpa=use_sdpa,
        )
        predictor = predictor.to(device)

        # --- Initialize target encoder (EMA) ---
        target_encoder = copy.deepcopy(encoder)
        for p in target_encoder.parameters():
            p.requires_grad = False
        target_encoder = target_encoder.to(device)

        logger.info(f"Encoder parameters: {sum(p.numel() for p in encoder.parameters()):,}")
        logger.info(f"Predictor parameters: {sum(p.numel() for p in predictor.parameters()):,}")

        # --- Initialize mask collator ---
        mask_collator = MB3DMaskCollator(
            crop_size=crop_size,
            num_frames=num_frames,
            patch_size=patch_size,
            tubelet_size=tubelet_size,
            cfgs_mask=[
                {
                    "aspect_ratio": [0.75, 1.5],
                    "num_blocks": 8,
                    "spatial_scale": [0.15, 0.15],
                    "temporal_scale": [1.0, 1.0],
                    "max_temporal_keep": 1.0,
                    "max_keep": None,
                },
                {
                    "aspect_ratio": [0.75, 1.5],
                    "num_blocks": 2,
                    "spatial_scale": [0.7, 0.7],
                    "temporal_scale": [1.0, 1.0],
                    "max_temporal_keep": 1.0,
                    "max_keep": None,
                },
            ],
        )

        # --- Create video transform ---
        # The jepa VideoDataset returns clips as numpy arrays in (T, H, W, C)
        # format.  The transform converts them to (C, T, H, W) tensors, crops
        # to crop_size, and normalises pixel values.  Without it the model
        # receives scrambled dimensions and crashes.
        transform = make_transforms(
            random_horizontal_flip=True,
            random_resize_aspect_ratio=[0.75, 1.35],
            random_resize_scale=[0.3, 1.0],
            reprob=0.0,
            auto_augment=False,
            motion_shift=False,
            crop_size=crop_size,
        )

        # --- Initialize data loader ---
        csv_path = dataset.csv_path  # From PlaceholderDataset
        batch_size = conf.batch_size
        num_clips = mcfg.num_clips
        sampling_rate = mcfg.sampling_rate
        # Exposed for worker sweeps; the main 18-run matrix keeps num_workers=0
        # for comparability. When num_workers>0, src.datasets and src.masks are
        # re-registered after the JEPA sys.modules swap so DataLoader workers
        # can unpickle MaskCollator.
        num_workers = mcfg.num_workers

        (unsupervised_loader, unsupervised_sampler) = init_data(
            batch_size=batch_size,
            transform=transform,
            data="VideoDataset",
            root_path=[csv_path],
            collator=mask_collator,
            pin_mem=True,
            num_workers=num_workers,
            world_size=1,
            rank=0,
            training=True,
            clip_len=num_frames,
            frame_sample_rate=sampling_rate,
            duration=None,
            num_clips=num_clips,
            filter_short_videos=False,
            decode_one_clip=True,
            drop_last=True,
            log_dir=None,
        )
        # Snapshot JEPA modules needed for DataLoader worker pickling: datasets,
        # transforms, and MaskCollator (src.masks.*). Without src.masks, workers
        # fail with PicklingError / No module named 'src.masks'.
        jepa_worker_modules = {
            key: module
            for key, module in sys.modules.items()
            if (
                key == "src.datasets"
                or key.startswith("src.datasets.")
                or key == "src.masks"
                or key.startswith("src.masks.")
            )
        }

    finally:
        # 4) Restore sys.path
        sys.path = _saved_path
        # Clean up jepa's src.* entries from sys.modules
        for key in list(sys.modules.keys()):
            if key == "src" or key.startswith("src."):
                del sys.modules[key]
        # Restore the project's src.* entries
        sys.modules.update(_saved_modules)

    if num_workers > 0 and jepa_worker_modules:
        # Keep vendored JEPA modules registered under `src.datasets.*` and
        # `src.masks.*` so worker processes resolve the same classes as the parent.
        sys.modules.update(jepa_worker_modules)

    # This import uses the *project's* src, which is now back in place
    from src.trainer.vjepa2_trainer import VJepa2Trainer

    # --- Optimizer ---
    param_groups = [
        {"params": (p for n, p in encoder.named_parameters() if p.requires_grad)},
        {"params": (p for n, p in predictor.named_parameters() if p.requires_grad)},
    ]

    num_epochs = mcfg.epochs
    max_runtime_minutes = mcfg.max_runtime_minutes
    max_steps = mcfg.max_steps
    ipe = len(unsupervised_loader)  # iterations per epoch
    ipe_scale = 1.25

    optimizer = torch.optim.AdamW(param_groups)

    # --- Schedulers ---
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=int(mcfg.warmup * ipe * ipe_scale),
        start_lr=0.0002,
        ref_lr=mcfg.lr,
        final_lr=1.0e-06,
        T_max=int(num_epochs * ipe * ipe_scale),
    )
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=mcfg.weight_decay,
        final_wd=0.4,
        T_max=int(num_epochs * ipe * ipe_scale),
    )

    # EMA momentum schedule
    ema_start = mcfg.ema_start
    ema_end = mcfg.ema_end
    total_steps = num_epochs * ipe
    momentum_schedule = (
        ema_start + i * (ema_end - ema_start) / total_steps
        for i in range(total_steps + 1)
    )

    # Mixed precision scaler
    scaler = torch.amp.GradScaler("cuda") if mixed_precision else None

    # --- Initialize trainer stats ---
    stats_obj = trainer_stats.init_from_conf(
        conf=conf,
        device=device,
        num_train_steps=ipe * num_epochs,
    )

    # --- Create VJepa2Trainer ---
    vjepa_trainer = VJepa2Trainer(
        loader=unsupervised_loader,
        encoder=encoder,
        predictor=predictor,
        target_encoder=target_encoder,
        optimizer=optimizer,
        scheduler=scheduler,
        wd_scheduler=wd_scheduler,
        momentum_scheduler=momentum_schedule,
        device=device,
        stats=stats_obj,
        loss_exp=mcfg.loss_exp,
        reg_coeff=mcfg.reg_coeff,
        clip_grad=mcfg.clip_grad,
        num_epochs=num_epochs,
        mixed_precision=mixed_precision,
        dtype=dtype,
        scaler=scaler,
        batch_size=batch_size,
        num_clips=num_clips,
        max_runtime_seconds=max_runtime_minutes * 60.0 if max_runtime_minutes > 0 else None,
        max_steps=max_steps if max_steps > 0 else None,
    )

    return vjepa_trainer, None
