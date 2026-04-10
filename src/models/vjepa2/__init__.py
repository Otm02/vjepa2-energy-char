# === import necessary modules ===
import src.config as config
import src.trainer as trainer
import src.trainer.stats as trainer_stats

# === import necessary external modules ===
from typing import Any, Dict, Optional, Tuple
import torch.utils.data as data

model_name = "vjepa2"


def init_model(
    conf: config.Config, dataset: data.Dataset
) -> Tuple[trainer.Trainer, Optional[Dict[str, Any]]]:
    from src.models.vjepa2.model import vjepa2_init
    return vjepa2_init(conf, dataset)