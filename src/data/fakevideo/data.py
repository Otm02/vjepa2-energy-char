import os
import sys
import logging
import src.config as config
import torch.utils.data

logger = logging.getLogger(__name__)

data_load_name = "fakevideo"


def load_data(conf: config.Config) -> torch.utils.data.Dataset:
    """
    Load the FakeVideo dataset for V-JEPA2 training.
    
    This returns a simple placeholder dataset. The actual data loading 
    for V-JEPA2 is handled internally by the jepa library's init_data 
    function, which reads from the CSV manifest file. We return a 
    minimal dataset here to satisfy the starter code's interface.
    
    Parameters
    ----------
    conf : config.Config
        The configuration object.
    
    Returns
    -------
    torch.utils.data.Dataset
        A placeholder dataset.
    """
    csv_path = conf.data_configs.fakevideo.csv_path
    if not csv_path or not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"FakeVideo CSV file not found at: {csv_path}. "
            "Generate it first using the dataset generation script."
        )
    logger.info(f"FakeVideo dataset CSV path: {csv_path}")
    
    # Return a simple placeholder dataset - V-JEPA manages its own data loading
    return PlaceholderDataset(csv_path)


class PlaceholderDataset(torch.utils.data.Dataset):
    """
    A minimal dataset that stores the CSV path.
    V-JEPA2's actual data loading is done internally via init_data().
    """
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        # Count lines in CSV for length
        with open(csv_path, 'r') as f:
            self.length = sum(1 for _ in f)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        return idx