from src.config.util.base_config import _BaseConfig, _Arg

config_name = "fakevideo"


class DataConfig(_BaseConfig):
    """Configuration for the FakeVideo dataset used by V-JEPA2."""
    
    def __init__(self):
        super().__init__()
        self._arg_csv_path = _Arg(
            type=str,
            help="Path to the CSV file listing video paths and labels.",
            default=""
        )
        self._arg_num_videos = _Arg(
            type=int,
            help="Number of fake videos to use.",
            default=200
        )