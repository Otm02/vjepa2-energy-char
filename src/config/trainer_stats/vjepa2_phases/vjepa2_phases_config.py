from src.config.util.base_config import _BaseConfig, _Arg

config_name = "vjepa2_phases"


class TrainerStatsConfig(_BaseConfig):
    """Configuration for V-JEPA2 fine-grained phase energy tracking."""
    
    def __init__(self):
        super().__init__()
        self._arg_output_dir = _Arg(
            type=str,
            help="Directory to save phase energy measurement files.",
            default="./vjepa2_energy_logs"
        )
        self._arg_run_num = _Arg(
            type=int,
            help="Run number for distinguishing repeated experiments.",
            default=1
        )
        self._arg_measure_power_secs = _Arg(
            type=float,
            help="Power measurement interval in seconds.",
            default=0.5
        )
        self._arg_project_name = _Arg(
            type=str,
            help="CodeCarbon project name.",
            default="vjepa2"
        )