import numpy as np
import torch
import math
from scipy.special import expi
import warnings
from scipy.integrate import quad
from config import Config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VirtualClient:
    def __init__(self, client_id, data_loader, config_instance: Config):
        self.client_id = client_id
        self.loader = data_loader
        self.config = config_instance
        self.cpu_freq = np.float32(np.random.uniform(*self.config.CPU_FREQ_RANGE_GHZ))
        self.channel_gain_variance = np.float32(np.random.uniform(*self.config.CHANNEL_GAIN_VAR_RANGE))
        self.cpu_cycles_per_flop = np.float32(np.random.uniform(*self.config.CPU_CYCLES_PER_FLOP_RANGE))
        self.bandwidth_hz = (self.config.TOTAL_BANDWIDTH_MHZ * 1e6) / self.config.num_clients
        self.tx_power_watts = 10 ** ((self.config.TX_POWER_DBM - 30) / 10)
        self.noise_power_watts = 10 ** ((self.config.NOISE_POWER_DBM - 30) / 10)
        self.bits_per_param = self.config.BITS_PER_PARAM
        self._expected_comp_time = None
        self._expected_comm_time = None

    def get_context_features(self) -> torch.Tensor:
        context_tensor = torch.tensor([[self.cpu_freq, self.channel_gain_variance]], device=device, dtype=torch.float32)
        return context_tensor

    
