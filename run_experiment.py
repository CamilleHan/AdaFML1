import torch
import datetime
from config import Config
from AdaFMLTrainer import AdaFMLTrainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_experiment(method='adafml', config_override=None):
    current_config = Config()
    if config_override:
        for key, value in config_override.items():
            if hasattr(current_config, key):
                setattr(current_config, key, value)
            else:
                print(f"Warning: Config key '{key}' not found.")
    current_config.log_dir = f'runs/{method}_TruncExpTime_{current_config.dataset}_{"iid" if current_config.iid else "noniid"}_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'

    print(f"Configuration: {vars(current_config)}")
    print(f"Logging to: {current_config.log_dir}")

    if method == 'adafml':
        try:
            trainer = AdaFMLTrainer(current_config)
        except ValueError as e:
            print(f"Error initializing trainer: {e}"); return None
    else:
        raise ValueError(f"Unsupported method: {method}")

    trainer.train()

    return {
        'losses': trainer.loss_records,
        'accuracy': trainer.accuracy_history,
        'log_dir': current_config.log_dir
    }


if __name__ == "__main__":
    results_adafml = run_experiment(method='adafml')

