import torch.nn as nn

class AdaFMLModel(nn.Module):
    def __init__(self, dataset='mnist'):
        super().__init__()
        if dataset == 'mnist':
            self.task_model = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2), nn.Flatten(),
                nn.Linear(64 * 7 * 7, 256), nn.ReLU(), nn.Linear(256, 10)
            )
            self.input_example_shape = (1, 28, 28)
        elif dataset == 'cifar10':
            self.task_model = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2), nn.Flatten(),
                nn.Linear(128 * 8 * 8, 512), nn.ReLU(), nn.Linear(512, 10)
            )
            self.input_example_shape = (3, 32, 32)
        else:
            print(f"Warning: Unsupported dataset '{dataset}' for AdaFMLModel structure. Using default input shape.")
            self.task_model = nn.Identity()
            self.input_example_shape = (1, 28, 28)

        self.time_model = nn.Sequential(
            nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def forward(self, x, context):
        task_output = self.task_model(x)
        if context.dim() == 1: context = context.unsqueeze(0)
        if context.size(0) != x.size(0) and context.size(0) == 1:
            context_expanded = context.expand(x.size(0), -1)
        else:
            context_expanded = context
        time_output = self.time_model(context_expanded)
        return task_output, time_output
