import mlx.nn as nn

class TemplateNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Dummy network matching the dummy data (32 input dim, 2 output classes)
        self.layer = nn.Linear(32, 2)

    def __call__(self, x):
        return self.layer(x)
