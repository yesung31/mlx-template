import mlx.nn as nn
import mlx.utils as mlx_utils
from rich import box
from rich.console import Console
from rich.table import Table


class ModelSummary:
    def __init__(self, model: nn.Module, max_depth: int = 1):
        self.model = model
        self.max_depth = max_depth
        self.console = Console()

    def _get_params(self, module):
        num_params = 0
        trainable_params = 0
        param_size = 0

        for _, p in mlx_utils.tree_flatten(module.parameters()):
            n = p.size
            num_params += n
            trainable_params += n
            param_size += p.nbytes

        return num_params, trainable_params, param_size

    def _format_size(self, num, suffix="B"):
        for unit in ["", "Ki", "Mi", "Gi", "Ti"]:
            if abs(num) < 1024.0:
                return f"{num:3.1f} {unit}{suffix}"
            num /= 1024.0
        return f"{num:.1f} Pi{suffix}"

    def summarize(self):
        table = Table(header_style="bold magenta", box=box.SIMPLE)
        table.add_column(" ", style="dim")
        table.add_column("Name", justify="left", style="cyan", no_wrap=True)
        table.add_column("Type", style="green")
        table.add_column("Params", justify="right")
        table.add_column("Mode", justify="right")

        rows = []

        def walk(module, name_prefix="", depth=0):
            if depth >= self.max_depth:
                return

            for name, child in module.children().items():
                full_name = f"{name_prefix}.{name}" if name_prefix else name

                if isinstance(child, nn.Module):
                    p_count, _, _ = self._get_params(child)
                    rows.append((full_name, child.__class__.__name__, f"{p_count:,}", "Trainable"))

        walk(self.model)

        for idx, row in enumerate(rows):
            table.add_row(str(idx), *row)

        self.console.print(table)

        total_params, total_trainable, total_size = self._get_params(self.model)
        non_trainable = total_params - total_trainable

        self.console.print(f"[bold]Trainable params[/bold]: {total_trainable:,}")
        self.console.print(f"[bold]Non-trainable params[/bold]: {non_trainable:,}")
        self.console.print(f"[bold]Total params[/bold]: {total_params:,}")
        self.console.print(
            f"[bold]Total estimated model params size[/bold]: {self._format_size(total_size)}"
        )
