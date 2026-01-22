import mlx.nn as nn
import mlx.utils as mlx_utils
from rich.table import Table
from rich import box
from rich.console import Console

class ModelSummary:
    def __init__(self, model: nn.Module, max_depth: int = 1):
        self.model = model
        self.max_depth = max_depth
        self.console = Console()

    def _get_params(self, module):
        num_params = 0
        trainable_params = 0
        param_size = 0
        
        # We need to flatten to count unique parameters correctly if shared
        # But for simple summary per layer, simple summation is often used.
        # MLX tree_flatten gives all arrays.
        
        params = module.parameters()
        for _, p in mlx_utils.tree_flatten(params):
            n = p.size
            num_params += n
            trainable_params += n # MLX doesn't have requires_grad flag on tensor yet easily accessible per-se in same way, usually all are trainable if in optimizer
            param_size += p.nbytes
            
        return num_params, trainable_params, param_size

    def summarize(self):
        table = Table(header_style="bold magenta", box=box.SIMPLE)
        table.add_column(" ", style="dim")
        table.add_column("Name", justify="left", style="cyan", no_wrap=True)
        table.add_column("Type", style="green")
        table.add_column("Params", justify="right")
        table.add_column("Mode", justify="right")

        total_params = 0
        total_trainable = 0
        total_size = 0

        # Root params (not in children)
        # It's hard to separate root params from children params in simple recursive walk if we just sum parameters()
        # So we iterate children.
        
        rows = []
        
        def walk(module, name_prefix="", depth=0):
            nonlocal total_params, total_trainable, total_size
            
            # If we are at max depth, we summarize this module fully
            # If we are strictly less than max depth, we might recurse?
            # Creating a flat list like PL
            
            if depth >= self.max_depth:
                return

            # In MLX, named_modules() isn't standard like PyTorch.
            # We inspect attributes.
            for name, child in module.children().items():
                full_name = f"{name_prefix}.{name}" if name_prefix else name
                
                if isinstance(child, nn.Module):
                    # Count params for this child
                    p_count, t_count, p_size = self._get_params(child)
                    
                    rows.append((full_name, child.__class__.__name__, f"{p_count:,}", "Trainable"))
                    
                    if depth == 0:
                        total_params += p_count
                        total_trainable += t_count
                        total_size += p_size
                        
                    # Recurse
                    # walk(child, full_name, depth + 1) # If we want nested. PL usually just shows top level or flattened.
                    # Let's stick to depth 1 (top level children) for the table like PL default.

        walk(self.model)
        
        for idx, row in enumerate(rows):
            table.add_row(str(idx), *row)

        self.console.print(table)
        
        # Summary footer
        # Re-calculate total exact from full model to be sure (in case of shared params or root params)
        final_p, final_t, final_s = self._get_params(self.model)
        
        # Convert size
        def sizeof_fmt(num, suffix="B"):
            for unit in ["", "Ki", "Mi", "Gi", "Ti"]:
                if abs(num) < 1024.0:
                    return f"{num:3.1f} {unit}{suffix}"
                num /= 1024.0
            return f"{num:.1f} Pi{suffix}"

        self.console.print(f"[bold]Trainable params[/bold]: {final_t:,}")
        self.console.print(f"[bold]Non-trainable params[/bold]: {final_p - final_t:,}")
        self.console.print(f"[bold]Total params[/bold]: {final_p:,}")
        self.console.print(f"[bold]Total estimated model params size[/bold]: {sizeof_fmt(final_s)}")

