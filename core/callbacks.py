import os
import shutil
import mlx.core as mx

class ModelCheckpoint:
    def __init__(self, dirpath, monitor="val_loss", mode="min", save_top_k=1, save_last=True, verbose=False):
        self.dirpath = dirpath
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        self.save_last = save_last
        self.verbose = verbose
        
        self.best_k_models = {} # path -> score
        self.best_score = float('inf') if mode == 'min' else float('-inf')

        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

    def on_validation_end(self, trainer, pl_module):
        logs = trainer.logged_metrics
        if self.monitor not in logs:
            if self.verbose:
                print(f"ModelCheckpoint: {self.monitor} not found in logs. Skipping save.")
            return

        current_score = logs[self.monitor]
        
        # Check if should save
        improve = (current_score < self.best_score) if self.mode == 'min' else (current_score > self.best_score)
        
        if improve:
            self.best_score = current_score
            fname = f"epoch={trainer.current_epoch}-{self.monitor}={current_score:.4f}.npz"
            path = os.path.join(self.dirpath, fname)
            
            if self.verbose:
                print(f"ModelCheckpoint: {self.monitor} improved to {current_score:.4f}. Saving to {path}")
            
            # Save weights
            # MLX save format is typically npz or safetensors. Using mx.savez for simplicity with dict
            weights = dict(pl_module.parameters())
            # Flatten or keep structure? mx.save_safetensors is better for nested, but let's just use mx.save which handles dict of arrays
            # Actually pl_module.parameters() returns a nested dict. mx.save_safetensors is preferred.
            # but standard save might be simpler.
            pl_module.save_weights(path)
            
            self.best_k_models[path] = current_score
            
            # Handle top k (cleanup old)
            if len(self.best_k_models) > self.save_top_k:
                # remove worst
                sorted_models = sorted(self.best_k_models.items(), key=lambda x: x[1], reverse=(self.mode!='min'))
                to_remove_path, _ = sorted_models[-1] # worst is last if we sorted by best first? 
                # wait: if min mode (lower better), sorted ascending puts best first. worst is last.
                # if max mode (higher better), sorted ascending puts worst first.
                
                # easier: sort by "goodness". 
                # if min: smallest is best. sort ascending. [0] is best. [-1] is worst.
                # if max: largest is best. sort descending. [0] is best. [-1] is worst.
                reverse = True if self.mode == 'max' else False
                sorted_models = sorted(self.best_k_models.items(), key=lambda x: x[1], reverse=reverse)
                
                # Keep top k
                to_keep = sorted_models[:self.save_top_k]
                to_remove = sorted_models[self.save_top_k:]
                
                self.best_k_models = dict(to_keep)
                for p, s in to_remove:
                    if os.path.exists(p):
                        os.remove(p)

        if self.save_last:
            last_path = os.path.join(self.dirpath, "last.npz")
            pl_module.save_weights(last_path)

