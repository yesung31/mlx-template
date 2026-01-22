import os


class ModelCheckpoint:
    def __init__(
        self, dirpath, monitor="val_loss", mode="min", save_top_k=1, save_last=True, verbose=False
    ):
        self.dirpath = dirpath
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        self.save_last = save_last
        self.verbose = verbose

        self.best_k_models = {}  # path -> score
        self.best_score = float("inf") if mode == "min" else float("-inf")

        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

    def on_validation_end(self, trainer, pl_module):
        logs = trainer.logged_metrics
        if self.monitor not in logs:
            return

        current_score = logs[self.monitor]

        # Check if should save
        improve = (
            (current_score < self.best_score)
            if self.mode == "min"
            else (current_score > self.best_score)
        )

        if improve:
            self.best_score = current_score
            fname = f"epoch={trainer.current_epoch}-{self.monitor}={current_score:.4f}.npz"
            path = os.path.join(self.dirpath, fname)

            # Save weights
            pl_module.save_weights(path)

            self.best_k_models[path] = current_score

            # Handle top k (cleanup old)
            if len(self.best_k_models) > self.save_top_k:
                # remove worst
                reverse = True if self.mode == "max" else False
                sorted_models = sorted(
                    self.best_k_models.items(), key=lambda x: x[1], reverse=reverse
                )

                # Keep top k
                to_keep = sorted_models[: self.save_top_k]
                to_remove = sorted_models[self.save_top_k :]

                self.best_k_models = dict(to_keep)
                for p, s in to_remove:
                    if os.path.exists(p):
                        os.remove(p)

        if self.save_last:
            last_path = os.path.join(self.dirpath, "last.npz")
            pl_module.save_weights(last_path)
