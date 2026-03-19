
import matplotlib.pyplot as plt
from IPython.display import clear_output
import pytorch_lightning as pl

class LossMonitor(pl.Callback):
    def __init__(self, plot_every_n_epochs=1):
        super().__init__()
        self.plot_every_n_epochs = plot_every_n_epochs
        self.history = {
            'loss_total': [],
            'loss_gp': [],
            'loss_recon': [],
            'loss_kl_total': [],
            'weight_local_div': [] 
        }

    def on_train_epoch_end(self, trainer, pl_module):
        """Lightning automatically calls this at the end of every training epoch."""
        
        # Lightning stores all your logged metrics in this dictionary
        metrics = trainer.callback_metrics

        # Helper function to safely extract the value and convert from Tensor to float
        def get_metric(name):
            val = metrics.get(name, 0)
            return val.item() if hasattr(val, 'item') else val

        # Append the current epoch's metrics to our history
        self.history['loss_total'].append(get_metric('gp_warmup_loss_total'))
        self.history['loss_gp'].append(get_metric('gp_warmup_loss_gp'))
        self.history['loss_recon'].append(get_metric('gp_warmup_loss_recon'))
        self.history['loss_kl_total'].append(get_metric('gp_warmup_loss_kls'))
        self.history['weight_local_div'].append(get_metric('gp_warmup_weight_vae.dirichlet.local_divergence'))

        # Trigger the plot update
        epoch = trainer.current_epoch
        if (epoch + 1) % self.plot_every_n_epochs == 0:
            self.plot_metrics(epoch)

    def plot_metrics(self, epoch):
        # Clears the previous plot output for a smooth live-updating effect in notebooks
        clear_output(wait=True) 
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        epochs = range(1, len(self.history['loss_total']) + 1)

        # Panel 1: The Big Picture
        axes[0].plot(epochs, self.history['loss_total'], label='Total Loss', color='black', linewidth=2)
        axes[0].plot(epochs, self.history['loss_gp'], label='GP Loss', color='blue')
        axes[0].plot(epochs, self.history['loss_recon'], label='Recon Loss', color='green')
        axes[0].set_title(f'Primary Losses (Epoch {epoch+1})')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss Value')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Panel 2: Total KL Divergences
        axes[1].plot(epochs, self.history['loss_kl_total'], label='Total KLs', color='orange', linewidth=2)
        axes[1].set_title('KL Divergence')
        axes[1].set_xlabel('Epoch')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Panel 3: Warmup Weight Tracking
        axes[2].plot(epochs, self.history['weight_local_div'], label='Dirichlet Weight', color='purple', linestyle='--')
        axes[2].set_title('Warmup Annealing Schedule')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Weight Value')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
