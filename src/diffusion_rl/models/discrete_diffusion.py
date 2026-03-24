import math

import lightning as L
import torch
from einops import rearrange

from diffusion_rl.models.weighted_noise_scheduler import WeightedNoiseScheduler
from diffusion_rl.modules.transformer import TokenSequenceScoreTransformer


class TextGenerationCallback(L.Callback):
    """
    Logs one generated text sample to TensorBoard at the end of each validation epoch.

    Args:
        decode: callable mapping a list of int token ids to a string.
                E.g. ``tiktoken.get_encoding("gpt2").decode`` for GPT-2 BPE,
                or ``lambda ids: bytes(ids).decode("ascii", errors="replace")`` for ASCII.
    """

    def __init__(self, decode):
        super().__init__()
        self.decode = decode

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.logger is None:
            return
        seq_len = pl_module.hparams.max_seq_len
        dummy = torch.zeros(1, seq_len, dtype=torch.long, device=pl_module.device)
        pl_module.eval()
        with torch.no_grad():
            token_ids = pl_module.predict_step(dummy)
        valid_ids = [t for t in token_ids[0].tolist() if t < pl_module.mask_val]
        text = self.decode(valid_ids)
        trainer.logger.experiment.add_text(
            "generated_text", text, global_step=trainer.global_step
        )


def integrate_ctmc(x0, model, n_steps, mask_val):
    """
    A simple integrator for the basic masked diffusion LLM. Based off
    https://arxiv.org/pdf/2406.07524
    """
    bs, L = x0.shape
    x = x0
    alphas = reversed(torch.linspace(0, 1, n_steps))
    for alpha_s, alpha_t in zip(alphas, alphas[1:]):
        t = torch.full((bs,), alpha_t, device=x0.device)
        masked = x == mask_val
        logits = model(x, t)
        flat_logits = rearrange(logits, "b L d -> (b L) d")
        flat_probs = torch.nn.functional.softmax(flat_logits, dim=1)
        flat_samples = torch.multinomial(flat_probs, 1)
        samples = rearrange(flat_samples, "(b L) d -> b L d", b=bs).squeeze(-1)
        leave_masked = torch.rand(bs, L, device=x0.device) < (1 - alpha_s) / (
            1 - alpha_t
        )
        samples[leave_masked] = mask_val
        x[masked] = samples[masked]
    return x


class DiffusionLLM(L.LightningModule):
    """
    A simple masked diffusion LLM, based off https://arxiv.org/pdf/2406.07524
    """

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        num_inference_steps=100,
        weight_decay=1e-5,
        lr=3e-4,
        log_every_n_steps=1000,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )
        self.save_hyperparameters()
        self.model = TokenSequenceScoreTransformer(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
        )
        self.mask_val = self.hparams.vocab_size
        self.noise_scheduler = WeightedNoiseScheduler(
            num_bins=100, min_time=1e-3, max_time=1.0, prior=math.log(vocab_size)
        )

    def training_step(self, batch, step):
        input_ids = batch.long()
        bs, L = input_ids.shape
        # alpha = torch.arange(bs, device=input_ids.device).unsqueeze(-1) / bs
        # alpha += torch.rand(bs, 1, device=input_ids.device) / bs  # 1 - The current time
        # Sample noise levels using the weighted noise scheduler
        alpha, weights = self.noise_scheduler.sample(bs)
        # Reshape alpha to (bs, 1) for broadcasting
        alpha = alpha.unsqueeze(-1)
        masked_index = torch.rand(bs, L, device=input_ids.device) < 1 - alpha
        masked_ids = input_ids.clone()
        masked_ids[masked_index] = self.mask_val  # Mask it
        logits = self.model(masked_ids, alpha)
        # gathered_logits = logits.gather(2, input_ids.unsqueeze(-1)).squeeze(-1)
        cross_entropy = rearrange(
            torch.nn.functional.cross_entropy(
                rearrange(logits, "b L d -> (b L) d"),
                rearrange(input_ids, "b L -> (b L)"),
                reduction="none",
            ),
            "(b L) -> b L",
            b=bs,
        )
        loss_per_noise_level = (
            cross_entropy * masked_index.float() / (1 - alpha.clamp(min=1e-4))
        ).mean(dim=1)
        loss = (weights * loss_per_noise_level).mean()
        # Update the noise scheduler with the losses for the sampled noise levels
        self.noise_scheduler.update_bins(alpha.detach(), loss_per_noise_level.detach())

        self.log("train_loss", loss, prog_bar=True)
        current_step = self.global_step
        if (
            self.logger is not None
            and current_step % self.hparams.log_every_n_steps == 0
        ):
            self.noise_scheduler.log_histogram(self.logger.experiment, self.global_step)
        return loss

    def configure_optimizers(self):
        lr = self.hparams.lr
        weight_decay = self.hparams.weight_decay
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95)
        )
        return {
            "optimizer": optimizer,
        }

    def validation_step(self, batch, _):
        input_ids = batch.long()
        bs, L = input_ids.shape
        # Sample noise uniformly for validation (no weighting)
        alpha = torch.arange(bs, device=input_ids.device).unsqueeze(-1) / bs
        alpha += torch.rand(bs, 1, device=input_ids.device) / bs  # 1 - The current time
        masked_index = torch.rand(bs, L, device=input_ids.device) < 1 - alpha
        masked_ids = input_ids.clone()
        masked_ids[masked_index] = self.mask_val
        logits = self.model(masked_ids, alpha)
        cross_entropy = rearrange(
            torch.nn.functional.cross_entropy(
                rearrange(logits, "b L d -> (b L) d"),
                rearrange(input_ids, "b L -> (b L)"),
                reduction="none",
            ),
            "(b L) -> b L",
            b=bs,
        )
        loss = (
            (cross_entropy * masked_index.float() / (1 - alpha.clamp(min=1e-4))).mean(
                dim=1
            )
        ).mean()
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def predict_step(self, batch, num_steps=None):
        num_steps = self.hparams.num_inference_steps if num_steps is None else num_steps
        bs, L = batch.shape
        output = torch.full(
            (bs, L), self.mask_val, dtype=torch.long, device=batch.device
        )
        return integrate_ctmc(output, self.model, num_steps, self.mask_val)
