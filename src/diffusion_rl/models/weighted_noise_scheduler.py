import torch
import torch.nn as nn


class WeightedNoiseScheduler(nn.Module):
    """
    Samples noise to minimize the variance of the loss, as described in https://arxiv.org/abs/2303.00848
    Requires the loss to be positive (as is the case for cross-entropy and MSE).
    """

    def __init__(self, num_bins, min_time=0, max_time=1.0, ema_decay=0.99, prior=1.0):
        super().__init__()
        self.num_bins = num_bins
        self.max_time = max_time
        self.min_time = min_time
        self.ema_decay = ema_decay
        self.register_buffer(
            "bin_edges", torch.linspace(min_time, max_time, num_bins + 1)
        )
        self.register_buffer("mean_losses", torch.full((num_bins,), prior))

    def update_bins(self, noise_levels, losses):
        """
        Updates the mean losses for each bin using an exponential moving average.
        """
        with torch.no_grad():
            losses = losses.flatten()  # Ensure losses is a 1D tensor
            noise_levels = noise_levels.flatten()  # Ensure noise_levels is a 1D tensor
            # Find the bin for each noise level
            bin_indices = torch.bucketize(noise_levels, self.bin_edges) - 1
            bin_indices = bin_indices.clamp(0, self.num_bins - 1)
            self.mean_losses[bin_indices] = (
                self.ema_decay * self.mean_losses[bin_indices]
                + (1 - self.ema_decay) * losses
            )

    def sample(self, batch_size):
        """
        Samples noise levels according to the mean losses of each bin.
        """
        with torch.no_grad():
            # Compute sampling probabilities proportional to mean losses
            probabilities = self.mean_losses / self.mean_losses.sum()
            # Sample bins according to the computed probabilities
            sampled_bins = torch.multinomial(
                probabilities, batch_size, replacement=True
            )
            # Sample noise levels uniformly within the selected bins
            bin_start = self.bin_edges[sampled_bins]
            bin_end = self.bin_edges[sampled_bins + 1]
            samples = (
                torch.rand(batch_size, device=self.bin_edges.device)
                * (bin_end - bin_start)
                + bin_start
            )
            # Weight the samples inversely proportional to the mean loss of the sampled bins
            inv_losses = 1 / (self.mean_losses + 1e-8)
            weights = inv_losses[sampled_bins]  # Weights for the sampled noise levels
            weights = weights / weights.mean()  # Normalize weights to sum to 1
        return samples, weights

    def log_histogram(
        self, writer, global_step, tag="noise_scheduler/sampling_probabilities"
    ):
        """
        Logs a histogram of the current sampling probabilities (1/mean_losses, normalized)
        to a TensorBoard SummaryWriter.
        """
        with torch.no_grad():
            inv_losses = 1 / (self.mean_losses + 1e-8)
            probabilities = inv_losses / inv_losses.sum()
        COUNT = self.num_bins * 1000  # Total count for the histogram
        counts = (probabilities * COUNT).long().cpu().numpy()
        writer.add_histogram_raw(
            tag,
            min=self.bin_edges[0].item(),
            max=self.bin_edges[-1].item(),
            bucket_limits=self.bin_edges[1:],
            num=COUNT,
            sum=counts.sum().item(),
            sum_squares=(counts**2).sum().item(),
            bucket_counts=counts,
            global_step=global_step,
        )
