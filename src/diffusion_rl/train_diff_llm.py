"""Training script for the Discrete Diffusion LLM on WikiText-103."""

import datetime

import click
import lightning as L
import tiktoken
from lightning.pytorch.callbacks import ModelCheckpoint

from diffusion_rl.data.wikitext import WikiText103DataModule
from diffusion_rl.models.discrete_diffusion import DiffusionLLM, TextGenerationCallback


@click.command()
@click.option(
    "--cache-dir",
    default="wikitext103_cache",
    show_default=True,
    help="Directory for the WikiText-103 token cache.",
)
@click.option(
    "--block-size",
    default=128,
    show_default=True,
    help="Token sequence length.",
)
@click.option(
    "--batch-size",
    default=32,
    show_default=True,
    help="Training batch size.",
)
@click.option(
    "--ascii",
    "use_ascii",
    is_flag=True,
    default=False,
    help="Use ASCII-only tokenization instead of GPT-2 BPE.",
)
@click.option(
    "--hidden-dim",
    default=256,
    show_default=True,
    help="Transformer hidden dimension.",
)
@click.option(
    "--num-layers",
    default=6,
    show_default=True,
    help="Number of transformer layers.",
)
@click.option(
    "--num-heads",
    default=8,
    show_default=True,
    help="Number of attention heads.",
)
@click.option(
    "--lr",
    default=3e-4,
    show_default=True,
    help="AdamW learning rate.",
)
@click.option(
    "--weight-decay",
    default=1e-5,
    show_default=True,
    help="AdamW weight decay.",
)
@click.option(
    "--max-minutes",
    default=10,
    show_default=True,
    help="Maximum training time in minutes.",
)
@click.option(
    "--checkpoint-dir",
    default="checkpoints",
    show_default=True,
    help="Directory for saving checkpoints.",
)
@click.option(
    "--checkpoint-every-minutes",
    default=5,
    show_default=True,
    help="Save a checkpoint every N minutes.",
)
@click.option(
    "--resume-from",
    default=None,
    type=click.Path(exists=True),
    help="Path to a checkpoint file to resume training from.",
)
@click.option(
    "--val-check-minutes",
    default=5,
    show_default=True,
    help="Run validation every N training minutes.",
)
@click.option(
    "--seed",
    default=42,
    show_default=True,
    help="Random seed.",
)
def train_diff_llm(
    cache_dir,
    block_size,
    batch_size,
    use_ascii,
    hidden_dim,
    num_layers,
    num_heads,
    lr,
    weight_decay,
    max_minutes,
    checkpoint_dir,
    checkpoint_every_minutes,
    resume_from,
    val_check_minutes,
    seed,
):
    """Train a discrete diffusion LLM on WikiText-103."""
    if use_ascii:
        vocab_size = 128
        decode = lambda ids: bytes(ids).decode("ascii", errors="replace")
    else:
        enc = tiktoken.get_encoding("gpt2")
        vocab_size = enc.max_token_value + 1
        decode = enc.decode

    L.seed_everything(seed)

    dm = WikiText103DataModule(
        cache_dir=cache_dir,
        block_size=block_size,
        batch_size=batch_size,
        ascii=use_ascii,
    )

    model = DiffusionLLM(
        vocab_size=vocab_size,
        max_seq_len=block_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        lr=lr,
        weight_decay=weight_decay,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="diffusion_llm-{step}",
        save_last=True,
        train_time_interval=datetime.timedelta(minutes=checkpoint_every_minutes),
    )

    trainer = L.Trainer(
        max_time={"minutes": max_minutes},
        val_check_interval={"minutes": val_check_minutes},
        gradient_clip_algorithm="norm",
        gradient_clip_val=1,
        callbacks=[
            TextGenerationCallback(decode=decode),
            checkpoint_callback,
        ],
    )

    trainer.fit(model, dm, ckpt_path=resume_from)


if __name__ == "__main__":
    train_diff_llm()
