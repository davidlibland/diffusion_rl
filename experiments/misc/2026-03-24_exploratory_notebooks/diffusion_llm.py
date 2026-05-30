# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: diffusion_rl
#     language: python
#     name: python3
# ---

# %%
import lightning as L

# %%
import tiktoken

from diffusion_rl.data.wikitext import WikiText103DataModule

if __name__ == "__main__":
    dm = WikiText103DataModule(
        cache_dir="../wikitext103_cache", block_size=128, batch_size=32, ascii=False
    )
    # dm.prepare_data()
    # dm.setup("fit")

    # train_ds = dm.train_dataset
    # print(f"Train samples : {len(train_ds):,}")
    # print(f"Val samples   : {len(dm.val_dataset):,}")

    # x = train_ds[0]
    # print(f"x shape: {x.shape}, dtype: {x.dtype}")
    enc = tiktoken.get_encoding("gpt2")
    # print(enc.decode(x.tolist()))

    # loader = dm.train_dataloader()
    # xb = next(iter(loader))
    # print(f"Batch x: {xb.shape}")
    # print("max token value:", enc.max_token_value)
    # vocab_size = enc.max_token_value

    # %%
    # dm = WikiText103DataModule(
    #     cache_dir="../wikitext103_cache", block_size=128, batch_size=8, ascii=False
    # )
    # dm.prepare_data()
    # dm.setup("fit")

    # train_ds = dm.train_dataset
    # print(f"Train samples : {len(train_ds):,}")
    # print(f"Val samples   : {len(dm.val_dataset):,}")

    # x = train_ds[0]
    # print(f"x shape: {x.shape}, dtype: {x.dtype}")
    # print("".join(bytes(x.tolist()).decode("ascii")))

    # loader = dm.train_dataloader()
    # xb = next(iter(loader))
    # print(f"Batch x: {xb.shape}")

    # %%
    from diffusion_rl.models.discrete_diffusion import (
        DiffusionLLM,
        TextGenerationCallback,
    )

    model = DiffusionLLM(vocab_size=enc.max_token_value + 1, max_seq_len=128)

    # %%
    L.seed_everything(42)
    trainer = L.Trainer(
        max_time={"minutes": 10},
        val_check_interval={"minutes": 1},
        gradient_clip_algorithm="norm",
        gradient_clip_val=1,
        callbacks=[TextGenerationCallback(decode=enc.decode)],
    )
    trainer.fit(
        model,
        dm,
    )

    # %%
    import torch
    from einops import rearrange

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

    # %%
    x0 = torch.full((1, 128), enc.max_token_value)
    x = integrate_ctmc(x0, model.model, 100, enc.max_token_value)
    print(enc.decode(x[0].tolist()))
    # x.max()
    # print(bytes(x[0].tolist()).decode("ascii"))

    # %%
    # x0 = torch.full((1, 1024), 128)
    # x = integrate_ctmc(x0, model.model, 100, 128)
    # x.max()
    # print(bytes(x[0].tolist()).decode("ascii"))

    # %%
    model.hparams.vocab_size

# %%
