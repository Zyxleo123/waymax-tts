from __future__ import annotations

import math
from typing import Sequence, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from model.diffusion.modules.attention import CrossAttention


def sinusoidal_timestep_embedding(t: jnp.ndarray, dim: int, max_period: int = 10_000) -> jnp.ndarray:
    t = jnp.asarray(t)
    if not jnp.issubdtype(t.dtype, jnp.floating):
        t = t.astype(jnp.float32)
    half = dim // 2
    freqs = jnp.exp(-math.log(max_period) * jnp.arange(0, half, dtype=t.dtype) / half)
    args = t[:, None] * freqs[None, :]
    emb = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
    if dim % 2 == 1:
        emb = jnp.concatenate([emb, jnp.zeros_like(emb[:, :1])], axis=-1)
    return emb


class Conv1dBlock(nnx.Module):
    def __init__(self, in_ch: int, out_ch: int, rngs: nnx.Rngs, kernel: int = 3, groups: int = 8):
        self.conv = nnx.Conv(
            in_features=in_ch,
            out_features=out_ch,
            kernel_size=(kernel,),
            padding="SAME",
            rngs=rngs,
        )
        self.gn = nnx.GroupNorm(num_features=out_ch, num_groups=min(groups, out_ch), epsilon=1e-5, rngs=rngs)

    def __call__(self, x_btc: jnp.ndarray) -> jnp.ndarray:
        h_btc = self.conv(x_btc)
        h_btc = self.gn(h_btc)
        return nnx.silu(h_btc)


class Identity1d(nnx.Module):
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return x


class ResBlock1d(nnx.Module):
    def __init__(self, in_ch: int, out_ch: int, time_ch: int, cond_ch: int, hidden_dim: int, rngs: nnx.Rngs, groups: int = 8):
        self.time = nnx.Conv(in_features=time_ch, out_features=in_ch, kernel_size=(1,), padding="SAME", rngs=rngs)
        self.block1 = Conv1dBlock(in_ch, out_ch, rngs=rngs, kernel=3, groups=groups)
        self.block2 = Conv1dBlock(out_ch, out_ch, rngs=rngs, kernel=3, groups=groups)
        self.skip = Identity1d() if in_ch == out_ch else nnx.Conv(in_features=in_ch, out_features=out_ch, kernel_size=(1,), padding="SAME", rngs=rngs)
        # Cross-attention for condition integration
        self.cross_attn = CrossAttention(query_dim=out_ch, context_dim=cond_ch, num_heads=8, hidden_dim=hidden_dim, rngs=rngs)

    def __call__(self, x_btc: jnp.ndarray, time_feat_b1c: jnp.ndarray, context_bnc: jnp.ndarray) -> jnp.ndarray:
        """Apply residual block with cross-attention conditioning.
        
        Args:
            x_btc: [B, T, C_in] input features
            time_feat_b1c: [B, 1, C_time] time embedding
            context_bnc: [B, n_tokens, C_cond] condition tokens
        
        Returns:
            [B, T, C_out] output features
        """
        time_b1c = self.time(time_feat_b1c)
        h = x_btc + time_b1c
        h = self.block1(h)
        h = self.block2(h)
        # Apply cross-attention with condition tokens
        # Provide an explicit boolean mask for condition tokens (assume all present)
        context_mask_bn = jnp.ones((context_bnc.shape[0], context_bnc.shape[1]), dtype=bool)
        h = h + self.cross_attn(h, context_bnc, context_mask_bn)

        if isinstance(self.skip, Identity1d):
            return h + x_btc
        skip = self.skip(x_btc)
        return h + skip


class Downsample1d(nnx.Module):
    def __init__(self, ch: int, rngs: nnx.Rngs):
        self.conv = nnx.Conv(in_features=ch, out_features=ch, kernel_size=(4,), strides=(2,), padding=((1, 1),), rngs=rngs)

    def __call__(self, x_btc: jnp.ndarray) -> jnp.ndarray:
        return self.conv(x_btc)


class Upsample1d(nnx.Module):
    def __init__(self, ch: int, rngs: nnx.Rngs):
        self.conv = nnx.Conv(in_features=ch, out_features=ch, kernel_size=(3,), padding="SAME", rngs=rngs)

    def __call__(self, x_btc: jnp.ndarray) -> jnp.ndarray:
        x_btc = jax.image.resize(x_btc, shape=(x_btc.shape[0], x_btc.shape[1] * 2, x_btc.shape[2]), method="nearest")
        return self.conv(x_btc)


class Projector(nnx.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, rngs: nnx.Rngs):
        self.fc1 = nnx.Linear(in_dim, hidden_dim, rngs=rngs)
        self.fc2 = nnx.Linear(hidden_dim, out_dim, rngs=rngs)

    def __call__(self, in_feat: jnp.ndarray) -> jnp.ndarray:
        h = self.fc1(in_feat)
        h = nnx.silu(h)
        return self.fc2(h)
    

class FiLMBlock(nnx.Module):
    def __init__(self, in_ch: int, cond_dim: int, hidden_dim: int, rngs: nnx.Rngs):
        self.fc1 = nnx.Linear(cond_dim, hidden_dim, rngs=rngs)
        self.fc2 = nnx.Linear(hidden_dim, 2 * in_ch, rngs=rngs)

    def __call__(self, x_btc: jnp.ndarray, cond_bc: jnp.ndarray, cond_mask: jnp.ndarray) -> jnp.ndarray:
        """Apply FiLM conditioning.
        
        Args:
            x_btc: [B, T, C] input features
            cond_bc: [B, C_cond] condition vectors
            cond_mask: [B] condition masks
        
        Returns:
            [B, T, C] modulated features
        """
        # Pool condition tokens to get a single vector per batch item
        gamma_beta = self.fc1(cond_bc)  # [B, hidden_dim]
        gamma_beta = nnx.silu(gamma_beta)
        
        gamma_beta = self.fc2(gamma_beta)  # [B, 2*in_ch]
        # After fc2: split to gamma/beta
        gamma, beta = jnp.split(gamma_beta, 2, axis=-1)  # each [B, C]
       
        return x_btc * (1 + gamma[:, None, :] * cond_mask[:, None, None]) + beta[:, None, :] * cond_mask[:, None, None]


class UNet1DConditioned(nnx.Module):
    def __init__(
        self,
        in_ch: int,
        rngs: nnx.Rngs,
        base_ch: int = 128,
        ch_mult: Tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int = 2,
        cond_dim: int = 256,
        cond2_dim: int = 256,
        time_emb_dim: int = 128,
        hidden_dim: int = 512,
        groups: int = 8,
        horizon: int = 25,
    ):
        self.in_ch = in_ch
        self.time_emb_dim = time_emb_dim
        self.num_res_blocks = num_res_blocks
        self.n_down = len(ch_mult)
        self.horizon = horizon

        # Condition projection: project tokens to base_ch dimension for cross-attention
        self.cond_proj = nnx.Linear(cond_dim, base_ch, rngs=rngs)
        self.time_proj = Projector(time_emb_dim, base_ch, hidden_dim=hidden_dim, rngs=rngs)
        self.stem = nnx.Conv(in_features=in_ch, out_features=base_ch, kernel_size=(3,), padding="SAME", rngs=rngs)

        ch = base_ch
        down_blocks = []
        downsamples = []
        skip_channels = []
        down_film_blocks = []

        for i, m in enumerate(ch_mult):
            out_ch = base_ch * m
            for _ in range(num_res_blocks):
                down_blocks.append(ResBlock1d(ch, out_ch, base_ch, base_ch, hidden_dim=hidden_dim, rngs=rngs, groups=groups))
                down_film_blocks.append(FiLMBlock(out_ch, cond2_dim, hidden_dim=hidden_dim, rngs=rngs))
                ch = out_ch
            skip_channels.append(ch)
            downsamples.append(Downsample1d(ch, rngs=rngs) if i != len(ch_mult) - 1 else Identity1d())\
            
        mid_film_blocks = []
        self.mid_block1 = ResBlock1d(ch, ch, base_ch, base_ch, hidden_dim=hidden_dim, rngs=rngs, groups=groups)
        mid_film_blocks.append(FiLMBlock(ch, cond2_dim, hidden_dim=hidden_dim, rngs=rngs))
        self.mid_block2 = ResBlock1d(ch, ch, base_ch, base_ch, hidden_dim=hidden_dim, rngs=rngs, groups=groups)
        mid_film_blocks.append(FiLMBlock(ch, cond2_dim, hidden_dim=hidden_dim, rngs=rngs))

        up_blocks = []
        upsamples = []
        up_film_blocks = []
        for i, m in reversed(list(enumerate(ch_mult))):
            out_ch = base_ch * m
            for _ in range(num_res_blocks):
                up_blocks.append(ResBlock1d(ch + skip_channels[i], out_ch, base_ch, base_ch, hidden_dim=hidden_dim, rngs=rngs, groups=groups))
                up_film_blocks.append(FiLMBlock(out_ch, cond2_dim, hidden_dim=hidden_dim, rngs=rngs))
                ch = out_ch
            upsamples.append(Upsample1d(ch, rngs=rngs) if i != 0 else Identity1d())

        # Register containers as NNX modules in a single assignment.
        self.down_blocks = nnx.List(down_blocks)
        self.downsamples = nnx.List(downsamples)
        self.skip_channels = tuple(skip_channels)
        self.up_blocks = nnx.List(up_blocks)
        self.upsamples = nnx.List(upsamples)
        self.down_film_blocks = nnx.List(down_film_blocks)
        self.mid_film_blocks = nnx.List(mid_film_blocks)
        self.up_film_blocks = nnx.List(up_film_blocks)
        self.out_norm = nnx.GroupNorm(num_features=ch, num_groups=min(groups, ch), epsilon=1e-5, rngs=rngs)
        self.out_conv = nnx.Conv(in_features=ch, out_features=in_ch, kernel_size=(3,), padding="SAME", rngs=rngs)

    def __call__(self, x: jnp.ndarray, t: jnp.ndarray, c1: jnp.ndarray, c2: jnp.ndarray, c2_mask: jnp.ndarray) -> jnp.ndarray:
        """Forward pass with cross-attention conditioning.
        
        Args:
            x: [B, T, 5] input trajectory
            t: [B] timestep indices
            c1: [B, n_cond_tokens, cond_dim] condition tokens for cross-attention
            c2: [B, cond_dim] second condition tokens
            c2_mask: [B] mask for second condition tokens

        Returns:
            [B, T, 5] denoised trajectory
        """
        divisor = 2 ** (self.n_down - 1)
        if self.horizon % divisor != 0:
            target_t = ((self.horizon // divisor) + 1) * divisor
            x = jnp.pad(x, ((0, 0), (0, 0), (0, target_t - self.horizon)), mode="constant")

        t_emb = sinusoidal_timestep_embedding(t, self.time_emb_dim)
        # Project condition tokens: [B, n_tokens, cond_dim] -> [B, n_tokens, base_ch]
        c1 = self.cond_proj(c1)  # [B, n_tokens, base_ch]
        time_feat = self.time_proj(t_emb)[:, None, :]

        h = self.stem(jnp.swapaxes(x, 1, 2))

        skips = []
        rb = 0
        for i in range(self.n_down):
            for _ in range(self.num_res_blocks):
                h = self.down_blocks[rb](h, time_feat, c1)
                h = self.down_film_blocks[rb](h, c2, c2_mask)
                rb += 1
            skips.append(h)
            h = self.downsamples[i](h)

        h = self.mid_block1(h, time_feat, c1)
        h = self.mid_film_blocks[0](h, c2, c2_mask)
        h = self.mid_block2(h, time_feat, c1)
        h = self.mid_film_blocks[1](h, c2, c2_mask)
        
        rb_up = 0
        for i in range(self.n_down):
            skip = skips.pop()
            if h.shape[1] != skip.shape[1]:
                min_len = min(h.shape[1], skip.shape[1])
                h = h[:, :min_len, :]
                skip = skip[:, :min_len, :]

            for _ in range(self.num_res_blocks):
                h = jnp.concatenate([h, skip], axis=-1)
                h = self.up_blocks[rb_up](h, time_feat, c1)
                h = self.up_film_blocks[rb_up](h, c2, c2_mask)
                rb_up += 1

            h = self.upsamples[i](h)

        h = self.out_norm(h)
        h = nnx.silu(h)
        h = self.out_conv(h)
        return jnp.swapaxes(h[:, :self.horizon, :], 1, 2)


def make_beta_schedule(timesteps: int, schedule: str = "cosine") -> jnp.ndarray:
    if schedule == "linear":
        return jnp.linspace(1e-4, 2e-2, timesteps, dtype=jnp.float32)
    if schedule == "cosine":
        s = 0.008
        steps = timesteps + 1
        x = jnp.linspace(0, timesteps, steps, dtype=jnp.float32)
        alphas_cumprod = jnp.cos(((x / timesteps) + s) / (1 + s) * jnp.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return jnp.clip(betas, 0.0, 0.999).astype(jnp.float32)
    raise ValueError(f"Unknown schedule: {schedule}")


class GaussianDiffusion(nnx.Module):
    def __init__(
        self,
        denoise_fn: nnx.Module,
        timesteps: int = 100,
        beta_schedule: str = "cosine",
        predict_type: str = "v",
        clip_denoised: bool = False,
    ):
        self.denoise_fn = denoise_fn
        self.timesteps = int(timesteps)
        self.predict_type = predict_type
        self.clip_denoised = clip_denoised

        betas = make_beta_schedule(self.timesteps, beta_schedule)
        alphas = 1.0 - betas
        alphas_cumprod = jnp.cumprod(alphas, axis=0)
        alphas_cumprod_prev = jnp.concatenate([jnp.array([1.0], dtype=alphas.dtype), alphas_cumprod[:-1]], axis=0)

        self.betas = nnx.Variable(betas)
        self.alphas = nnx.Variable(alphas)
        self.alphas_cumprod = nnx.Variable(alphas_cumprod)
        self.alphas_cumprod_prev = nnx.Variable(alphas_cumprod_prev)
        self.sqrt_alphas_cumprod = nnx.Variable(jnp.sqrt(alphas_cumprod))
        self.sqrt_one_minus_alphas_cumprod = nnx.Variable(jnp.sqrt(1.0 - alphas_cumprod))

        posterior_var = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_variance = jnp.maximum(posterior_var, 1e-20)
        self.posterior_variance = nnx.Variable(posterior_variance)
        self.posterior_log_variance_clipped = nnx.Variable(jnp.log(posterior_variance))

        self.posterior_mean_coef1 = nnx.Variable(betas * jnp.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.posterior_mean_coef2 = nnx.Variable((1.0 - alphas_cumprod_prev) * jnp.sqrt(alphas) / (1.0 - alphas_cumprod))

    @staticmethod
    def _extract(a: jnp.ndarray, t: jnp.ndarray, x_shape: Sequence[int]) -> jnp.ndarray:
        out = a[t]
        return out.reshape((x_shape[0], 1, 1))

    def q_sample(self, x0: jnp.ndarray, t: jnp.ndarray, noise: jnp.ndarray) -> jnp.ndarray:
        sqrt_ab = self._extract(self.sqrt_alphas_cumprod.value, t, x0.shape)
        sqrt_1mab = self._extract(self.sqrt_one_minus_alphas_cumprod.value, t, x0.shape)
        return sqrt_ab * x0 + sqrt_1mab * noise

    def predict_x0_from_eps(self, x_t: jnp.ndarray, t: jnp.ndarray, eps: jnp.ndarray) -> jnp.ndarray:
        sqrt_ab = self._extract(self.sqrt_alphas_cumprod.value, t, x_t.shape)
        sqrt_1mab = self._extract(self.sqrt_one_minus_alphas_cumprod.value, t, x_t.shape)
        return (x_t - sqrt_1mab * eps) / sqrt_ab

    def p_mean_variance(self, x_t: jnp.ndarray, t: jnp.ndarray, cond1: jnp.ndarray, cond2: jnp.ndarray, cond2_mask: jnp.ndarray, denoise_fn=None) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        # ``denoise_fn`` lets a caller substitute a different denoiser (e.g. the
        # DPPO-trainable copy for the late steps) without changing default
        # inference, which keeps using ``self.denoise_fn``.
        fn = self.denoise_fn if denoise_fn is None else denoise_fn
        pred = fn(x_t, t, cond1, cond2, cond2_mask)

        if self.predict_type == "eps":
            x0_pred = self.predict_x0_from_eps(x_t, t, pred)
        elif self.predict_type == "mu":
            x0_pred = pred
        elif self.predict_type == "v":
            sqrt_ab = self._extract(self.sqrt_alphas_cumprod.value, t, x_t.shape)
            sqrt_1mab = self._extract(self.sqrt_one_minus_alphas_cumprod.value, t, x_t.shape)
            x0_pred = sqrt_ab * x_t - sqrt_1mab * pred
        else:
            raise ValueError(f"Unsupported predict_type: {self.predict_type}")

        if self.clip_denoised:
            x0_pred = jnp.clip(x0_pred, -1.0, 1.0)

        mean = self._extract(self.posterior_mean_coef1.value, t, x_t.shape) * x0_pred + self._extract(self.posterior_mean_coef2.value, t, x_t.shape) * x_t
        var = self._extract(self.posterior_variance.value, t, x_t.shape)
        return mean, var, x0_pred

    @staticmethod
    def _temp_schedule(sampling_temp: float, temp_mode: str, num_steps: int) -> jnp.ndarray:
        """Precompute per-timestep effective temperature.

        Returns array of shape [num_steps] indexed by diffusion loop step index
        (0 = highest noise timestep, num_steps-1 = lowest).
        """
        t_idx = jnp.arange(num_steps, dtype=jnp.float32)
        denom = jnp.maximum(num_steps - 1, 1)  # avoid /0 when num_steps==1
        if temp_mode == "early":
            # Full temp at high-noise (step 0 = high t), taper to 1.0 at low-noise
            return 1.0 + (sampling_temp - 1.0) * (1.0 - t_idx / denom)
        elif temp_mode == "late":
            # 1.0 at high-noise, full temp at low-noise (step num_steps-1)
            return 1.0 + (sampling_temp - 1.0) * (t_idx / denom)
        else:  # "uniform"
            return jnp.full((num_steps,), sampling_temp, dtype=jnp.float32)

    def _ddim_update(self, x_t: jnp.ndarray, t: jnp.ndarray, x0_pred: jnp.ndarray,
                     noise: jnp.ndarray, eta: float = 1.0, sampling_temp: float = 1.0) -> jnp.ndarray:
        """The eta-DDIM reverse step given an already-predicted ``x0``.

        Factored out of :meth:`p_sample` so a caller that has already run the
        denoiser once (e.g. :meth:`sample_with_trace`, which also needs the
        posterior stats) can reuse the single prediction instead of invoking the
        network twice -- the network is not purely functional across calls (nnx
        rng streams), so a second call would perturb the trajectory.
        """
        # Predicted noise direction
        sqrt_ab = self._extract(self.sqrt_alphas_cumprod.value, t, x_t.shape)
        sqrt_1mab = self._extract(self.sqrt_one_minus_alphas_cumprod.value, t, x_t.shape)
        eps_pred = (x_t - sqrt_ab * x0_pred) / sqrt_1mab

        # Scale noise direction by sampling temperature
        eps_pred = eps_pred * sampling_temp
        x0_pred = (x_t - sqrt_1mab * eps_pred) / sqrt_ab

        # DDIM variance: σ_t = η * sqrt(posterior_variance_t)
        post_var = self._extract(self.posterior_variance.value, t, x_t.shape)
        sigma = eta * jnp.sqrt(post_var)

        # Direction coefficient: sqrt(max(1 - ᾱ_{t-1} - σ², 0))
        abar_prev = self._extract(self.alphas_cumprod_prev.value, t, x_t.shape)
        dir_coeff = jnp.sqrt(jnp.maximum(1.0 - abar_prev - sigma ** 2, 0.0))

        # DDIM update: sqrt(ᾱ_{t-1}) * x̂₀ + dir_coeff * ε_pred + σ * noise
        nonzero_mask = (t != 0).astype(x_t.dtype).reshape((-1, 1, 1))
        return jnp.sqrt(abar_prev) * x0_pred + dir_coeff * eps_pred + nonzero_mask * sigma * noise

    def p_sample(self, x_t: jnp.ndarray, t: jnp.ndarray, cond1: jnp.ndarray, cond2: jnp.ndarray, cond2_mask: jnp.ndarray,
                 noise: jnp.ndarray, eta: float = 1.0, sampling_temp: float = 1.0, denoise_fn=None) -> jnp.ndarray:
        _, _, x0_pred = self.p_mean_variance(x_t, t, cond1, cond2, cond2_mask, denoise_fn=denoise_fn)
        return self._ddim_update(x_t, t, x0_pred, noise, eta=eta, sampling_temp=sampling_temp)

    def _sample_impl(self, shape: Tuple[int, int, int], cond1: jnp.ndarray, cond2: jnp.ndarray, cond2_mask: jnp.ndarray, 
                     rng: jax.Array, eta: float = 1.0, sampling_temp: float = 1.0, temp_mode: str = "uniform") -> jnp.ndarray:
        batch_size = shape[0]
        key_x, key_steps = jax.random.split(rng)
        x_init = jax.random.normal(key_x, shape, dtype=jnp.float32)
        num_steps = self.timesteps
        temp_sched = self._temp_schedule(sampling_temp, temp_mode, num_steps)

        def body(i: int, carry: tuple[jnp.ndarray, jax.Array]) -> tuple[jnp.ndarray, jax.Array]:
            x_curr, key = carry
            timestep = num_steps - 1 - i
            t = jnp.full((batch_size,), timestep, dtype=jnp.int32)
            key, step_key = jax.random.split(key)
            noise = jax.random.normal(step_key, shape, dtype=jnp.float32)
            eff_temp = temp_sched[i]
            return self.p_sample(x_curr, t, cond1, cond2, cond2_mask, noise, eta=eta, sampling_temp=eff_temp), key

        x_final, _ = jax.lax.fori_loop(0, num_steps, body, (x_init, key_steps))
        return x_final

    def _compute_loss_per_example(self, x0: jnp.ndarray, cond1: jnp.ndarray, cond2: jnp.ndarray, cond2_mask: jnp.ndarray, noise: jnp.ndarray, t: jnp.ndarray, denoise_fn=None) -> jnp.ndarray:
        """Per-example diffusion loss, averaged over channels and time but NOT
        over the batch. Returns ``[B]``. ``_compute_loss`` is its batch-mean."""
        fn = self.denoise_fn if denoise_fn is None else denoise_fn
        x_t = self.q_sample(x0, t, noise=noise)
        pred = fn(x_t, t, cond1, cond2, cond2_mask)

        if self.predict_type == "eps":
            target = noise
        elif self.predict_type == "mu":
            target = x0
        elif self.predict_type == "v":
            sqrt_ab = self._extract(self.sqrt_alphas_cumprod.value, t, x0.shape)
            sqrt_1mab = self._extract(self.sqrt_one_minus_alphas_cumprod.value, t, x0.shape)
            target = sqrt_ab * noise - sqrt_1mab * x0
        else:
            raise ValueError(f"Unsupported predict_type: {self.predict_type}")

        se = (pred - target) ** 2
        return jnp.mean(se, axis=tuple(range(1, se.ndim)))  # mean over C, T -> [B]

    def _compute_loss(self, x0: jnp.ndarray, cond1: jnp.ndarray, cond2: jnp.ndarray, cond2_mask: jnp.ndarray, noise: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        return jnp.mean(self._compute_loss_per_example(x0, cond1, cond2, cond2_mask, noise, t))

    def _loss_impl(self, x0: jnp.ndarray, cond1: jnp.ndarray, cond2: jnp.ndarray, cond2_mask: jnp.ndarray, rng: jax.Array, per_example: bool = False, denoise_fn=None) -> jnp.ndarray:
        batch = x0.shape[0]
        key_t, key_noise = jax.random.split(rng)
        t = jax.random.randint(key_t, (batch,), 0, self.timesteps, dtype=jnp.int32)
        noise = jax.random.normal(key_noise, x0.shape, dtype=jnp.float32)
        per = self._compute_loss_per_example(x0, cond1, cond2, cond2_mask, noise, t, denoise_fn=denoise_fn)
        return per if per_example else jnp.mean(per)

    def _resample_impl(self, proposals: jnp.ndarray, cond1: jnp.ndarray, cond2: jnp.ndarray, cond2_mask: jnp.ndarray, n_timesteps: int, rng: jax.Array, noise_scale: float = 1.0, eta: float = 1.0, sampling_temp: float = 1.0, temp_mode: str = "uniform") -> jnp.ndarray:
        batch_size = proposals.shape[0]
        key_q, key_steps = jax.random.split(rng)
        q_noise = jax.random.normal(key_q, proposals.shape, dtype=jnp.float32) * noise_scale
        t0 = jnp.full((batch_size,), n_timesteps - 1, dtype=jnp.int32)
        x = self.q_sample(proposals, t0, noise=q_noise)
        temp_sched = self._temp_schedule(sampling_temp, temp_mode, n_timesteps)

        def body(i: int, carry: tuple[jnp.ndarray, jax.Array]) -> tuple[jnp.ndarray, jax.Array]:
            x_curr, key = carry
            timestep = n_timesteps - 1 - i
            t = jnp.full((batch_size,), timestep, dtype=jnp.int32)
            key, step_key = jax.random.split(key)
            noise = jax.random.normal(step_key, proposals.shape, dtype=jnp.float32)
            eff_temp = temp_sched[i]
            return self.p_sample(x_curr, t, cond1, cond2, cond2_mask, noise, eta=eta, sampling_temp=eff_temp), key

        x, _ = jax.lax.fori_loop(0, n_timesteps, body, (x, key_steps))
        return x

    def sample(
        self,
        shape: Tuple[int, int, int],
        cond1: jnp.ndarray,
        cond2: jnp.ndarray,
        cond2_mask: jnp.ndarray,
        *,
        rng: jax.Array,
        eta: float = 1.0,
        sampling_temp: float = 1.0,
        temp_mode: str = "uniform",
    ) -> jnp.ndarray:
        return self._sample_impl(shape, cond1, cond2, cond2_mask, rng, eta=eta, sampling_temp=sampling_temp, temp_mode=temp_mode)

    def loss(
        self,
        x0: jnp.ndarray,
        cond1: jnp.ndarray,
        cond2: jnp.ndarray,
        cond2_mask: jnp.ndarray,
        *,
        rng: jax.Array,
    ) -> jnp.ndarray:
        return self._loss_impl(x0, cond1, cond2, cond2_mask, rng)

    def loss_per_example(
        self,
        x0: jnp.ndarray,
        cond1: jnp.ndarray,
        cond2: jnp.ndarray,
        cond2_mask: jnp.ndarray,
        *,
        rng: jax.Array,
        denoise_fn=None,
    ) -> jnp.ndarray:
        """Diffusion loss per batch element, ``[B]`` (mean over channels + time).

        ``loss(...) == loss_per_example(...).mean()`` for the same rng, so this is
        a drop-in for reward-weighted / filtered fine-tuning where each example
        carries a different weight. ``denoise_fn`` substitutes the denoiser (e.g.
        the DPPO-trainable copy for an expert-anchor minibatch).
        """
        return self._loss_impl(x0, cond1, cond2, cond2_mask, rng, per_example=True, denoise_fn=denoise_fn)

    # ----------------------------------------------------------------------- #
    # RL / DPPO support: reverse-transition statistics, log-probabilities, and
    # a sampler that records the last K' denoising transitions.
    # ----------------------------------------------------------------------- #
    _LOG_2PI = float(math.log(2.0 * math.pi))

    def _posterior_mean_std(
        self, x_t: jnp.ndarray, t: jnp.ndarray, cond1: jnp.ndarray, cond2: jnp.ndarray,
        cond2_mask: jnp.ndarray, denoise_fn=None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """DDPM reverse-transition mean and std at step ``t``.

        Returns ``(mean, std, x0_pred)`` where ``x_{t-1} ~ N(mean, std**2 I)``.
        The mean is the posterior mean ``coef1*x0 + coef2*x_t`` (independent of any
        exploration-noise floor), and ``std = sqrt(posterior_variance_t)`` (0 at
        ``t=0``; callers floor it for likelihoods). Equivalent to the eta=1 DDIM
        update the default sampler uses, so likelihoods are consistent with it.
        """
        mean, var, x0_pred = self.p_mean_variance(x_t, t, cond1, cond2, cond2_mask, denoise_fn=denoise_fn)
        std = jnp.sqrt(jnp.maximum(var, 0.0))
        return mean, std, x0_pred

    @staticmethod
    def _gaussian_logdensity(x: jnp.ndarray, mean: jnp.ndarray, std: jnp.ndarray) -> jnp.ndarray:
        """Elementwise diagonal-Gaussian log density (same shape as ``x``)."""
        return -0.5 * ((x - mean) / std) ** 2 - jnp.log(std) - 0.5 * GaussianDiffusion._LOG_2PI

    @staticmethod
    def _reduce_logdensity(ld: jnp.ndarray, prefix_len: int | None) -> jnp.ndarray:
        """Average a per-element log density over channels and (executed) time.

        ``ld`` is ``[B, C, T]``. We average over channels and time rather than
        summing all C*T dims, so the PPO ratio stays a sane per-transition scalar
        (DPPO note: "do not sum all 125 dimensions into one enormous ratio").
        With ``prefix_len`` set, only the first ``prefix_len`` time points (the
        executed prefix that receives policy-gradient credit) are averaged.
        """
        c = ld.shape[1]
        t_dim = ld.shape[2]
        if prefix_len is None:
            return jnp.mean(ld, axis=(1, 2))
        pl = int(prefix_len)
        mask_t = (jnp.arange(t_dim) < pl).astype(ld.dtype)  # [T]
        summed = jnp.sum(ld * mask_t[None, None, :], axis=(1, 2))
        return summed / (c * pl)

    def transition_log_prob(
        self,
        x_t: jnp.ndarray,
        x_prev: jnp.ndarray,
        t: jnp.ndarray,
        cond1: jnp.ndarray,
        cond2: jnp.ndarray,
        cond2_mask: jnp.ndarray,
        *,
        sigma_floor: float = 0.1,
        prefix_len: int | None = None,
        denoise_fn=None,
    ) -> jnp.ndarray:
        """Log-prob of the reverse transition ``x_t -> x_prev`` at step ``t``.

        Converts the current prediction to ``x0`` and then to the posterior mean
        (see :meth:`_posterior_mean_std`), floors the std at ``sigma_floor`` for a
        stable likelihood, and averages the Gaussian log density over channels and
        the executed time prefix. Returns ``[B]``.
        """
        mean, std, _ = self._posterior_mean_std(x_t, t, cond1, cond2, cond2_mask, denoise_fn=denoise_fn)
        std_lp = jnp.maximum(std, sigma_floor)
        ld = self._gaussian_logdensity(x_prev, mean, std_lp)
        return self._reduce_logdensity(ld, prefix_len)

    def sample_with_trace(
        self,
        shape: Tuple[int, int, int],
        cond1: jnp.ndarray,
        cond2: jnp.ndarray,
        cond2_mask: jnp.ndarray,
        *,
        rng: jax.Array,
        k_trainable: int,
        trainable_denoise_fn=None,
        rl_mode: bool = False,
        sigma_sample_floor: float = 0.0,
        sigma_logprob_floor: float = 0.1,
        prefix_len: int | None = None,
        eta: float = 1.0,
        sampling_temp: float = 1.0,
        temp_mode: str = "uniform",
    ):
        """Sample and record the last ``k_trainable`` denoising transitions.

        The early ``timesteps - k_trainable`` steps run in a compact ``fori_loop``
        with the frozen ``self.denoise_fn`` (the default sampler math). The final
        ``k_trainable`` steps (diffusion timesteps ``k_trainable-1 .. 0``) are
        unrolled so each transition is recorded, using ``trainable_denoise_fn`` if
        given (the DPPO-trainable copy) else ``self.denoise_fn``.

        With ``rl_mode=False`` and default settings the produced trajectory is
        bit-identical to :meth:`sample` under the same rng (Gate 2). With
        ``rl_mode=True`` the late steps are stochastic Gaussian transitions with an
        exploration-std floor -- including at ``t=0``, where the default sampler
        suppresses noise -- matching DPPO's training sampler.

        Returns ``(trajectory, trace)`` where ``trace`` is a dict of arrays stacked
        over the ``k_trainable`` steps (leading axis K, ordered high-t -> t=0):
        ``x_t, x_prev, timestep, mean, std, log_prob``.
        """
        batch_size = shape[0]
        num_steps = self.timesteps
        k = int(k_trainable)
        if k < 0 or k > num_steps:
            raise ValueError(f"k_trainable must be in [0, {num_steps}], got {k}.")
        n_early = num_steps - k

        key_x, key_steps = jax.random.split(rng)
        x = jax.random.normal(key_x, shape, dtype=jnp.float32)
        temp_sched = self._temp_schedule(sampling_temp, temp_mode, num_steps)

        # --- early (frozen) steps: compact loop, no trace --------------------- #
        def body(i, carry):
            x_curr, key = carry
            timestep = num_steps - 1 - i
            t = jnp.full((batch_size,), timestep, dtype=jnp.int32)
            key, step_key = jax.random.split(key)
            noise = jax.random.normal(step_key, shape, dtype=jnp.float32)
            x_next = self.p_sample(
                x_curr, t, cond1, cond2, cond2_mask, noise,
                eta=eta, sampling_temp=temp_sched[i],
            )
            return x_next, key

        x, key = jax.lax.fori_loop(0, n_early, body, (x, key_steps))

        # --- late (trainable) steps: unrolled, record transitions ------------ #
        xt_l, xprev_l, t_l, mean_l, std_l, lp_l = [], [], [], [], [], []
        for j in range(k):
            i = n_early + j
            timestep = num_steps - 1 - i  # runs k-1 .. 0
            t = jnp.full((batch_size,), timestep, dtype=jnp.int32)
            key, step_key = jax.random.split(key)
            noise = jax.random.normal(step_key, shape, dtype=jnp.float32)
            x_in = x

            # ONE denoiser call per step; both the update and the recorded
            # posterior stats derive from this single prediction (calling the
            # network twice would perturb the trajectory via its nnx rng streams).
            mean, _var, x0_pred = self.p_mean_variance(
                x_in, t, cond1, cond2, cond2_mask, denoise_fn=trainable_denoise_fn
            )
            std = jnp.sqrt(jnp.maximum(_var, 0.0))

            if rl_mode:
                std_s = jnp.maximum(std, sigma_sample_floor)
                # Exploration noise on every trainable step, including t=0 (the
                # default sampler suppresses it there); DPPO relies on this.
                x_out = mean + std_s * noise
                rec_std = std_s
            else:
                # Standard sampler math (bit-identical to sample()).
                x_out = self._ddim_update(x_in, t, x0_pred, noise, eta=eta, sampling_temp=temp_sched[i])
                rec_std = std

            std_lp = jnp.maximum(std, sigma_logprob_floor)
            lp = self._reduce_logdensity(
                self._gaussian_logdensity(x_out, mean, std_lp), prefix_len
            )

            xt_l.append(x_in); xprev_l.append(x_out); t_l.append(t)
            mean_l.append(mean); std_l.append(rec_std); lp_l.append(lp)
            x = x_out

        def _stack(lst, empty_shape):
            return jnp.stack(lst, axis=0) if lst else jnp.zeros(empty_shape, dtype=jnp.float32)

        trace = {
            "x_t": _stack(xt_l, (0,) + tuple(shape)),
            "x_prev": _stack(xprev_l, (0,) + tuple(shape)),
            "timestep": jnp.stack(t_l, axis=0) if t_l else jnp.zeros((0, batch_size), jnp.int32),
            "mean": _stack(mean_l, (0,) + tuple(shape)),
            "std": _stack(std_l, (0,) + tuple(shape)),
            "log_prob": jnp.stack(lp_l, axis=0) if lp_l else jnp.zeros((0, batch_size), jnp.float32),
        }
        return x, trace

    def resample(
        self,
        proposals: jnp.ndarray,
        cond1: jnp.ndarray,
        cond2: jnp.ndarray,
        cond2_mask: jnp.ndarray,
        n_timesteps: int,
        *,
        rng: jax.Array,
        noise_scale: float = 1.0,
        eta: float = 1.0,
        sampling_temp: float = 1.0,
        temp_mode: str = "uniform",
    ) -> jnp.ndarray:
        if n_timesteps <= 0 or n_timesteps > self.timesteps:
            raise ValueError(f"n_timesteps must be in [1, {self.timesteps}], got {n_timesteps}")
        return self._resample_impl(proposals, cond1, cond2, cond2_mask, n_timesteps, rng,
                                   noise_scale=noise_scale, eta=eta, sampling_temp=sampling_temp, temp_mode=temp_mode)