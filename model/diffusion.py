from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
from flax import nnx


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

    def __call__(self, x_bct: jnp.ndarray) -> jnp.ndarray:
        x_btc = jnp.swapaxes(x_bct, 1, 2)
        h_btc = self.conv(x_btc)
        h_btc = self.gn(h_btc)
        h_btc = nnx.silu(h_btc)
        return jnp.swapaxes(h_btc, 1, 2)


class Identity1d(nnx.Module):
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return x


class ResBlock1d(nnx.Module):
    def __init__(self, in_ch: int, out_ch: int, cond_ch: int, time_ch: int, rngs: nnx.Rngs, groups: int = 8):
        self.cond = nnx.Conv(in_features=cond_ch, out_features=in_ch, kernel_size=(1,), padding="SAME", rngs=rngs)
        self.time = nnx.Conv(in_features=time_ch, out_features=in_ch, kernel_size=(1,), padding="SAME", rngs=rngs)
        self.block1 = Conv1dBlock(in_ch, out_ch, rngs=rngs, kernel=3, groups=groups)
        self.block2 = Conv1dBlock(out_ch, out_ch, rngs=rngs, kernel=3, groups=groups)
        self.skip = Identity1d() if in_ch == out_ch else nnx.Conv(in_features=in_ch, out_features=out_ch, kernel_size=(1,), padding="SAME", rngs=rngs)

    def __call__(self, x: jnp.ndarray, cond_feat: jnp.ndarray, time_feat: jnp.ndarray) -> jnp.ndarray:
        t_len = x.shape[-1]
        cond = self.cond(jnp.swapaxes(cond_feat, 1, 2))
        tim = self.time(jnp.swapaxes(time_feat, 1, 2))
        cond = jnp.swapaxes(cond, 1, 2)
        tim = jnp.swapaxes(tim, 1, 2)

        h = x + jnp.broadcast_to(cond, (x.shape[0], cond.shape[1], t_len)) + jnp.broadcast_to(tim, (x.shape[0], tim.shape[1], t_len))
        h = self.block1(h)
        h = self.block2(h)

        if isinstance(self.skip, Identity1d):
            return h + x
        skip = self.skip(jnp.swapaxes(x, 1, 2))
        skip = jnp.swapaxes(skip, 1, 2)
        return h + skip


class Downsample1d(nnx.Module):
    def __init__(self, ch: int, rngs: nnx.Rngs):
        self.conv = nnx.Conv(in_features=ch, out_features=ch, kernel_size=(4,), strides=(2,), padding=((1, 1),), rngs=rngs)

    def __call__(self, x_bct: jnp.ndarray) -> jnp.ndarray:
        y_btc = self.conv(jnp.swapaxes(x_bct, 1, 2))
        return jnp.swapaxes(y_btc, 1, 2)


class Upsample1d(nnx.Module):
    def __init__(self, ch: int, rngs: nnx.Rngs):
        self.conv = nnx.Conv(in_features=ch, out_features=ch, kernel_size=(3,), padding="SAME", rngs=rngs)

    def __call__(self, x_bct: jnp.ndarray) -> jnp.ndarray:
        x_btc = jnp.swapaxes(x_bct, 1, 2)
        x_btc = jax.image.resize(x_btc, shape=(x_btc.shape[0], x_btc.shape[1] * 2, x_btc.shape[2]), method="nearest")
        y_btc = self.conv(x_btc)
        return jnp.swapaxes(y_btc, 1, 2)


class Projector(nnx.Module):
    def __init__(self, in_dim: int, out_dim: int, rngs: nnx.Rngs):
        self.fc1 = nnx.Linear(in_dim, out_dim * 4, rngs=rngs)
        self.fc2 = nnx.Linear(out_dim * 4, out_dim, rngs=rngs)

    def __call__(self, in_feat: jnp.ndarray) -> jnp.ndarray:
        h = self.fc1(in_feat)
        h = nnx.silu(h)
        return self.fc2(h)


class UNet1DConditioned(nnx.Module):
    def __init__(
        self,
        in_ch: int,
        rngs: nnx.Rngs,
        base_ch: int = 128,
        ch_mult: Tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int = 2,
        cond_dim: int = 256,
        time_emb_dim: int = 128,
        groups: int = 8,
    ):
        self.in_ch = in_ch
        self.time_emb_dim = time_emb_dim
        self.num_res_blocks = num_res_blocks
        self.n_down = len(ch_mult)

        self.cond_proj = Projector(cond_dim, base_ch, rngs=rngs)
        self.time_proj = Projector(time_emb_dim, base_ch, rngs=rngs)
        self.stem = nnx.Conv(in_features=in_ch, out_features=base_ch, kernel_size=(3,), padding="SAME", rngs=rngs)

        ch = base_ch
        self.down_blocks = []
        self.downsamples = []
        self.skip_channels = []

        for i, m in enumerate(ch_mult):
            out_ch = base_ch * m
            for _ in range(num_res_blocks):
                self.down_blocks.append(ResBlock1d(ch, out_ch, base_ch, base_ch, rngs=rngs, groups=groups))
                ch = out_ch
            self.skip_channels.append(ch)
            self.downsamples.append(Downsample1d(ch, rngs=rngs) if i != len(ch_mult) - 1 else Identity1d())

        self.mid_block1 = ResBlock1d(ch, ch, base_ch, base_ch, rngs=rngs, groups=groups)
        self.mid_block2 = ResBlock1d(ch, ch, base_ch, base_ch, rngs=rngs, groups=groups)

        self.up_blocks = []
        self.upsamples = []
        for i, m in reversed(list(enumerate(ch_mult))):
            out_ch = base_ch * m
            for _ in range(num_res_blocks):
                self.up_blocks.append(ResBlock1d(ch + self.skip_channels[i], out_ch, base_ch, base_ch, rngs=rngs, groups=groups))
                ch = out_ch
            self.upsamples.append(Upsample1d(ch, rngs=rngs) if i != 0 else Identity1d())

        self.out_norm = nnx.GroupNorm(num_features=ch, num_groups=min(groups, ch), epsilon=1e-5, rngs=rngs)
        self.out_conv = nnx.Conv(in_features=ch, out_features=in_ch, kernel_size=(3,), padding="SAME", rngs=rngs)

    def __call__(self, x: jnp.ndarray, t: jnp.ndarray, c: jnp.ndarray) -> jnp.ndarray:
        orig_t = x.shape[-1]
        divisor = 2 ** (self.n_down - 1)
        if orig_t % divisor != 0:
            target_t = ((orig_t // divisor) + 1) * divisor
            x = jnp.pad(x, ((0, 0), (0, 0), (0, target_t - orig_t)), mode="constant")

        t_emb = sinusoidal_timestep_embedding(t, self.time_emb_dim)
        cond_feat = self.cond_proj(c)[:, :, None]
        time_feat = self.time_proj(t_emb)[:, :, None]

        h = self.stem(jnp.swapaxes(x, 1, 2))
        h = jnp.swapaxes(h, 1, 2)

        skips = []
        rb = 0
        for i in range(self.n_down):
            for _ in range(self.num_res_blocks):
                h = self.down_blocks[rb](h, cond_feat, time_feat)
                rb += 1
            skips.append(h)
            h = self.downsamples[i](h)

        h = self.mid_block1(h, cond_feat, time_feat)
        h = self.mid_block2(h, cond_feat, time_feat)

        rb_up = 0
        for i in range(self.n_down):
            skip = skips.pop()
            if h.shape[-1] != skip.shape[-1]:
                min_len = min(h.shape[-1], skip.shape[-1])
                h = h[..., :min_len]
                skip = skip[..., :min_len]

            for _ in range(self.num_res_blocks):
                h = jnp.concatenate([h, skip], axis=1)
                h = self.up_blocks[rb_up](h, cond_feat, time_feat)
                rb_up += 1

            h = self.upsamples[i](h)

        h = jnp.swapaxes(h, 1, 2)
        h = self.out_norm(h)
        h = nnx.silu(h)
        h = self.out_conv(h)
        h = jnp.swapaxes(h, 1, 2)
        return h[:, :, :orig_t]


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
        timesteps: int = 50,
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

        self._jit_loss_with_noise_t = nnx.jit(self._loss_with_noise_t_impl)
        self._jit_sample_with_fixed_noises = nnx.jit(self._sample_with_fixed_noises_impl)

    @staticmethod
    def _extract(a: jnp.ndarray, t: jnp.ndarray, x_shape: Sequence[int]) -> jnp.ndarray:
        out = a[t]
        return out.reshape((x_shape[0], 1, 1))

    def q_sample(self, x0: jnp.ndarray, t: jnp.ndarray, noise: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        if noise is None:
            raise ValueError("q_sample requires explicit noise for deterministic behavior")
        sqrt_ab = self._extract(self.sqrt_alphas_cumprod.value, t, x0.shape)
        sqrt_1mab = self._extract(self.sqrt_one_minus_alphas_cumprod.value, t, x0.shape)
        return sqrt_ab * x0 + sqrt_1mab * noise

    def predict_x0_from_eps(self, x_t: jnp.ndarray, t: jnp.ndarray, eps: jnp.ndarray) -> jnp.ndarray:
        sqrt_ab = self._extract(self.sqrt_alphas_cumprod.value, t, x_t.shape)
        sqrt_1mab = self._extract(self.sqrt_one_minus_alphas_cumprod.value, t, x_t.shape)
        return (x_t - sqrt_1mab * eps) / sqrt_ab

    def p_mean_variance(self, x_t: jnp.ndarray, t: jnp.ndarray, cond: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        pred = self.denoise_fn(x_t, t, cond)

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

    def p_sample(self, x_t: jnp.ndarray, t: jnp.ndarray, cond: jnp.ndarray, noise: jnp.ndarray) -> jnp.ndarray:
        mean, var, _ = self.p_mean_variance(x_t, t, cond)
        nonzero_mask = (t != 0).astype(x_t.dtype).reshape((-1, 1, 1))
        return mean + nonzero_mask * jnp.sqrt(var) * noise

    def _sample_with_fixed_noises_impl(
        self,
        shape: Tuple[int, int, int],
        cond: jnp.ndarray,
        x_init: jnp.ndarray,
        step_noises: jnp.ndarray,
    ) -> jnp.ndarray:
        del shape
        batch_size = x_init.shape[0]
        num_steps = step_noises.shape[0]

        def body(i: int, x_curr: jnp.ndarray) -> jnp.ndarray:
            timestep = num_steps - 1 - i
            t = jnp.full((batch_size,), timestep, dtype=jnp.int32)
            return self.p_sample(x_curr, t, cond, step_noises[timestep])

        return jax.lax.fori_loop(0, num_steps, body, x_init)

    def _loss_with_noise_t_impl(self, x0: jnp.ndarray, cond: jnp.ndarray, noise: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        x_t = self.q_sample(x0, t, noise=noise)
        pred = self.denoise_fn(x_t, t, cond)

        if self.predict_type == "eps":
            return jnp.mean((pred - noise) ** 2)
        if self.predict_type == "mu":
            return jnp.mean((pred - x0) ** 2)
        if self.predict_type == "v":
            sqrt_ab = self._extract(self.sqrt_alphas_cumprod.value, t, x0.shape)
            sqrt_1mab = self._extract(self.sqrt_one_minus_alphas_cumprod.value, t, x0.shape)
            v = sqrt_ab * noise - sqrt_1mab * x0
            return jnp.mean((pred - v) ** 2)
        raise ValueError(f"Unsupported predict_type: {self.predict_type}")

    def sample(
        self,
        shape: Tuple[int, int, int],
        cond: jnp.ndarray,
        *,
        rng: Optional[jax.Array] = None,
        x_init: Optional[jnp.ndarray] = None,
        step_noises: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        if x_init is not None and step_noises is not None:
            return self._jit_sample_with_fixed_noises(shape, cond, x_init, step_noises)
        if rng is None:
            raise ValueError("Provide either (x_init and step_noises) or rng")
        key_x, key_steps = jax.random.split(rng)
        x_init = jax.random.normal(key_x, shape, dtype=jnp.float32)
        step_noises = jax.random.normal(key_steps, (self.timesteps, *shape), dtype=jnp.float32)
        return self._jit_sample_with_fixed_noises(shape, cond, x_init, step_noises)

    def loss(
        self,
        x0: jnp.ndarray,
        cond: jnp.ndarray,
        *,
        rng: Optional[jax.Array] = None,
        noise: Optional[jnp.ndarray] = None,
        t: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        if noise is not None and t is not None:
            return self._loss_with_noise_t_impl(x0, cond, noise, t)
        if rng is None:
            raise ValueError("Provide either (noise and t) or rng")
        batch = x0.shape[0]
        key_t, key_noise = jax.random.split(rng)
        t = jax.random.randint(key_t, (batch,), 0, self.timesteps, dtype=jnp.int32)
        noise = jax.random.normal(key_noise, x0.shape, dtype=jnp.float32)
        return self._jit_loss_with_noise_t(x0, cond, noise, t)
