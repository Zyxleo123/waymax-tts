import math
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx


def sinusoidal_timestep_embedding(t: jnp.ndarray, dim: int, max_period: int = 10_000) -> jnp.ndarray:
    if not jnp.issubdtype(t.dtype, jnp.floating):
        t = t.astype(jnp.float32)
    half = dim // 2
    freqs = jnp.exp(-math.log(max_period) * jnp.arange(0, half, dtype=t.dtype) / half)
    args = t[:, None] * freqs[None, :]
    emb = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
    if dim % 2 == 1:
        emb = jnp.concatenate([emb, jnp.zeros_like(emb[:, :1])], axis=-1)
    return emb


class Conv1d(nnx.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int = 1, padding: int = 0, *, rngs: nnx.Rngs):
        self.conv = nnx.Conv(
            in_features=in_ch,
            out_features=out_ch,
            kernel_size=(kernel_size,),
            strides=(stride,),
            padding=[(padding, padding)],
            rngs=rngs,
        )

    def __call__(self, x_ntc: jnp.ndarray) -> jnp.ndarray:
        return self.conv(x_ntc)


class Conv1dBlock(nnx.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, groups: int = 8, *, rngs: nnx.Rngs):
        pad = kernel // 2
        self.conv = Conv1d(in_ch, out_ch, kernel, padding=pad, rngs=rngs)
        self.gn = nnx.GroupNorm(num_features=out_ch, num_groups=min(groups, out_ch), rngs=rngs)

    def __call__(self, x_ntc: jnp.ndarray) -> jnp.ndarray:
        y = self.conv(x_ntc)
        y = self.gn(y)
        return nnx.silu(y)


class ResBlock1d(nnx.Module):
    def __init__(self, in_ch: int, out_ch: int, cond_ch: int, time_ch: int, groups: int = 8, *, rngs: nnx.Rngs):
        self.cond = Conv1d(cond_ch, in_ch, 1, rngs=rngs)
        self.time = Conv1d(time_ch, in_ch, 1, rngs=rngs)
        self.block1 = Conv1dBlock(in_ch, out_ch, kernel=3, groups=groups, rngs=rngs)
        self.block2 = Conv1dBlock(out_ch, out_ch, kernel=3, groups=groups, rngs=rngs)
        self.skip = None if in_ch == out_ch else Conv1d(in_ch, out_ch, 1, rngs=rngs)

    def __call__(self, x_ntc: jnp.ndarray, cond_feat: jnp.ndarray, time_feat: jnp.ndarray) -> jnp.ndarray:
        h = x_ntc + jnp.broadcast_to(self.cond(cond_feat), x_ntc.shape) + jnp.broadcast_to(self.time(time_feat), x_ntc.shape)
        h = self.block1(h)
        h = self.block2(h)
        skip = x_ntc if self.skip is None else self.skip(x_ntc)
        return h + skip


class Downsample1d(nnx.Module):
    def __init__(self, ch: int, *, rngs: nnx.Rngs):
        self.conv = Conv1d(ch, ch, kernel_size=4, stride=2, padding=1, rngs=rngs)

    def __call__(self, x_ntc: jnp.ndarray) -> jnp.ndarray:
        return self.conv(x_ntc)


class Upsample1d(nnx.Module):
    def __init__(self, ch: int, *, rngs: nnx.Rngs):
        self.conv = Conv1d(ch, ch, kernel_size=3, padding=1, rngs=rngs)

    def __call__(self, x_ntc: jnp.ndarray) -> jnp.ndarray:
        x_ntc = jnp.repeat(x_ntc, repeats=2, axis=1)
        return self.conv(x_ntc)


class Projector(nnx.Module):
    def __init__(self, in_dim: int, out_dim: int, *, rngs: nnx.Rngs):
        self.dense0 = nnx.Linear(in_dim, out_dim * 4, rngs=rngs)
        self.dense1 = nnx.Linear(out_dim * 4, out_dim, rngs=rngs)

    def __call__(self, in_feat: jnp.ndarray) -> jnp.ndarray:
        h = self.dense0(in_feat)
        h = nnx.silu(h)
        h = self.dense1(h)
        return h


class UNet1DConditioned(nnx.Module):
    def __init__(
        self,
        in_ch: int,
        base_ch: int = 128,
        ch_mult: Tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int = 2,
        cond_dim: int = 256,
        time_emb_dim: int = 128,
        groups: int = 8,
        compute_dtype: jnp.dtype = jnp.float32,
        *,
        rngs: nnx.Rngs,
    ):
        self.in_ch = in_ch
        self.base_ch = base_ch
        self.ch_mult = tuple(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.cond_dim = cond_dim
        self.time_emb_dim = time_emb_dim
        self.groups = groups
        self.compute_dtype = compute_dtype

        self.cond_proj = Projector(cond_dim, base_ch, rngs=rngs)
        self.time_proj = Projector(time_emb_dim, base_ch, rngs=rngs)
        self.stem = Conv1d(in_ch, base_ch, kernel_size=3, padding=1, rngs=rngs)

        self.down_blocks = []
        self.downsamples = []
        self.skip_channels = []

        ch = base_ch
        for i, m in enumerate(self.ch_mult):
            out_ch = base_ch * m
            for _ in range(self.num_res_blocks):
                self.down_blocks.append(ResBlock1d(ch, out_ch, base_ch, base_ch, groups=groups, rngs=rngs))
                ch = out_ch
            self.skip_channels.append(ch)
            if i != len(self.ch_mult) - 1:
                self.downsamples.append(Downsample1d(ch, rngs=rngs))
            else:
                self.downsamples.append(None)

        self.mid_block1 = ResBlock1d(ch, ch, base_ch, base_ch, groups=groups, rngs=rngs)
        self.mid_block2 = ResBlock1d(ch, ch, base_ch, base_ch, groups=groups, rngs=rngs)

        self.up_blocks = []
        self.upsamples = []

        for i, m in reversed(list(enumerate(self.ch_mult))):
            out_ch = base_ch * m
            for _ in range(self.num_res_blocks):
                self.up_blocks.append(ResBlock1d(ch + self.skip_channels[i], out_ch, base_ch, base_ch, groups=groups, rngs=rngs))
                ch = out_ch
            if i != 0:
                self.upsamples.append(Upsample1d(ch, rngs=rngs))
            else:
                self.upsamples.append(None)

        self.out_norm = nnx.GroupNorm(num_features=ch, num_groups=min(groups, ch), rngs=rngs)
        self.out_conv = Conv1d(ch, in_ch, kernel_size=3, padding=1, rngs=rngs)
        self.n_down = len(self.ch_mult)

    def __call__(self, x_nct: jnp.ndarray, t: jnp.ndarray, c: jnp.ndarray) -> jnp.ndarray:
        x_nct = x_nct.astype(self.compute_dtype)
        c = c.astype(self.compute_dtype)

        _, _, t_len = x_nct.shape
        x_ntc = jnp.swapaxes(x_nct, 1, 2)

        divisor = 2 ** (self.n_down - 1)
        if t_len % divisor != 0:
            target_t = ((t_len // divisor) + 1) * divisor
            pad_t = target_t - t_len
            x_ntc = jnp.pad(x_ntc, ((0, 0), (0, pad_t), (0, 0)))

        t_emb = sinusoidal_timestep_embedding(t, self.time_emb_dim).astype(self.compute_dtype)
        cond_feat = self.cond_proj(c)[:, None, :]
        time_feat = self.time_proj(t_emb)[:, None, :]

        h = self.stem(x_ntc)
        skips = []
        rb = 0
        for i in range(self.n_down):
            for _ in range(self.num_res_blocks):
                h = self.down_blocks[rb](h, cond_feat, time_feat)
                rb += 1
            skips.append(h)
            if self.downsamples[i] is not None:
                h = self.downsamples[i](h)

        h = self.mid_block1(h, cond_feat, time_feat)
        h = self.mid_block2(h, cond_feat, time_feat)

        rb_up = 0
        for i in range(self.n_down):
            skip = skips.pop()
            if h.shape[1] != skip.shape[1]:
                l = min(h.shape[1], skip.shape[1])
                h = h[:, :l, :]
                skip = skip[:, :l, :]
            for _ in range(self.num_res_blocks):
                h = jnp.concatenate([h, skip], axis=-1)
                h = self.up_blocks[rb_up](h, cond_feat, time_feat)
                rb_up += 1
            if self.upsamples[i] is not None:
                h = self.upsamples[i](h)

        h = self.out_norm(h)
        h = nnx.silu(h)
        h = self.out_conv(h)
        h = h[:, :t_len, :]
        return jnp.swapaxes(h, 1, 2)


def make_beta_schedule(timesteps: int, schedule: str = "cosine") -> jnp.ndarray:
    if schedule == "linear":
        return jnp.linspace(1e-4, 2e-2, timesteps, dtype=jnp.float32)
    if schedule == "cosine":
        s = 0.008
        steps = timesteps + 1
        x = jnp.linspace(0, timesteps, steps, dtype=jnp.float32)
        alphas_cumprod = jnp.cos(((x / timesteps) + s) / (1 + s) * math.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return jnp.clip(betas, 0.0, 0.999).astype(jnp.float32)
    raise ValueError(f"Unknown schedule: {schedule}")


class GaussianDiffusion(nnx.Module):
    def __init__(
        self,
        denoise_fn: UNet1DConditioned,
        timesteps: int = 50,
        beta_schedule: str = "cosine",
        predict_type: str = "v",
        clip_denoised: bool = False,
    ) -> None:
        self.denoise_fn = denoise_fn
        self.timesteps = timesteps
        self.beta_schedule = beta_schedule
        self.predict_type = predict_type
        self.clip_denoised = clip_denoised

        betas = make_beta_schedule(self.timesteps, self.beta_schedule)
        alphas = 1.0 - betas
        alphas_cumprod = jnp.cumprod(alphas, axis=0)
        alphas_cumprod_prev = jnp.concatenate([jnp.array([1.0], dtype=alphas.dtype), alphas_cumprod[:-1]], axis=0)
        posterior_var = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

        self.sqrt_alphas_cumprod = nnx.BatchStat(jnp.sqrt(alphas_cumprod))
        self.sqrt_one_minus_alphas_cumprod = nnx.BatchStat(jnp.sqrt(1.0 - alphas_cumprod))
        self.posterior_variance = nnx.BatchStat(jnp.maximum(posterior_var, 1e-20))
        self.posterior_mean_coef1 = nnx.BatchStat(betas * jnp.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.posterior_mean_coef2 = nnx.BatchStat((1.0 - alphas_cumprod_prev) * jnp.sqrt(alphas) / (1.0 - alphas_cumprod))
        self._loss_jit_fn = nnx.jit(lambda m, x, c, k: m.loss(x, c, key=k))
        self._sample_jit_fns = {}

    @staticmethod
    def _extract(a: jnp.ndarray, t: jnp.ndarray, x_shape: Tuple[int, ...], dtype: jnp.dtype) -> jnp.ndarray:
        if hasattr(a, "value"):
            a = a.value
        out = a[t]
        return jnp.broadcast_to(out[:, None, None], (x_shape[0], 1, 1)).astype(dtype)

    def q_sample(self, x0: jnp.ndarray, t: jnp.ndarray, noise: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        if noise is None:
            raise ValueError("noise must be provided for q_sample in the JAX path.")
        sqrt_ab = self._extract(self.sqrt_alphas_cumprod, t, x0.shape, x0.dtype)
        sqrt_1mab = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape, x0.dtype)
        return sqrt_ab * x0 + sqrt_1mab * noise

    def predict_x0_from_eps(self, x_t: jnp.ndarray, t: jnp.ndarray, eps: jnp.ndarray) -> jnp.ndarray:
        sqrt_ab = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape, x_t.dtype)
        sqrt_1mab = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape, x_t.dtype)
        return (x_t - sqrt_1mab * eps) / sqrt_ab

    def p_mean_variance(self, x_t: jnp.ndarray, t: jnp.ndarray, cond: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        pred = self.denoise_fn(x_t, t, cond)
        if self.predict_type == "eps":
            x0_pred = self.predict_x0_from_eps(x_t, t, pred)
        elif self.predict_type == "mu":
            x0_pred = pred
        elif self.predict_type == "v":
            sqrt_ab = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape, x_t.dtype)
            sqrt_1mab = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape, x_t.dtype)
            x0_pred = sqrt_ab * x_t - sqrt_1mab * pred
        else:
            raise ValueError(f"Unsupported predict_type: {self.predict_type}")

        if self.clip_denoised:
            x0_pred = jnp.clip(x0_pred, -1.0, 1.0)

        mean = self._extract(self.posterior_mean_coef1, t, x_t.shape, x_t.dtype) * x0_pred + self._extract(
            self.posterior_mean_coef2, t, x_t.shape, x_t.dtype
        ) * x_t
        var = self._extract(self.posterior_variance, t, x_t.shape, x_t.dtype)
        return mean, var, x0_pred

    def p_sample(self, x_t: jnp.ndarray, t: jnp.ndarray, cond: jnp.ndarray, *, key: jax.Array) -> jnp.ndarray:
        mean, var, _ = self.p_mean_variance(x_t, t, cond)
        noise = jax.random.normal(key, shape=x_t.shape, dtype=x_t.dtype)
        nonzero_mask = (t != 0).astype(x_t.dtype)[:, None, None]
        return mean + nonzero_mask * jnp.sqrt(var) * noise

    def sample(self, shape: Tuple[int, int, int], cond: jnp.ndarray, *, key: jax.Array) -> jnp.ndarray:
        b, _, _ = shape
        keys = jax.random.split(key, self.timesteps + 1)
        x0 = jax.random.normal(keys[0], shape=shape, dtype=cond.dtype)

        def body(i, x):
            t_val = self.timesteps - 1 - i
            t = jnp.full((b,), t_val, dtype=jnp.int32)
            return self.p_sample(x, t, cond, key=keys[i + 1])

        return jax.lax.fori_loop(0, self.timesteps, body, x0)

    def sample_jit(self, shape: Tuple[int, int, int], cond: jnp.ndarray, *, key: jax.Array) -> jnp.ndarray:
        fn = self._sample_jit_fns.get(shape)
        if fn is None:
            fn = nnx.jit(lambda m, c, k: m.sample(shape, c, key=k))
            self._sample_jit_fns[shape] = fn
        return fn(self, cond, key)

    def loss(self, x0: jnp.ndarray, cond: jnp.ndarray, *, key: jax.Array) -> jnp.ndarray:
        b = x0.shape[0]
        key_t, key_n = jax.random.split(key)
        t = jax.random.randint(key_t, shape=(b,), minval=0, maxval=self.timesteps, dtype=jnp.int32)
        noise = jax.random.normal(key_n, shape=x0.shape, dtype=x0.dtype)
        x_t = self.q_sample(x0, t, noise=noise)
        pred = self.denoise_fn(x_t, t, cond)

        if self.predict_type == "eps":
            target = noise
        elif self.predict_type == "mu":
            target = x0
        elif self.predict_type == "v":
            sqrt_ab = self._extract(self.sqrt_alphas_cumprod, t, x0.shape, x0.dtype)
            sqrt_1mab = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape, x0.dtype)
            target = sqrt_ab * noise - sqrt_1mab * x0
        else:
            raise ValueError(f"Unsupported predict_type: {self.predict_type}")

        return jnp.mean((pred - target) ** 2)

    def loss_jit(self, x0: jnp.ndarray, cond: jnp.ndarray, *, key: jax.Array) -> jnp.ndarray:
        return self._loss_jit_fn(self, x0, cond, key)
