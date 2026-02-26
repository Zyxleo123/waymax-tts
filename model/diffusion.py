import math
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn


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


class GroupNormNCT(nn.Module):
    num_groups: int
    num_channels: int
    eps: float = 1e-5

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        b, c, t = x.shape
        g = min(self.num_groups, self.num_channels)
        assert c == self.num_channels
        assert c % g == 0
        xg = x.reshape(b, g, c // g, t)
        mean = jnp.mean(xg, axis=(2, 3), keepdims=True)
        var = jnp.mean((xg - mean) ** 2, axis=(2, 3), keepdims=True)
        y = ((xg - mean) / jnp.sqrt(var + self.eps)).reshape(b, c, t)
        scale = self.param("scale", nn.initializers.ones, (c,))
        bias = self.param("bias", nn.initializers.zeros, (c,))
        return y * scale[None, :, None] + bias[None, :, None]


class Conv1d(nn.Module):
    features: int
    kernel_size: int
    stride: int = 1
    padding: int = 0

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x_ntc = jnp.swapaxes(x, 1, 2)
        y_ntc = nn.Conv(
            features=self.features,
            kernel_size=(self.kernel_size,),
            strides=(self.stride,),
            padding=[(self.padding, self.padding)],
        )(x_ntc)
        return jnp.swapaxes(y_ntc, 1, 2)


class Conv1dBlock(nn.Module):
    in_ch: int
    out_ch: int
    kernel: int = 3
    groups: int = 8

    def setup(self) -> None:
        pad = self.kernel // 2
        self.conv = Conv1d(features=self.out_ch, kernel_size=self.kernel, padding=pad)
        self.gn = GroupNormNCT(num_groups=min(self.groups, self.out_ch), num_channels=self.out_ch)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return nn.silu(self.gn(self.conv(x)))


class ResBlock1d(nn.Module):
    in_ch: int
    out_ch: int
    cond_ch: int
    time_ch: int
    groups: int = 8

    def setup(self) -> None:
        self.cond = Conv1d(features=self.in_ch, kernel_size=1)
        self.time = Conv1d(features=self.in_ch, kernel_size=1)
        self.block1 = Conv1dBlock(self.in_ch, self.out_ch, kernel=3, groups=self.groups)
        self.block2 = Conv1dBlock(self.out_ch, self.out_ch, kernel=3, groups=self.groups)
        self.skip = None if self.in_ch == self.out_ch else Conv1d(features=self.out_ch, kernel_size=1)

    def __call__(self, x: jnp.ndarray, cond_feat: jnp.ndarray, time_feat: jnp.ndarray) -> jnp.ndarray:
        h = x + jnp.broadcast_to(self.cond(cond_feat), x.shape) + jnp.broadcast_to(self.time(time_feat), x.shape)
        h = self.block1(h)
        h = self.block2(h)
        skip = x if self.skip is None else self.skip(x)
        return h + skip


class Downsample1d(nn.Module):
    ch: int

    def setup(self) -> None:
        self.conv = Conv1d(features=self.ch, kernel_size=4, stride=2, padding=1)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.conv(x)


class Upsample1d(nn.Module):
    ch: int

    def setup(self) -> None:
        self.conv = Conv1d(features=self.ch, kernel_size=3, padding=1)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = jnp.repeat(x, repeats=2, axis=-1)
        return self.conv(x)


class Projector(nn.Module):
    in_dim: int
    out_dim: int

    @nn.compact
    def __call__(self, in_feat: jnp.ndarray) -> jnp.ndarray:
        h = nn.Dense(self.out_dim * 4)(in_feat)
        h = nn.silu(h)
        h = nn.Dense(self.out_dim)(h)
        return h


class UNet1DConditioned(nn.Module):
    in_ch: int
    base_ch: int = 128
    ch_mult: Tuple[int, ...] = (1, 2, 4)
    num_res_blocks: int = 2
    cond_dim: int = 256
    time_emb_dim: int = 128
    groups: int = 8

    def setup(self) -> None:
        self.cond_proj = Projector(self.cond_dim, self.base_ch)
        self.time_proj = Projector(self.time_emb_dim, self.base_ch)
        self.stem = Conv1d(features=self.base_ch, kernel_size=3, padding=1)

        down_blocks: List[nn.Module] = []
        downsamples: List[Optional[nn.Module]] = []
        skip_channels: List[int] = []
        ch = self.base_ch
        for i, m in enumerate(self.ch_mult):
            out_ch = self.base_ch * m
            for j in range(self.num_res_blocks):
                down_blocks.append(
                    ResBlock1d(
                        in_ch=ch,
                        out_ch=out_ch,
                        cond_ch=self.base_ch,
                        time_ch=self.base_ch,
                        groups=self.groups,
                        name=f"down_rb_{i}_{j}",
                    )
                )
                ch = out_ch
            skip_channels.append(ch)
            if i != len(self.ch_mult) - 1:
                downsamples.append(Downsample1d(ch=ch, name=f"downsample_{i}"))
            else:
                downsamples.append(None)

        self.down_blocks = down_blocks
        self.downsamples = downsamples
        self.skip_channels = tuple(skip_channels)

        self.mid_block1 = ResBlock1d(ch, ch, self.base_ch, self.base_ch, groups=self.groups, name="mid_1")
        self.mid_block2 = ResBlock1d(ch, ch, self.base_ch, self.base_ch, groups=self.groups, name="mid_2")

        up_blocks: List[nn.Module] = []
        upsamples: List[Optional[nn.Module]] = []
        for i, m in reversed(list(enumerate(self.ch_mult))):
            out_ch = self.base_ch * m
            for j in range(self.num_res_blocks):
                up_blocks.append(
                    ResBlock1d(
                        in_ch=ch + self.skip_channels[i],
                        out_ch=out_ch,
                        cond_ch=self.base_ch,
                        time_ch=self.base_ch,
                        groups=self.groups,
                        name=f"up_rb_{i}_{j}",
                    )
                )
                ch = out_ch
            if i != 0:
                upsamples.append(Upsample1d(ch=ch, name=f"upsample_{i}"))
            else:
                upsamples.append(None)
        self.up_blocks = up_blocks
        self.upsamples = upsamples

        self.out_norm = GroupNormNCT(num_groups=min(self.groups, ch), num_channels=ch)
        self.out_conv = Conv1d(features=self.in_ch, kernel_size=3, padding=1)
        self.n_down = len(self.ch_mult)

    def __call__(self, x: jnp.ndarray, t: jnp.ndarray, c: jnp.ndarray) -> jnp.ndarray:
        _, _, t_len = x.shape
        divisor = 2 ** (self.n_down - 1)
        if t_len % divisor != 0:
            target_t = ((t_len // divisor) + 1) * divisor
            pad_t = target_t - t_len
            x = jnp.pad(x, ((0, 0), (0, 0), (0, pad_t)))

        t_emb = sinusoidal_timestep_embedding(t, self.time_emb_dim)
        cond_feat = self.cond_proj(c)[:, :, None]
        time_feat = self.time_proj(t_emb)[:, :, None]

        h = self.stem(x)
        skips: List[jnp.ndarray] = []
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
            if h.shape[-1] != skip.shape[-1]:
                l = min(h.shape[-1], skip.shape[-1])
                h = h[..., :l]
                skip = skip[..., :l]
            for _ in range(self.num_res_blocks):
                h = jnp.concatenate([h, skip], axis=1)
                h = self.up_blocks[rb_up](h, cond_feat, time_feat)
                rb_up += 1
            if self.upsamples[i] is not None:
                h = self.upsamples[i](h)

        h = nn.silu(self.out_norm(h))
        return self.out_conv(h)[:, :, :t_len]


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


class GaussianDiffusion(nn.Module):
    denoise_fn: nn.Module
    timesteps: int = 50
    beta_schedule: str = "cosine"
    predict_type: str = "v"
    clip_denoised: bool = False

    def _coeffs(self) -> Dict[str, jnp.ndarray]:
        betas = make_beta_schedule(self.timesteps, self.beta_schedule)
        alphas = 1.0 - betas
        alphas_cumprod = jnp.cumprod(alphas, axis=0)
        alphas_cumprod_prev = jnp.concatenate([jnp.array([1.0], dtype=alphas.dtype), alphas_cumprod[:-1]], axis=0)
        posterior_var = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        return {
            "betas": betas,
            "alphas": alphas,
            "alphas_cumprod": alphas_cumprod,
            "alphas_cumprod_prev": alphas_cumprod_prev,
            "sqrt_alphas_cumprod": jnp.sqrt(alphas_cumprod),
            "sqrt_one_minus_alphas_cumprod": jnp.sqrt(1.0 - alphas_cumprod),
            "posterior_variance": jnp.maximum(posterior_var, 1e-20),
            "posterior_log_variance_clipped": jnp.log(jnp.maximum(posterior_var, 1e-20)),
            "posterior_mean_coef1": betas * jnp.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
            "posterior_mean_coef2": (1.0 - alphas_cumprod_prev) * jnp.sqrt(alphas) / (1.0 - alphas_cumprod),
        }

    @staticmethod
    def _extract(a: jnp.ndarray, t: jnp.ndarray, x_shape: Tuple[int, ...]) -> jnp.ndarray:
        out = a[t]
        return jnp.broadcast_to(out[:, None, None], (x_shape[0], 1, 1))

    def q_sample(self, x0: jnp.ndarray, t: jnp.ndarray, noise: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        if noise is None:
            raise ValueError("noise must be provided for q_sample in the JAX path.")
        coeffs = self._coeffs()
        sqrt_ab = self._extract(coeffs["sqrt_alphas_cumprod"], t, x0.shape)
        sqrt_1mab = self._extract(coeffs["sqrt_one_minus_alphas_cumprod"], t, x0.shape)
        return sqrt_ab * x0 + sqrt_1mab * noise

    def predict_x0_from_eps(self, x_t: jnp.ndarray, t: jnp.ndarray, eps: jnp.ndarray) -> jnp.ndarray:
        coeffs = self._coeffs()
        sqrt_ab = self._extract(coeffs["sqrt_alphas_cumprod"], t, x_t.shape)
        sqrt_1mab = self._extract(coeffs["sqrt_one_minus_alphas_cumprod"], t, x_t.shape)
        return (x_t - sqrt_1mab * eps) / sqrt_ab

    def p_mean_variance(self, x_t: jnp.ndarray, t: jnp.ndarray, cond: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        coeffs = self._coeffs()
        pred = self.denoise_fn(x_t, t, cond)
        if self.predict_type == "eps":
            x0_pred = self.predict_x0_from_eps(x_t, t, pred)
        elif self.predict_type == "mu":
            x0_pred = pred
        elif self.predict_type == "v":
            sqrt_ab = self._extract(coeffs["sqrt_alphas_cumprod"], t, x_t.shape)
            sqrt_1mab = self._extract(coeffs["sqrt_one_minus_alphas_cumprod"], t, x_t.shape)
            x0_pred = sqrt_ab * x_t - sqrt_1mab * pred
        else:
            raise ValueError(f"Unsupported predict_type: {self.predict_type}")

        if self.clip_denoised:
            x0_pred = jnp.clip(x0_pred, -1.0, 1.0)

        mean = (
            self._extract(coeffs["posterior_mean_coef1"], t, x_t.shape) * x0_pred
            + self._extract(coeffs["posterior_mean_coef2"], t, x_t.shape) * x_t
        )
        var = self._extract(coeffs["posterior_variance"], t, x_t.shape)
        return mean, var, x0_pred

    def p_sample(self, x_t: jnp.ndarray, t: jnp.ndarray, cond: jnp.ndarray, *, key: jax.Array) -> jnp.ndarray:
        mean, var, _ = self.p_mean_variance(x_t, t, cond)
        noise = jax.random.normal(key, shape=x_t.shape, dtype=x_t.dtype)
        nonzero_mask = (t != 0).astype(x_t.dtype)[:, None, None]
        return mean + nonzero_mask * jnp.sqrt(var) * noise

    def sample(self, shape: Tuple[int, int, int], cond: jnp.ndarray, *, key: jax.Array) -> jnp.ndarray:
        b, _, _ = shape
        x = jax.random.normal(key, shape=shape, dtype=cond.dtype)
        for i in reversed(range(self.timesteps)):
            key, subkey = jax.random.split(key)
            t = jnp.full((b,), i, dtype=jnp.int32)
            x = self.p_sample(x, t, cond, key=subkey)
        return x

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
            coeffs = self._coeffs()
            sqrt_ab = self._extract(coeffs["sqrt_alphas_cumprod"], t, x0.shape)
            sqrt_1mab = self._extract(coeffs["sqrt_one_minus_alphas_cumprod"], t, x0.shape)
            target = sqrt_ab * noise - sqrt_1mab * x0
        else:
            raise ValueError(f"Unsupported predict_type: {self.predict_type}")

        return jnp.mean((pred - target) ** 2)
