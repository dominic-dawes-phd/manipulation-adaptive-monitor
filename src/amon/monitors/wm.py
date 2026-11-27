"""World-model monitor: latent prediction residual on the arriving frame (spec section 6.2).

A frozen image encoder maps frame t to z_t, the mean of its patch tokens. A small predictor maps
the latents z_{t-k-L+1..t-k} and the executed actions a_{t-k..t-1} to ẑ_t. The score at t is
1 - cos(ẑ_t, z_t): it uses only frames and actions up to t, so it is causal.
"""

import logging
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from amon.settings import WmSettings

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def device() -> torch.device:
    """Returns the GPU if there is one."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_encoder(settings: WmSettings) -> nn.Module:
    """Loads the frozen encoder from its pinned torch.hub repo, in eval mode on `device()`."""
    encoder = torch.hub.load(settings.encoder_repo, settings.encoder, trust_repo=True)
    return encoder.to(device()).eval().requires_grad_(False)


def encode_frames(encoder: nn.Module, frames: np.ndarray, batch_size: int) -> np.ndarray:
    """Returns the mean patch token of every frame.

    Args:
        encoder: From `load_encoder`.
        frames: [T, H, W, 3] uint8; H and W must be multiples of the encoder's patch size.
        batch_size: Frames per forward pass.

    Returns:
        [T, D] float32.
    """
    mean = torch.tensor(IMAGENET_MEAN, device=device()).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device()).view(1, 3, 1, 1)
    latents = []
    with torch.inference_mode():
        for start in range(0, len(frames), batch_size):
            batch = torch.from_numpy(frames[start : start + batch_size]).to(device())
            pixels = (batch.permute(0, 3, 1, 2).float() / 255.0 - mean) / std
            tokens = encoder.forward_features(pixels)["x_norm_patchtokens"]
            latents.append(tokens.mean(dim=1).cpu())
    return torch.cat(latents).numpy().astype(np.float32)


class LatentPredictor(nn.Module):
    """Residual MLP: ẑ_t = z_{t-k} + f(z_{t-k-L+1..t-k}, a_{t-k..t-1})."""

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        settings: WmSettings,
        action_mean: np.ndarray,
        action_std: np.ndarray,
    ) -> None:
        """Builds the network; actions are standardized with the training statistics."""
        super().__init__()
        self.register_buffer("action_mean", torch.as_tensor(action_mean, dtype=torch.float32))
        self.register_buffer("action_std", torch.as_tensor(action_std, dtype=torch.float32))
        inputs = settings.history * latent_dim + settings.horizon * action_dim
        self.network = nn.Sequential(
            nn.Linear(inputs, settings.hidden_dim),
            nn.GELU(),
            nn.Linear(settings.hidden_dim, settings.hidden_dim),
            nn.GELU(),
            nn.Linear(settings.hidden_dim, latent_dim),
        )

    def forward(self, latents: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Maps [B, L, D] latents and [B, k, A] actions to [B, D] predictions."""
        scaled = (actions - self.action_mean) / self.action_std
        features = torch.cat([latents.flatten(1), scaled.flatten(1)], dim=1)
        return latents[:, -1] + self.network(features)


def prediction_windows(
    latents: np.ndarray, actions: np.ndarray, history: int, horizon: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns the steps t that can be predicted, with their input latents and actions.

    Args:
        latents: [T, D].
        actions: [T, A] executed actions; a[t] is applied after frame t.
        history: L.
        horizon: k.

    Returns:
        Steps [N] (t from L + k - 1 to T - 1), latent windows [N, L, D] = z[t-k-L+1 .. t-k] and
        action windows [N, k, A] = a[t-k .. t-1].
    """
    steps = np.arange(history + horizon - 1, len(latents))
    latent_rows = steps[:, None] - horizon - history + 1 + np.arange(history)[None, :]
    action_rows = steps[:, None] - horizon + np.arange(horizon)[None, :]
    return steps, latents[latent_rows], actions[action_rows]


def stack_windows(
    latents: list[np.ndarray],
    actions: list[np.ndarray],
    episodes: np.ndarray,
    settings: WmSettings,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Concatenates the prediction windows of the given episodes into tensors on `device()`.

    Returns:
        Latent windows [N, L, D], action windows [N, k, A] and target latents [N, D].
    """
    latent_windows, action_windows, targets = [], [], []
    for index in episodes:
        steps, window_latents, window_actions = prediction_windows(
            latents[index], actions[index], settings.history, settings.horizon
        )
        latent_windows.append(window_latents)
        action_windows.append(window_actions)
        targets.append(latents[index][steps])
    return (
        torch.from_numpy(np.concatenate(latent_windows)).to(device()),
        torch.from_numpy(np.concatenate(action_windows)).to(device()),
        torch.from_numpy(np.concatenate(targets)).to(device()),
    )


def cosine_residual(predicted: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
    """Returns 1 - cos(ẑ, z) per row."""
    return 1.0 - nn.functional.cosine_similarity(predicted, actual, dim=1)


def train_predictor(
    latents: list[np.ndarray], actions: list[np.ndarray], settings: WmSettings
) -> tuple[LatentPredictor, list[float]]:
    """Trains the predictor on whole episodes, holding out `val_fraction` of them.

    Args:
        latents: Per-episode [T, D] latents of the training episodes.
        actions: Per-episode [T, A] executed actions, in the same order.
        settings: Architecture and optimization parameters.

    Returns:
        The trained predictor in eval mode and the validation loss after each epoch.
    """
    rng = np.random.default_rng(settings.seed)
    torch.manual_seed(settings.seed)
    held_out = max(1, round(settings.val_fraction * len(latents)))
    order = rng.permutation(len(latents))
    val_episodes, train_episodes = order[:held_out], order[held_out:]
    train_latents, train_window_actions, train_targets = stack_windows(
        latents, actions, train_episodes, settings
    )
    val_latents, val_window_actions, val_targets = stack_windows(
        latents, actions, val_episodes, settings
    )
    train_actions = np.concatenate([actions[i] for i in train_episodes])
    predictor = LatentPredictor(
        latents[0].shape[1],
        actions[0].shape[1],
        settings,
        train_actions.mean(axis=0),
        train_actions.std(axis=0) + 1e-6,
    ).to(device())
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=settings.lr)
    val_losses = []
    for epoch in range(settings.epochs):
        predictor.train()
        for batch in torch.randperm(len(train_targets), device=device()).split(settings.batch_size):
            predicted = predictor(train_latents[batch], train_window_actions[batch])
            loss = cosine_residual(predicted, train_targets[batch]).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        predictor.eval()
        with torch.inference_mode():
            predicted = predictor(val_latents, val_window_actions)
            val_losses.append(float(cosine_residual(predicted, val_targets).mean()))
        logging.info("epoch %d: train %.5f, val %.5f", epoch + 1, float(loss), val_losses[-1])
    return predictor, val_losses


def score_episode(
    predictor: LatentPredictor, latents: np.ndarray, actions: np.ndarray, settings: WmSettings
) -> np.ndarray:
    """Returns s(t) = 1 - cos(ẑ_t, z_t) for every step; NaN before step L + k - 1."""
    steps, window_latents, window_actions = prediction_windows(
        latents, actions, settings.history, settings.horizon
    )
    scores = np.full(len(latents), np.nan, dtype=np.float32)
    with torch.inference_mode():
        predicted = predictor(
            torch.from_numpy(window_latents).to(device()),
            torch.from_numpy(window_actions).to(device()),
        )
        residual = cosine_residual(predicted, torch.from_numpy(latents[steps]).to(device()))
    scores[steps] = residual.cpu().numpy()
    return scores


def measure_check_latency(
    encoder: nn.Module, predictor: LatentPredictor, settings: WmSettings
) -> np.ndarray:
    """Times `latency_repeats` live checks: encode one frame and predict one latent.

    Returns:
        The duration of each check in ms.
    """
    height, width = settings.input_size
    frame = np.zeros((1, height, width, 3), dtype=np.uint8)
    window = settings.history + settings.horizon
    latents = np.zeros((window, predictor.network[-1].out_features), dtype=np.float32)
    actions = np.zeros((window, predictor.action_mean.shape[0]), dtype=np.float32)
    durations = []
    for _ in range(settings.latency_repeats):
        start = time.perf_counter()
        encode_frames(encoder, frame, 1)
        score_episode(predictor, latents, actions, settings)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        durations.append(1000.0 * (time.perf_counter() - start))
    return np.array(durations)


def save_predictor(path: Path, predictor: LatentPredictor) -> None:
    """Writes the predictor weights and its standardization buffers."""
    torch.save(predictor.state_dict(), path)


def load_predictor(
    path: Path, latent_dim: int, action_dim: int, settings: WmSettings
) -> LatentPredictor:
    """Reads a predictor written by `save_predictor`, in eval mode on `device()`."""
    state = torch.load(path, map_location=device())
    predictor = LatentPredictor(
        latent_dim, action_dim, settings, state["action_mean"], state["action_std"]
    )
    predictor.load_state_dict(state)
    return predictor.to(device()).eval()
