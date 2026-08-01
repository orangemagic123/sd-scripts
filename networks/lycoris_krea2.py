"""Krea 2 architecture adapter for :mod:`lycoris.kohya`.

LyCORIS' built-in kohya presets know about ``SingleStreamBlock``, but they do
not cover the standalone input, timestep, output, and text-fusion projections
in Krea 2.  This wrapper installs a Krea 2-specific preset and delegates the
actual adapter implementation to LyCORIS.
"""

from __future__ import annotations

from typing import Any

from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


KREA2_LYCORIS_PRESET: dict[str, Any] = {
    "enable_conv": False,
    # Targeting the root SingleStreamDiT would add an extra underscore to
    # LyCORIS names (``lora_unet__first``).  Structural child modules plus the
    # six root-level Linear names cover the same layers as lora_krea2 while
    # retaining the conventional ``lora_unet_<model key>`` naming scheme.
    "unet_target_module": ["SingleStreamBlock", "TextFusionTransformer", "LastLayer"],
    "unet_target_name": [
        r"^first$",
        r"^tmlp\.0$",
        r"^tmlp\.2$",
        r"^txtmlp\.1$",
        r"^txtmlp\.3$",
        r"^tproj\.1$",
    ],
    "text_encoder_target_module": [],
    "text_encoder_target_name": [],
}

_KREA2_PRESET_NAME = "krea2"
_EMPTY_PRESET: dict[str, Any] = {
    "enable_conv": False,
    "unet_target_module": [],
    "unet_target_name": [],
    "text_encoder_target_module": [],
    "text_encoder_target_name": [],
}


def _load_lycoris_kohya():
    try:
        from lycoris import kohya as lycoris_kohya
    except ImportError as exc:
        raise ImportError(
            "Krea 2 LyCORIS training requires lycoris-lora>=3.4.0. "
            "Install the repository requirements or run: "
            "python -m pip install 'lycoris-lora>=3.4.0'"
        ) from exc

    if not hasattr(lycoris_kohya, "PRESET") or not hasattr(lycoris_kohya, "LycorisNetworkKohya"):
        raise ImportError("The installed lycoris-lora is too old for Krea 2; install lycoris-lora>=3.4.0")
    return lycoris_kohya


def _prepare_network_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    kwargs = dict(kwargs)
    requested_preset = kwargs.get("preset")
    if requested_preset not in (None, _KREA2_PRESET_NAME):
        logger.warning(
            "Ignoring LyCORIS preset=%s for Krea 2; using the built-in Krea 2 target preset",
            requested_preset,
        )

    if str(kwargs.get("train_norm", "false")).lower() in ("1", "true", "yes", "on"):
        raise ValueError("Krea 2 LyCORIS does not support train_norm because Krea 2 uses a custom RMSNorm")

    # T-LoRA needs the trainer to update a timestep mask for every batch.  The
    # Krea 2 trainer does not provide that hook yet, so fail instead of silently
    # training a time-independent or stale mask.
    if str(kwargs.get("algo", "lora")).lower() == "tlora":
        raise ValueError("Krea 2 LyCORIS does not currently support algo=tlora")

    kwargs["preset"] = _KREA2_PRESET_NAME
    return kwargs


def create_network(
    multiplier,
    network_dim,
    network_alpha,
    vae,
    text_encoder,
    unet,
    **kwargs,
):
    """Create a LyCORIS network that covers every Krea 2 DiT Linear layer."""

    lycoris_kohya = _load_lycoris_kohya()
    lycoris_kohya.PRESET[_KREA2_PRESET_NAME] = KREA2_LYCORIS_PRESET
    return lycoris_kohya.create_network(
        multiplier,
        network_dim,
        network_alpha,
        vae,
        [],  # Qwen3-VL is cached, moved to meta, and never adapted.
        unet,
        **_prepare_network_kwargs(kwargs),
    )


def create_network_from_weights(
    multiplier,
    file,
    vae,
    text_encoder,
    unet,
    weights_sd=None,
    for_inference=False,
    **kwargs,
):
    """Restore any LyCORIS algorithm from its saved Krea 2 state dict."""

    lycoris_kohya = _load_lycoris_kohya()
    # LyCORIS reconstructs modules directly from the state dict, but first
    # instantiates an otherwise unused network.  Empty the class targets for
    # that temporary instance to avoid allocating a second adapter set for the
    # 14B-parameter Krea 2 model.
    lycoris_kohya.LycorisNetworkKohya.apply_preset(_EMPTY_PRESET)
    return lycoris_kohya.create_network_from_weights(
        multiplier,
        file,
        vae,
        [],  # Krea 2 checkpoints cannot contain trainable text-encoder adapters.
        unet,
        weights_sd=weights_sd,
        for_inference=for_inference,
        **kwargs,
    )
