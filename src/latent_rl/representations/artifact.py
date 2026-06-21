"""Sistema de artefactos para encoders preentrenados."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from latent_rl.representations.base import LatentEncoder
from latent_rl.representations.factory import build_encoder


def save_encoder_artifact(
    encoder: LatentEncoder,
    norm_stats: Optional[Dict[str, Any]],
    provenance: Dict[str, Any],
    path: str | Path,
) -> None:
    """
    Guarda un artefacto de encoder autocontenido.

    El artefacto incluye:
    - encoder_state_dict: pesos del encoder (sin cabezas auxiliares).
    - arch_config: tipo y kwargs de arquitectura (permite reconstruir el encoder).
    - norm_stats: estadísticas de normalización del corpus de entrenamiento.
    - provenance: metadatos de procedencia (universo, fechas, features, etc.).

    Args:
        encoder: Encoder entrenado.
        norm_stats: Dict con "mean" y "std" por feature (puede ser None).
        provenance: Metadatos de procedencia.
        path: Ruta de destino (.pt o .pth).
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "encoder_state_dict": encoder.state_dict(),
            "arch_config": encoder.get_arch_config(),
            "norm_stats": norm_stats,
            "provenance": provenance,
        },
        path,
    )


def load_encoder_artifact(
    path: str | Path,
    cfg=None,
) -> Tuple[LatentEncoder, Optional[Dict[str, Any]], Dict[str, Any]]:
    """
    Carga un artefacto de encoder y valida compatibilidad con cfg.

    Args:
        path: Ruta del artefacto.
        cfg: ExperimentConfig (opcional). Si se proporciona, valida que
             L, F y feature_names coincidan con cfg.lookback_window,
             len(cfg.features) y cfg.features.

    Returns:
        (encoder, norm_stats, provenance)

    Raises:
        ValueError: Si L, F o feature_names del artefacto no coinciden con cfg.
        FileNotFoundError: Si el archivo no existe.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Artefacto no encontrado: {path}")

    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    arch: Dict[str, Any] = ckpt["arch_config"]
    encoder_type: str = arch["encoder_type"]

    # Reconstruir encoder desde arch_config
    kwargs = {k: v for k, v in arch.items() if k not in ("encoder_type", "feature_names")}
    # Mapear L/F → input_len/n_features si es necesario
    if "L" in kwargs:
        kwargs.setdefault("input_len", kwargs.pop("L"))
    if "F" in kwargs:
        kwargs.setdefault("n_features", kwargs.pop("F"))
    encoder = build_encoder(encoder_type, **kwargs)
    encoder.load_state_dict(ckpt["encoder_state_dict"])

    # Validar compatibilidad si se pasa cfg
    if cfg is not None:
        _validate_artifact_compat(arch, cfg)

    return encoder, ckpt.get("norm_stats"), ckpt.get("provenance", {})


def _validate_artifact_compat(arch: Dict[str, Any], cfg) -> None:
    """Lanza excepción si arch_config es incompatible con cfg."""
    artifact_L = arch.get("L") or arch.get("input_len")
    artifact_F = arch.get("F") or arch.get("n_features")
    artifact_features: List[str] = arch.get("feature_names", [])

    errors: List[str] = []

    if artifact_L is not None and artifact_L != cfg.lookback_window:
        errors.append(
            f"L del artefacto ({artifact_L}) != cfg.lookback_window ({cfg.lookback_window})"
        )

    cfg_F = len(cfg.features) if cfg.features else None
    if artifact_F is not None and cfg_F is not None and artifact_F != cfg_F:
        errors.append(
            f"F del artefacto ({artifact_F}) != len(cfg.features) ({cfg_F})"
        )

    if artifact_features and cfg.features and artifact_features != list(cfg.features):
        errors.append(
            f"feature_names del artefacto {artifact_features} "
            f"!= cfg.features {list(cfg.features)}"
        )

    if errors:
        raise ValueError(
            "Artefacto incompatible con la configuración del experimento:\n"
            + "\n".join(f"  - {e}" for e in errors)
        )
