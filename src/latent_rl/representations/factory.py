"""Factory de encoders latentes."""

from typing import Any, Dict
from latent_rl.representations.base import LatentEncoder


def build_encoder(encoder_type: str, **kwargs: Any) -> LatentEncoder:
    """
    Construye un encoder latente por tipo y kwargs de arquitectura.

    Args:
        encoder_type: "mlp" | "tcn" | "gru"
        **kwargs: Argumentos de arquitectura específicos del tipo.
            - mlp: input_dim (o input_len + n_features), latent_dim, hidden_dims,
                   activation, dropout, input_len, n_features
            - tcn: input_len, n_features, latent_dim, kernel_size, dilations,
                   channels, activation, dropout
            - gru: input_len, n_features, latent_dim, hidden_dim, num_layers, dropout

    Returns:
        Encoder instanciado.

    Raises:
        ValueError: Si encoder_type no es reconocido.
    """
    etype = encoder_type.lower()

    if etype == "mlp":
        from latent_rl.representations.mlp_encoder import MLPLatentEncoder

        input_len: int = kwargs.get("input_len") or kwargs.get("L")
        n_features: int = kwargs.get("n_features") or kwargs.get("F")
        input_dim = kwargs.get("input_dim")
        if input_dim is None:
            if input_len is None or n_features is None:
                raise ValueError(
                    "mlp encoder requiere input_dim o (input_len, n_features)."
                )
            input_dim = input_len * n_features

        return MLPLatentEncoder(
            input_dim=input_dim,
            latent_dim=kwargs["latent_dim"],
            hidden_dims=kwargs.get("hidden_dims"),
            activation=kwargs.get("activation", "relu"),
            dropout=kwargs.get("dropout", 0.0),
            input_len=input_len,
            n_features=n_features,
        )

    if etype == "tcn":
        from latent_rl.representations.tcn_encoder import TCNLatentEncoder

        return TCNLatentEncoder(
            input_len=kwargs.get("input_len") or kwargs["L"],
            n_features=kwargs.get("n_features") or kwargs["F"],
            latent_dim=kwargs["latent_dim"],
            kernel_size=kwargs.get("kernel_size", 3),
            dilations=kwargs.get("dilations"),
            channels=kwargs.get("channels", 32),
            activation=kwargs.get("activation", "relu"),
            dropout=kwargs.get("dropout", 0.0),
        )

    if etype == "gru":
        from latent_rl.representations.gru_encoder import GRULatentEncoder

        return GRULatentEncoder(
            input_len=kwargs.get("input_len") or kwargs["L"],
            n_features=kwargs.get("n_features") or kwargs["F"],
            latent_dim=kwargs["latent_dim"],
            hidden_dim=kwargs.get("hidden_dim", 64),
            num_layers=kwargs.get("num_layers", 1),
            dropout=kwargs.get("dropout", 0.0),
        )

    raise ValueError(
        f"encoder_type desconocido: '{encoder_type}'. "
        "Disponibles: 'mlp', 'tcn', 'gru'."
    )


def encoder_kwargs_from_config(arch_config: Dict[str, Any]) -> Dict[str, Any]:
    """Extrae kwargs de arquitectura de un arch_config de artefacto."""
    cfg = dict(arch_config)
    cfg.pop("encoder_type", None)
    cfg.pop("feature_names", None)
    # Renombrar L/F a input_len/n_features si es necesario
    if "L" in cfg:
        cfg.setdefault("input_len", cfg.pop("L"))
    if "F" in cfg:
        cfg.setdefault("n_features", cfg.pop("F"))
    return cfg
