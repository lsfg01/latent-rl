"""Configuración básica de la librería."""

from typing import Any, Dict, Optional


class Config:
    """Gestor de configuración simple."""

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        Inicializa la configuración.

        Args:
            config_dict: Diccionario con valores de configuración
        """
        self._config = config_dict or {}

    def get(self, key: str, default: Any = None) -> Any:
        """
        Obtiene un valor de configuración.

        Args:
            key: Clave de configuración
            default: Valor por defecto si la clave no existe

        Returns:
            Valor de configuración
        """
        return self._config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """
        Establece un valor de configuración.

        Args:
            key: Clave de configuración
            value: Valor a establecer
        """
        self._config[key] = value

    def update(self, config_dict: Dict[str, Any]) -> None:
        """
        Actualiza la configuración con un diccionario.

        Args:
            config_dict: Diccionario con valores de configuración
        """
        self._config.update(config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convierte la configuración a diccionario.

        Returns:
            Diccionario con la configuración
        """
        return self._config.copy()

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """
        Crea una configuración desde un diccionario.

        Args:
            config_dict: Diccionario con valores de configuración

        Returns:
            Instancia de Config
        """
        return cls(config_dict)

    def __repr__(self) -> str:
        """Representación string de la configuración."""
        return f"Config({self._config})"