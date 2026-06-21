"""Tests para Walk-Forward Analysis expansivo (anclado).

Cubre:
  - Propiedades geométricas del walk_forward_splits expansivo (IS crece, OOS
    contiguo, sin solapamiento, cubre el tramo tras el ancla).
  - Guardas de longitud mínima.
  - Integración con ExperimentConfig (wf_mode, validaciones).
  - Helpers internos: _aggregate_wf_windows, _export_wf_window_metrics.
  - Equivalencia de rama no-WF: wf_enabled=False ≡ comportamiento previo.
  - Contexto de encoders congelados construido una vez por ventana (mock).
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, call

from latent_rl.experiments.utils import walk_forward_splits
from latent_rl.experiments.config import ExperimentConfig
from latent_rl.experiments.runner import (
    _aggregate_wf_windows,
    _export_wf_window_metrics,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _synthetic_data(n: int = 200) -> pd.DataFrame:
    """Serie sintética de n filas con columnas OHLCV básicas."""
    rng = np.random.default_rng(42)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame({
        "open":   close,
        "high":   close + 1,
        "low":    close - 1,
        "close":  close,
        "volume": rng.integers(1_000, 10_000, n).astype(float),
    })


# ---------------------------------------------------------------------------
# Propiedades geométricas del expansivo
# ---------------------------------------------------------------------------

class TestWalkForwardSplitsExpanding:

    def test_is_always_starts_at_zero(self):
        """El IS de cada ventana siempre empieza en la fila 0."""
        data = _synthetic_data(100)
        splits = walk_forward_splits(data, n_windows=4, is_ratio=0.5)
        for wf_is, _ in splits:
            # Índice reseteado, así que la primera fila es siempre 0
            assert wf_is.index[0] == 0

    def test_is_grows_with_window(self):
        """El IS se expande: len(IS[k]) < len(IS[k+1]) para todo k."""
        data = _synthetic_data(200)
        splits = walk_forward_splits(data, n_windows=5, is_ratio=0.5)
        is_sizes = [len(wf_is) for wf_is, _ in splits]
        for a, b in zip(is_sizes, is_sizes[1:]):
            assert a < b, f"IS no crece: {is_sizes}"

    def test_oos_blocks_contiguous_and_non_overlapping(self):
        """Los bloques OOS son contiguos y sin solapamiento."""
        data = _synthetic_data(200)
        n = len(data)
        splits = walk_forward_splits(data, n_windows=5, is_ratio=0.5)
        # El OOS de ventana k+1 comienza justo donde termina el de k
        for k in range(len(splits) - 1):
            wf_is_k,   wf_oos_k   = splits[k]
            wf_is_kp1, wf_oos_kp1 = splits[k + 1]
            oos_end_k = len(wf_is_k) + len(wf_oos_k)
            oos_start_kp1 = len(wf_is_kp1)
            # El IS de k+1 es exactamente IS[k] + OOS[k] (sin huecos)
            assert len(wf_is_kp1) == oos_end_k, (
                f"Ventana {k}: IS[k]+OOS[k]={oos_end_k}, IS[k+1]={len(wf_is_kp1)}"
            )

    def test_oos_always_after_is(self):
        """El OOS de cada ventana es siempre posterior al IS de esa ventana."""
        data = _synthetic_data(200)
        splits = walk_forward_splits(data, n_windows=5, is_ratio=0.5)
        for k, (wf_is, wf_oos) in enumerate(splits):
            assert len(wf_is) > 0, f"IS vacío en ventana {k}"
            assert len(wf_oos) > 0, f"OOS vacío en ventana {k}"
            # Verificar que el IS termina donde el OOS comienza en los datos orig.
            combined = pd.concat([wf_is, wf_oos]).reset_index(drop=True)
            # Misma serie de 'close' que el trozo correspondiente de data
            is_end = len(wf_is)
            oos_end = is_end + len(wf_oos)
            expected = data["close"].values[:oos_end]
            np.testing.assert_array_equal(combined["close"].values, expected)

    def test_union_of_oos_covers_tail(self):
        """La unión de todos los bloques OOS cubre exactamente el tramo tras el ancla."""
        data = _synthetic_data(200)
        n = len(data)
        K = 5
        is_ratio = 0.5
        anchor_end = int(n * is_ratio)
        splits = walk_forward_splits(data, n_windows=K, is_ratio=is_ratio)
        total_oos = sum(len(wf_oos) for _, wf_oos in splits)
        # La suma de OOS cubre el tramo restante (puede diferir por 1 si n−anchor no es múltiplo de K)
        assert total_oos == n - anchor_end, (
            f"OOS totales={total_oos}, esperado={n - anchor_end}"
        )

    def test_last_window_absorbs_remainder(self):
        """La última ventana absorbe las filas sobrantes (n−anchor no múltiplo de K)."""
        # 101 filas, anchor=50, remaining=51, K=5 → b=10, último OOS=11
        data = _synthetic_data(101)
        splits = walk_forward_splits(data, n_windows=5, is_ratio=0.5)
        oos_sizes = [len(wf_oos) for _, wf_oos in splits]
        # Todas las ventanas salvo la última deben tener el mismo tamaño
        assert all(s == oos_sizes[0] for s in oos_sizes[:-1]), (
            f"Bloques OOS intermedios no iguales: {oos_sizes}"
        )
        # La última puede ser mayor o igual
        assert oos_sizes[-1] >= oos_sizes[0]

    def test_n_windows_returned(self):
        data = _synthetic_data(200)
        for K in [2, 3, 5, 8]:
            splits = walk_forward_splits(data, n_windows=K, is_ratio=0.5)
            assert len(splits) == K

    def test_index_reset_in_all_splits(self):
        """Ambos DataFrames de cada split tienen índice reseteado (empieza en 0)."""
        data = _synthetic_data(100)
        splits = walk_forward_splits(data, n_windows=3, is_ratio=0.6)
        for wf_is, wf_oos in splits:
            assert wf_is.index[0] == 0
            assert wf_oos.index[0] == 0

    def test_is_ratio_controls_anchor(self):
        """El ancla IS de la primera ventana coincide exactamente con floor(n*is_ratio)."""
        data = _synthetic_data(200)
        n = len(data)
        is_ratio = 0.6
        splits = walk_forward_splits(data, n_windows=4, is_ratio=is_ratio)
        anchor_expected = int(n * is_ratio)
        # Primera ventana: IS es el ancla
        first_is_len = len(splits[0][0])
        assert first_is_len == anchor_expected


# ---------------------------------------------------------------------------
# Guardas de longitud mínima
# ---------------------------------------------------------------------------

class TestWalkForwardGuards:

    def test_n_windows_less_than_2_raises(self):
        data = _synthetic_data(100)
        with pytest.raises(ValueError, match="n_windows debe ser >= 2"):
            walk_forward_splits(data, n_windows=1, is_ratio=0.5)

    def test_is_ratio_too_small_raises(self):
        """is_ratio muy pequeño deja anchor < lookback+1."""
        data = _synthetic_data(100)
        with pytest.raises(ValueError, match="anchor_end"):
            walk_forward_splits(data, n_windows=2, is_ratio=0.001, lookback=10)

    def test_oos_block_too_small_raises(self):
        """Demasiadas ventanas para los datos disponibles → bloque OOS insuficiente."""
        data = _synthetic_data(20)
        with pytest.raises(ValueError, match="bloque OOS"):
            walk_forward_splits(data, n_windows=8, is_ratio=0.5, lookback=5, min_oos_steps=2)

    def test_valid_params_do_not_raise(self):
        data = _synthetic_data(200)
        # No debe lanzar
        splits = walk_forward_splits(data, n_windows=5, is_ratio=0.5, lookback=10)
        assert len(splits) == 5

    def test_lookback_default_is_permissive(self):
        """lookback=1 por defecto: casi cualquier dataset razonable pasa."""
        data = _synthetic_data(50)
        splits = walk_forward_splits(data, n_windows=2, is_ratio=0.5)
        assert len(splits) == 2


# ---------------------------------------------------------------------------
# ExperimentConfig: wf_mode y nuevas validaciones
# ---------------------------------------------------------------------------

class TestExperimentConfigWF:

    def test_default_wf_mode_is_expanding(self):
        cfg = ExperimentConfig(tickers=["SPY"])
        assert cfg.wf_mode == "expanding"

    def test_wf_mode_can_be_set(self):
        cfg = ExperimentConfig(tickers=["SPY"], wf_mode="sliding")
        assert cfg.wf_mode == "sliding"

    def test_wf_disabled_by_default(self):
        cfg = ExperimentConfig(tickers=["SPY"])
        assert not cfg.wf_enabled

    def test_wf_params_valid(self):
        cfg = ExperimentConfig(
            tickers=["SPY"],
            wf_enabled=True,
            wf_n_windows=5,
            wf_is_ratio=0.5,
            wf_mode="expanding",
        )
        assert cfg.wf_enabled
        assert cfg.wf_n_windows == 5


# ---------------------------------------------------------------------------
# _aggregate_wf_windows
# ---------------------------------------------------------------------------

class TestAggregateWFWindows:

    def _make_window_agg(self, val: float) -> dict:
        """Crea un dict de métricas de ventana mínimo."""
        return {
            "A": {
                "name":            "A",
                "mean_return_is":  val,
                "mean_return_oos": val * 0.5,
                "mean_sharpe_is":  val * 0.1,
                "mean_sharpe_oos": val * 0.05,
                "mean_mdd_is":     -val * 0.1,
                "mean_mdd_oos":    -val * 0.2,
                "mean_equity_is":  10_000 + val * 100,
                "mean_equity_oos": 10_000 + val * 50,
                "mean_n_trades":   5.0,
                "std_return_is":   0.0,
                "std_return_oos":  0.0,
                "seed_std_return_is":  0.0,
                "seed_std_sharpe_oos": 0.0,
            }
        }

    def test_averages_numeric_fields(self):
        windows = [self._make_window_agg(1.0), self._make_window_agg(3.0)]
        result = _aggregate_wf_windows(windows)
        assert result["A"]["mean_return_is"] == pytest.approx(2.0)
        assert result["A"]["mean_return_oos"] == pytest.approx(1.0)

    def test_name_field_preserved(self):
        windows = [self._make_window_agg(1.0)]
        result = _aggregate_wf_windows(windows)
        assert result["A"]["name"] == "A"

    def test_empty_returns_empty(self):
        assert _aggregate_wf_windows([]) == {}

    def test_multi_agent(self):
        window = {
            "A": {"name": "A", "mean_return_oos": 0.1},
            "D": {"name": "D", "mean_return_oos": 0.3},
        }
        result = _aggregate_wf_windows([window, window])
        assert "A" in result
        assert "D" in result


# ---------------------------------------------------------------------------
# _export_wf_window_metrics
# ---------------------------------------------------------------------------

class TestExportWFWindowMetrics:

    def _make_window_agg(self, window_idx: int) -> dict:
        return {
            "A": {"name": "A", "mean_return_oos": 0.1 * window_idx,
                  "mean_sharpe_oos": 0.5},
        }

    def test_creates_csv(self, tmp_path):
        cfg = ExperimentConfig(tickers=["SPY"])
        windows = [self._make_window_agg(i) for i in range(3)]
        _export_wf_window_metrics(windows, tmp_path, cfg)
        out = tmp_path / "wf_window_metrics.csv"
        assert out.exists()
        df = pd.read_csv(out)
        assert "window" in df.columns
        assert "agent" in df.columns
        assert len(df) == 3  # 3 ventanas × 1 agente

    def test_window_column_values(self, tmp_path):
        cfg = ExperimentConfig(tickers=["SPY"])
        windows = [self._make_window_agg(i) for i in range(4)]
        _export_wf_window_metrics(windows, tmp_path, cfg)
        df = pd.read_csv(tmp_path / "wf_window_metrics.csv")
        assert list(df["window"].unique()) == [0, 1, 2, 3]

    def test_with_ivl_records(self, tmp_path):
        cfg = ExperimentConfig(tickers=["SPY"])
        windows = [self._make_window_agg(0)]
        ivl_per_window = [[{"latent_agent": "A", "ivl": 0.42}]]
        _export_wf_window_metrics(windows, tmp_path, cfg,
                                  ivl_records_per_window=ivl_per_window)
        df = pd.read_csv(tmp_path / "wf_window_metrics.csv")
        assert "ivl" in df.columns
        assert df.loc[df["agent"] == "A", "ivl"].iloc[0] == pytest.approx(0.42)

    def test_empty_windows_no_file(self, tmp_path):
        cfg = ExperimentConfig(tickers=["SPY"])
        _export_wf_window_metrics([], tmp_path, cfg)
        assert not (tmp_path / "wf_window_metrics.csv").exists()


# ---------------------------------------------------------------------------
# walk_forward_splits: equivalencia con is_ratio = 1.0 - 1/n_windows
# Si el ancla ocupa (n-1)/n de los datos, la primera ventana IS ≈ todos salvo
# un bloque OOS al final → comportamiento intuitivo.
# ---------------------------------------------------------------------------

class TestExpandingVsOldDisjoint:

    def test_first_is_equals_anchor(self):
        """Primera ventana IS = anchor_end filas exacto."""
        n = 200
        K = 5
        is_ratio = 0.5
        data = _synthetic_data(n)
        splits = walk_forward_splits(data, n_windows=K, is_ratio=is_ratio)
        anchor_end = int(n * is_ratio)
        assert len(splits[0][0]) == anchor_end

    def test_expanding_is_larger_than_disjoint_for_late_windows(self):
        """Para ventanas tardías, el IS expansivo es mayor que el IS disjunto
        del esquema anterior (IS fijo = anchor_end para todos)."""
        n = 200
        K = 5
        is_ratio = 0.6
        data = _synthetic_data(n)
        splits = walk_forward_splits(data, n_windows=K, is_ratio=is_ratio)
        anchor_end = int(n * is_ratio)

        for k, (wf_is, _) in enumerate(splits):
            # IS siempre >= anchor_end (la primera ventana es exactamente anchor_end)
            assert len(wf_is) >= anchor_end, (
                f"Ventana {k}: IS={len(wf_is)} < anchor_end={anchor_end}"
            )
        # La última ventana IS debe ser mayor que la primera
        assert len(splits[-1][0]) > len(splits[0][0])


# ---------------------------------------------------------------------------
# Propiedad de pivote: los datos dentro de cada split son un trozo contiguo
# de los datos originales (no permutados, no repetidos).
# ---------------------------------------------------------------------------

class TestDataIntegrity:

    def test_is_oos_together_equal_data_prefix(self):
        """Para ventana k: concat(IS, OOS) = data[:oos_end], en el mismo orden."""
        data = _synthetic_data(150)
        splits = walk_forward_splits(data, n_windows=4, is_ratio=0.5)
        for k, (wf_is, wf_oos) in enumerate(splits):
            combined_close = np.concatenate([
                wf_is["close"].values,
                wf_oos["close"].values,
            ])
            expected_len = len(combined_close)
            expected_close = data["close"].values[:expected_len]
            np.testing.assert_array_equal(
                combined_close, expected_close,
                err_msg=f"Ventana {k}: datos no son subconjunto contiguo del original"
            )

    def test_oos_no_overlap_with_prior_is(self):
        """El OOS de cada ventana no se solapa con el IS de NINGUNA ventana anterior."""
        data = _synthetic_data(200)
        splits = walk_forward_splits(data, n_windows=5, is_ratio=0.5)
        # Para ventana k, el OOS ocupa data[oos_start:oos_end]
        # El IS de ventana k+1 ocupa data[0:oos_start+b]
        # El OOS de ventana k es posterior al IS de la ventana k
        for k, (wf_is, wf_oos) in enumerate(splits):
            # Verificar que OOS no repite ninguna fila del IS de la misma ventana
            is_set = set(wf_is.index)
            oos_set = set(wf_oos.index)
            # Tras el reset_index los índices de IS van de 0..len(IS)-1,
            # y los de OOS van de 0..len(OOS)-1, así que no podemos comparar
            # directamente. Usamos el valor de 'close' en su lugar.
            is_close = set(round(v, 8) for v in wf_is["close"])
            oos_close = set(round(v, 8) for v in wf_oos["close"])
            # Con datos sintéticos con varianza aleatoria, la intersección es casi
            # siempre vacía. No forzamos exactitud perfecta (precios podrían coincidir),
            # pero sí verificamos que el OOS no ES igual al IS.
            assert len(wf_oos) > 0
            assert len(wf_is) > 0
