"""Tests de configuracion para la run A/B/C/D robusta."""

from __future__ import annotations

import json

import pytest

from scripts.experiments.abcd_robust_h64 import (
    _preflight_heavy_encoder,
    _save_effective_config,
    build_config,
)


def test_full_config_matches_h64_robust_protocol(tmp_path):
    cfg = build_config(results_dir=str(tmp_path))
    assert cfg.run_arms == ["A", "B", "C", "D"]
    assert cfg.seeds == [0, 1, 2, 3, 4]
    assert cfg.n_training_episodes == 30
    assert cfg.dqn_hidden_dim == cfg.latent_q_hidden_dim == 64
    assert cfg.dqn_lr == cfg.latent_lr == 5e-4
    assert cfg.dqn_batch_size == cfg.latent_batch_size == 64
    assert cfg.random_start_train is True
    assert cfg.reward_mode == "log_return"
    assert cfg.use_internal_validation is True


def test_smoke_config_remains_aligned_after_overrides(tmp_path):
    cfg = build_config(smoke=True, results_dir=str(tmp_path))
    assert cfg.seeds == [0]
    assert cfg.n_training_episodes == 3
    assert cfg.n_eval_episodes == 1
    assert cfg.dqn_batch_size == cfg.latent_batch_size == 32
    assert cfg.dqn_buffer_capacity == cfg.latent_buffer_capacity == 500
    assert cfg.dqn_target_update == cfg.latent_target_update == 50
    assert cfg.dqn_epsilon_decay == cfg.latent_epsilon_decay == 0.99


def test_effective_config_is_structured_json(tmp_path):
    cfg = build_config(smoke=True, results_dir=str(tmp_path))
    path = _save_effective_config(cfg)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["ticker_configs"][0]["ticker"] == "BTC-USD"
    assert payload["run_arms"] == ["A", "B", "C", "D"]


def test_allow_missing_d_removes_it_from_effective_protocol(tmp_path):
    cfg = build_config(smoke=True, results_dir=str(tmp_path))
    cfg.heavy_encoder_path = str(tmp_path / "missing.pt")
    assert _preflight_heavy_encoder(cfg, allow_missing=True) is False
    assert cfg.run_arms == ["A", "B", "C"]
    assert cfg.latent_agents == ["B", "C"]


def test_missing_d_fails_before_expensive_run(tmp_path):
    cfg = build_config(smoke=True, results_dir=str(tmp_path))
    cfg.heavy_encoder_path = str(tmp_path / "missing.pt")
    with pytest.raises(RuntimeError, match="artefacto D"):
        _preflight_heavy_encoder(cfg)
