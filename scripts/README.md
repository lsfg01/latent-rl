# Scripts del laboratorio

Los módulos de `scripts/experiments/` definen campañas reproducibles; la lógica
reutilizable permanece en `src/latent_rl/`. Ejecútalos desde la raíz con `python -m`.

| Campaña | Comando | Salida por defecto |
|---|---|---|
| Preentrenamiento del encoder | `python -m scripts.experiments.pretrain_encoder` | `models/encoders/tcn_heavy.pt` |
| Evaluación A/B/C/D | `python -m scripts.experiments.latent_abcd` | `results/latent_abcd/` |
| Comparación A vs D | `python -m scripts.experiments.a_vs_d` | `results/a_vs_d/` |
| Sweep de `hidden_dim` | `python -m scripts.experiments.a_hidden_sweep` | `results/a_hidden_sweep/` |
| Protocolo robusto h=64 | `python -m scripts.experiments.abcd_robust_h64` | `results/abcd_robust_h64/` |
| Baseline con datos reales | `python -m scripts.experiments.real_data_baseline` | `results/real_data_baseline/` |

`--smoke` escribe en `results/smoke/<campaña>/`; el preentrenamiento smoke usa
`models/smoke/encoders/tcn_heavy.pt`. Así las comprobaciones cortas no pisan artefactos
ni resultados reales. `--results-dir` permite sobrescribir la salida de cada campaña.

Utilidades:

```powershell
python -m scripts.utilities.generate_example_data
python -m scripts.utilities.compute_ivl --help
python dashboard\app.py --results-dir results/abcd_robust_h64
python scripts\export_memory_figures.py `
  --results-dir results/abcd_robust_h64 `
  --out-dir artifacts/memory_figures/ABCD_robust_h64
```

El exportador solo lee los CSV finales. Genera F1-F4 y un
`figure_manifest.json`; `--include-optional` añade F5/F6. Si Plotly no puede
usar Kaleido para PNG/SVG, conserva la ejecución y crea HTML interactivo como
fallback.
