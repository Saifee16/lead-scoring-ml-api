# Generated ML Artifacts

Run:

```bash
python -m ml.train
```

The training pipeline generates:

- `model.joblib` — trusted local scikit-learn pipeline bundle; ignored by Git.
- `metrics.json` — held-out test metrics, calibration diagnostics, and threshold policy.
- `model_manifest.json` — artifact contract and SHA-256 integrity metadata.
- `confusion_matrix.png` — classification errors at the selected threshold.
- `calibration_curve.png` — reliability diagram for predicted probabilities.
- `threshold_analysis.png` — precision/recall/F1 and normalized business-cost trade-off.

`ModelService` validates the model SHA-256 against `model_manifest.json` before calling
`joblib.load`. Do not load model artifacts from untrusted sources.
