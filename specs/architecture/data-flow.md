# Data Flow

```text
data/raw/SBAnational.csv
  -> training/features/FeatureBuilder
  -> training/pipelines/train_pipeline.py
  -> models/artifacts/*.joblib
  -> app/page.py
  -> prediction output
```

`data/processed/power_bi_data.csv` is an existing processed export used for analysis or reporting.

`reports/analysis/` contains existing model evaluation images.
