# Migration Report

## Original Structure

The repository originally kept most executable files at the root:

- `feature_builder.py`
- `train_pipeline.py`
- `test_pipeline.py`
- `page.py`
- root-level joblib artifacts
- `model/`
- `notebook/`
- `分析用/`
- `data/`
- `reports/`

## Final Structure

The repository now separates responsibilities:

```text
app/                       Streamlit runtime application
training/features/         Feature engineering
training/pipelines/        Training entry points
tests/unit/                Feature engineering tests
models/artifacts/          Serialized model artifacts
data/raw/                  Raw dataset and source documentation
data/processed/            Existing processed CSV export
notebooks/                 Exploratory notebooks
reports/analysis/          Existing evaluation images
specs/                     Product, requirements, architecture, ADRs, policies, Units
```

## Files Moved

| Original | New location |
|---|---|
| `page.py` | `app/page.py` |
| `feature_builder.py` | `training/features/feature_builder.py` |
| `train_pipeline.py` | `training/pipelines/train_pipeline.py` |
| `test_pipeline.py` | `tests/unit/test_pipeline.py` |
| `best_pipeline.joblib` | `models/artifacts/best_pipeline.joblib` |
| `test_pipeline.joblib` | `models/artifacts/test_pipeline.joblib` |
| `model/sba_xgboost_model.json` | `models/artifacts/sba_xgboost_model.json` |
| `notebook/20260203.ipynb` | `notebooks/20260203.ipynb` |
| `data/SBAnational.csv` | `data/raw/SBAnational.csv` |
| `data/*.pdf` | `data/raw/*.pdf` |
| `data.7z` | `data/raw/data.7z` |
| `reports/power_bi_data.csv` | `data/processed/power_bi_data.csv` |
| `分析用/*.png` | `reports/analysis/*.png` |
| `機器學習流程.txt` | `specs/architecture/legacy-ml-workflow-notes.txt` |
| `本份專案價值.txt` | `specs/product/legacy-project-value-notes.txt` |

## Files Renamed

No source modules were renamed semantically. The existing singular `notebook/` folder was normalized to `notebooks/`.

## Imports Changed

- Training code now imports `FeatureBuilder` from `training.features.feature_builder`.
- Tests now import `FeatureBuilder` from `training.features.feature_builder`.
- A root-level `feature_builder.py` compatibility shim remains so existing joblib artifacts referencing `feature_builder.FeatureBuilder` can still load.

## Configuration Changed

- `.github/workflows/check.yml` now points to relocated test, training, syntax-check, and artifact paths.
- `.gitignore` now also ignores `.pytest_cache/`.

## Path Changes

- Training reads raw data from `data/raw/SBAnational.csv`.
- Training writes artifacts to `models/artifacts/`.
- Streamlit loads `models/artifacts/best_pipeline.joblib`.

## Tests Executed

```text
python -m py_compile app/page.py feature_builder.py training/features/feature_builder.py training/pipelines/train_pipeline.py tests/unit/test_pipeline.py
python tests/unit/test_pipeline.py
python -c "import joblib; joblib.load('models/artifacts/best_pipeline.joblib')"
python training/pipelines/train_pipeline.py --mode=test
```

## Validation Results

- Syntax check: passed.
- Unit tests: passed, 3 tests.
- Existing `best_pipeline.joblib` load: passed.
- Training smoke test: passed and produced `models/artifacts/test_pipeline.joblib`.

## Known Issues

- Several legacy comments and console strings still contain mojibake from prior encoding issues.
- The notebook contains historical exploratory code and may include legacy hardcoded paths.
- Existing joblib artifacts depend on a historical module path, so `feature_builder.py` must remain as a compatibility shim until artifacts are intentionally regenerated or migrated.
- The training smoke test emits scikit-learn future warnings related to `LogisticRegressionCV`; these were not introduced by restructuring.

## Technical Debt Discovered

- Console output is verbose because existing estimators use `verbose=3`.
- Tests currently cover feature engineering only; model regression tests are not yet present.
- Streamlit input validation is basic and should be reviewed before any external deployment.

## Recommended Next Unit

`UNIT-001-model-packaging`: define model artifact metadata, compatibility strategy, and regression fixtures before any intentional artifact regeneration.
