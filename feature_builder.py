"""Compatibility import for existing serialized model artifacts.

Historical joblib pipelines reference ``feature_builder.FeatureBuilder``.
Keep this shim so old artifacts can still be loaded after moving the
implementation into ``training.features``.
"""

from training.features.feature_builder import FeatureBuilder

__all__ = ["FeatureBuilder"]
