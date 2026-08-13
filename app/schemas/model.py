from pydantic import BaseModel, Field


class ConfusionMatrixData(BaseModel):
    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int


class MetricData(BaseModel):
    accuracy: float = Field(ge=0, le=1)
    precision: float = Field(ge=0, le=1)
    recall: float = Field(ge=0, le=1)
    f1: float = Field(ge=0, le=1)
    roc_auc: float = Field(ge=0, le=1)
    brier_score: float = Field(ge=0)
    log_loss: float = Field(ge=0)


class CandidateMetricData(MetricData):
    model_name: str


class CalibrationBinData(BaseModel):
    mean_predicted_probability: float = Field(ge=0, le=1)
    fraction_positive: float = Field(ge=0, le=1)


class CalibrationData(BaseModel):
    brier_score: float = Field(ge=0)
    log_loss: float = Field(ge=0)
    bins: list[CalibrationBinData]


class ThresholdStrategyMetricData(BaseModel):
    strategy: str
    threshold: float = Field(ge=0, le=1)
    precision: float = Field(ge=0, le=1)
    recall: float = Field(ge=0, le=1)
    f1: float = Field(ge=0, le=1)
    false_positive: int
    false_negative: int
    business_cost: float = Field(ge=0)


class ThresholdAnalysisData(BaseModel):
    default_threshold: float = Field(ge=0, le=1)
    f1_optimal_threshold: float = Field(ge=0, le=1)
    business_cost_threshold: float = Field(ge=0, le=1)
    selected_threshold: float = Field(ge=0, le=1)
    selected_strategy: str
    false_positive_cost: float = Field(ge=0)
    false_negative_cost: float = Field(ge=0)
    strategies: list[ThresholdStrategyMetricData]


class MetricsData(BaseModel):
    selected_model: str
    selection_metric: str
    selection_split: str
    dataset_rows: int
    training_rows: int
    validation_rows: int
    test_rows: int
    positive_rate: float = Field(ge=0, le=1)
    threshold: float = Field(ge=0, le=1)
    metrics: MetricData
    confusion_matrix: ConfusionMatrixData
    calibration: CalibrationData
    threshold_analysis: ThresholdAnalysisData
    candidates: list[CandidateMetricData]


class FeatureWeight(BaseModel):
    feature: str
    coefficient: float


class ModelInfoData(BaseModel):
    model_version: str
    schema_version: str
    algorithm: str
    selected_model: str
    trained_at_utc: str
    dataset_rows: int
    training_rows: int
    validation_rows: int
    test_rows: int
    target: str
    threshold: float
    threshold_strategy: str
    numeric_features: list[str]
    categorical_features: list[str]
    top_positive_features: list[FeatureWeight]
    top_negative_features: list[FeatureWeight]
    library_versions: dict[str, str]


class ModelManifestData(BaseModel):
    schema_version: str
    model_version: str
    algorithm: str
    selected_model: str
    trained_at_utc: str
    model_sha256: str
    dataset_sha256: str
    feature_contract: list[str]
    feature_count: int = Field(ge=1)
    target: str
    threshold: float = Field(ge=0, le=1)
    threshold_strategy: str
    library_versions: dict[str, str]
