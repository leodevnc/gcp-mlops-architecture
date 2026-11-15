from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat


class DeliveryFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    distance_km: Annotated[float, Field(strict=True, gt=0, le=100)]
    package_count: Annotated[int, Field(strict=True, ge=1, le=20)]

    def vector(self) -> tuple[float, int]:
        return self.distance_km, self.package_count


class DeliveryRecord(DeliveryFeatures):
    delivery_minutes: Annotated[float, Field(strict=True, gt=0, le=1440)]


class PreparedData(BaseModel):
    format_version: Literal[1] = 1
    source_sha256: str
    split_seed: int
    train: list[DeliveryRecord] = Field(min_length=3)
    holdout: list[DeliveryRecord] = Field(min_length=1)


class LinearModel(BaseModel):
    format_version: Literal[1] = 1
    prepared_sha256: str
    intercept: FiniteFloat
    distance_coefficient: FiniteFloat
    package_coefficient: FiniteFloat
    baseline_minutes: FiniteFloat

    def predict(self, features: DeliveryFeatures) -> float:
        distance_km, package_count = features.vector()
        return (
            self.intercept
            + self.distance_coefficient * distance_km
            + self.package_coefficient * package_count
        )


class Evaluation(BaseModel):
    format_version: Literal[1] = 1
    model_sha256: str
    holdout_rows: int
    mae: FiniteFloat
    baseline_mae: FiniteFloat
    max_mae: FiniteFloat
    passed: bool
