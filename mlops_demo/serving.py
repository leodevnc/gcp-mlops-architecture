from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .schemas import DeliveryFeatures, LinearModel


class BatchRequest(BaseModel):
    instances: list[DeliveryFeatures] = Field(min_length=1)


def create_app(model: LinearModel, *, vertex: bool = False) -> FastAPI:
    if vertex:
        return create_batch_app(model, health_path="/health", prediction_path="/predict")
    app = _prediction_app("/health")

    @app.post("/predict")
    def predict(features: DeliveryFeatures) -> dict[str, float]:
        return {"delivery_minutes": model.predict(features)}

    return app


def create_batch_app(model: LinearModel, *, health_path: str, prediction_path: str) -> FastAPI:
    app = _prediction_app(health_path)

    @app.post(prediction_path)
    def predict_batch(request: BatchRequest) -> dict[str, list[dict[str, float]]]:
        return {
            "predictions": [
                {"delivery_minutes": model.predict(features)} for features in request.instances
            ]
        }

    return app


def _prediction_app(health_path: str) -> FastAPI:
    app = FastAPI(title="Delivery Time Prediction")

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_request: Request, exc: RequestValidationError) -> JSONResponse:
        # A number beyond the float range can reach validation as infinity.
        # Return the error without echoing the raw input in the JSON response.
        details = [
            {"loc": issue["loc"], "msg": issue["msg"], "type": issue["type"]}
            for issue in exc.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": details})

    @app.get(health_path)
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app
