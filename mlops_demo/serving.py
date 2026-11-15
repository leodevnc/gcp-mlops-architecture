from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .schemas import DeliveryFeatures, LinearModel


def create_app(model: LinearModel) -> FastAPI:
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

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/predict")
    def predict(features: DeliveryFeatures) -> dict[str, float]:
        return {"delivery_minutes": model.predict(features)}

    return app
