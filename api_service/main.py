# api_service/main.py
from fastapi import FastAPI
from api_service.routes import predict, health

app = FastAPI(
    title="Prediction & Forecast API",
    description="RandomForest / LinearRegression / LSTM Forecast API",
    version="1.0.0",
    docs_url="/docs",       # Swagger UI adresi (varsayılan /docs)
    redoc_url="/redoc"       # Redoc adresi (varsayılan /redoc)
)


# Router’ları include et
app.include_router(predict.router)
app.include_router(health.router)

# Swagger UI otomatik /docs adresinde
# ReDoc otomatik /redoc adresinde

# Opsiyonel root endpoint
@app.get("/")
def root():
    return {"message":"Prediction & Forecast API is running"}
