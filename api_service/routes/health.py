# api_service/routes/health.py
from fastapi import APIRouter
import yaml

router = APIRouter()

with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

MODEL_TYPE = config["model"]["type"]
MODEL_STAGE = config["deploy"]["model_stage"]

@router.get("/health", tags=["Health"])
def healthcheck():
    return {"status":"ok","model_type":MODEL_TYPE,"model_stage":MODEL_STAGE}
