# scripts/run_pipeline.py
import yaml
from ingestion.data_ingestor import DataIngestor
from synthetic_data.merge_synthetic import add_synthetic_data
from model_pipeline.train_text import train_model

#Config Yükle
with open("configs/config.yaml","r") as f:
    config = yaml.safe_load(f)

#Veri çek (config’te hangi kaynak seçiliyse)
ingestor = DataIngestor(config)
df = ingestor.fetch()

#Sentetik veri ekle (Faker veya Gemini)
df = add_synthetic_data(
    df,
    method="faker",
    n_rows=config["synthetic"]["faker_rows"]
)

#Model eğit
model = train_model(
    df,
    model_type=config["model"]["type"],
    params=config["model"]["params"],
    target_col="target"  # kendi hedef kolonun ismi
)

print("✅ Pipeline tamamlandı. Model eğitildi ve MLflow’a kaydedildi.")
