# ----------------------
# API Servisi için Dockerfile
# ----------------------
    FROM python:3.10-slim

    # Çalışma dizini
    WORKDIR /app
    
    # Gereksinimleri kopyala ve kur
    COPY requirements.txt .
    RUN pip install --no-cache-dir -r requirements.txt
    
    # Config ve API dosyalarını kopyala
    COPY configs/ ./configs/
    COPY api_service/ ./api_service/
    COPY model_pipeline/ ./model_pipeline/
    
    # Port aç
    EXPOSE 8000
    
    # Uygulama başlat
    CMD ["uvicorn","api_service.main:app","--host","0.0.0.0","--port","8000"]
    