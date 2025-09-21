# ingestion/data_ingestor.py
import pandas as pd
import requests
import psycopg2
import boto3
from io import StringIO, BytesIO

class DataIngestor:
    def __init__(self, config):
        self.config = config

    def fetch(self):
        source_type = self.config["data_source"]["type"]
        if source_type == "api":
            return self._fetch_from_api()
        elif source_type == "db":
            return self._fetch_from_db()
        elif source_type == "s3":
            return self._fetch_from_s3()
        elif source_type == "minio":
            return self._fetch_from_minio()
        elif source_type == "file":
            return self._fetch_from_file()
        else:
            raise ValueError(f"Bilinmeyen kaynak tipi: {source_type}")

    def _fetch_from_api(self):
        endpoint = self.config["data_source"]["endpoint"]
        headers = {"Authorization": f"Bearer {self.config['data_source']['api_key']}"}
        response = requests.get(endpoint, headers=headers)
        response.raise_for_status()
        data = response.json()
        return pd.DataFrame(data)

    def _fetch_from_db(self):
        db_cfg = self.config["db"]
        conn = psycopg2.connect(
            host=db_cfg["host"],
            port=db_cfg["port"],
            database=db_cfg["database"],
            user=db_cfg["user"],
            password=db_cfg["password"]
        )
        df = pd.read_sql_query(db_cfg["query"], conn)
        conn.close()
        return df

    def _fetch_from_s3(self):
        s3_cfg = self.config["s3"]
        s3 = boto3.client(
            "s3",
            endpoint_url=s3_cfg.get("endpoint_url"),  # AWS endpoint
            aws_access_key_id=s3_cfg["aws_access_key_id"],
            aws_secret_access_key=s3_cfg["aws_secret_access_key"]
        )
        return self._read_object(s3, s3_cfg["bucket_name"], s3_cfg["object_key"])

    def _fetch_from_minio(self):
        minio_cfg = self.config["minio"]
        s3 = boto3.client(
            "s3",
            endpoint_url=minio_cfg["endpoint_url"],  # MinIO endpoint
            aws_access_key_id=minio_cfg["access_key"],
            aws_secret_access_key=minio_cfg["secret_key"]
        )
        return self._read_object(s3, minio_cfg["bucket_name"], minio_cfg["object_key"])

    def _fetch_from_file(self):
        file_path = self.config["file"]["path"]
        if file_path.endswith(".csv"):
            return pd.read_csv(file_path)
        elif file_path.endswith(".xlsx"):
            return pd.read_excel(file_path)
        else:
            raise ValueError("Desteklenmeyen dosya formatı (local)")

    def _read_object(self, s3_client, bucket_name, object_key):
        """CSV veya Excel'i S3/MinIO'dan oku"""
        obj = s3_client.get_object(Bucket=bucket_name, Key=object_key)
        body = obj["Body"].read()
        if object_key.endswith(".csv"):
            return pd.read_csv(StringIO(body.decode()))
        elif object_key.endswith(".xlsx"):
            return pd.read_excel(BytesIO(body))
        else:
            raise ValueError("Desteklenmeyen dosya formatı (S3/MinIO)")
