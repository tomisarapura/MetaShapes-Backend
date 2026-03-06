import os
import boto3
from botocore.client import Config
import json
from dotenv import load_dotenv

load_dotenv()

# Credenciales y configuración que definimos en el docker-compose.yml
MINIO_URL = os.getenv("MINIO_URL")
MINIO_ACCESS_KEY = os.getenv("MINIO_ROOT_USER")
MINIO_SECRET_KEY = os.getenv("MINIO_ROOT_PASSWORD")
BUCKET_NAME = os.getenv("BUCKET_NAME")

# Creamos el cliente S3 apuntando a nuestro MinIO local
s3_client = boto3.client(
    's3',
    endpoint_url=MINIO_URL,
    aws_access_key_id=MINIO_ACCESS_KEY,
    aws_secret_access_key=MINIO_SECRET_KEY,
    config=Config(signature_version='s3v4')
)

def init_storage():
    """Verifica si el bucket existe. Si no, lo crea y le da permisos públicos de lectura."""
    try:
        s3_client.head_bucket(Bucket=BUCKET_NAME)
        print(f"✅ Bucket '{BUCKET_NAME}' listo y operativo.")
    except Exception:
        print(f"⚙️ Creando bucket '{BUCKET_NAME}'...")
        s3_client.create_bucket(Bucket=BUCKET_NAME)
        
        # Hacemos que los archivos sean de lectura pública para que el frontend pueda renderizarlos
        policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": "*",
                    "Action": ["s3:GetObject"],
                    "Resource": [f"arn:aws:s3:::{BUCKET_NAME}/*"]
                }
            ]
        }
        s3_client.put_bucket_policy(Bucket=BUCKET_NAME, Policy=json.dumps(policy))
        print(f"✅ Bucket '{BUCKET_NAME}' creado con políticas públicas.")

def upload_file(file_path: str, object_name: str) -> str:
    """Sube un archivo local a MinIO y devuelve su URL pública."""
    try:
        s3_client.upload_file(file_path, BUCKET_NAME, object_name)
        # Retorna la URL pública del archivo
        return f"{MINIO_URL}/{BUCKET_NAME}/{object_name}"
    except Exception as e:
        print(f"Error subiendo archivo: {e}")
        return None