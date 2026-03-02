import requests
import subprocess
import os
import re
import uuid
import storage
from database import SessionLocal
import models
from dotenv import load_dotenv

# Cargamos las variables de entorno
load_dotenv()

# --- CONFIGURACIÓN ---
# Busca la URL en el .env, si no la encuentra usa localhost por defecto
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/generate") 
MODEL_NAME = "deepseek-coder-v2"

# Prompt del sistema para forzar a la IA a actuar como un compilador
SYSTEM_PROMPT = """
Eres un experto en diseño paramétrico 3D usando OpenSCAD. 
Tu única tarea es traducir las descripciones de los usuarios a código OpenSCAD válido, eficiente y bien estructurado.
Usa variables paramétricas al principio del script.
REGLA ESTRICTA: Devuelve ÚNICAMENTE el código OpenSCAD. No incluyas explicaciones, ni introducciones, ni saludos.
"""

def extract_code(response_text: str) -> str:
    """Limpia la respuesta de la IA por si incluyó formato Markdown."""
    # Busca código dentro de bloques ```openscad ... ``` o ``` ... ```
    match = re.search(r'```(?:openscad)?(.*?)```', response_text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return response_text.strip()

def process_3d_generation(job_id: str, prompt: str):
    """
    Esta es la tarea pesada que correrá en segundo plano.
    """
    db = SessionLocal()
    job = db.query(models.GenerationJob).filter(models.GenerationJob.id == job_id).first()
    
    if not job:
        db.close()
        return

    try:
        # Actualizamos estado a procesando
        job.status = "processing"
        db.commit()

        # 1. Llamar a la IA
        print(f"[{job_id}] Enviando prompt a la IA...")
        payload = {
            "model": MODEL_NAME,
            "system": SYSTEM_PROMPT,
            "prompt": prompt,
            "stream": False
        }
        
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        
        raw_output = response.json().get("response", "")
        scad_code = extract_code(raw_output)
        
        # 2. Guardar el código temporalmente
        scad_filename = f"{job_id}.scad"
        stl_filename = f"{job_id}.stl"
        
        with open(scad_filename, "w", encoding="utf-8") as f:
            f.write(scad_code)
            
        # 3. Compilar usando OpenSCAD
        print(f"[{job_id}] Compilando código OpenSCAD a STL...")
        # Ejecuta el comando: openscad -o archivo.stl archivo.scad
        subprocess.run(
            ["wsl", "openscad", "-o", stl_filename, scad_filename], 
            check=True, 
            capture_output=True
        )
        
        # 4. Subir a MinIO
        print(f"[{job_id}] Subiendo modelo a almacenamiento local...")
        file_url = storage.upload_file(stl_filename, stl_filename)
        
        if not file_url:
            raise Exception("Error al subir el archivo a MinIO")

        # 5. Actualizar la base de datos con éxito
        job.status = "completed"
        job.file_url = file_url
        db.commit()
        print(f"✅ [{job_id}] Generación completada con éxito!")

    except subprocess.CalledProcessError as e:
        print(f"❌ Error compilando OpenSCAD: {e.stderr.decode()}")
        job.status = "failed"
        db.commit()
    except Exception as e:
        print(f"❌ Error general en proceso: {e}")
        job.status = "failed"
        db.commit()
    finally:
        db.close()
        # Limpiamos los archivos temporales
        if os.path.exists(scad_filename):
            os.remove(scad_filename)
        if os.path.exists(stl_filename):
            os.remove(stl_filename)