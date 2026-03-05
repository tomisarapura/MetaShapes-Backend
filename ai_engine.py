import requests
import subprocess
import os
import re
import ast
import sys
from datetime import datetime
from botocore.exceptions import ClientError
import storage
from database import SessionLocal
import models
from dotenv import load_dotenv

# Cargamos las variables de entorno
load_dotenv()

# --- CONFIGURACIÓN ---
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
MODEL_NAME = "qwen2.5-coder:14b"

# --- SYSTEM PROMPTS (PIPELINE DE 3 PASOS) ---
SYSTEM_PROMPT_STEP_1 = """
Eres un diseñador industrial experto. 
Tu tarea es tomar la idea base de un usuario y expandirla en lenguaje natural detallado. 
Describe su función principal, las partes exactas que lo componen, sus proporciones generales y cómo interactúan las piezas entre sí.
"""

SYSTEM_PROMPT_STEP_2 = """
Eres un analista geométrico experto. 
Toma la descripción del diseño industrial proporcionada y tradúcela ESTRICTAMENTE a una lista técnica estructurada de primitivas geométricas 3D y operaciones booleanas.

REQUISITOS OBLIGATORIOS:
- Para cada parte, especifica dimensiones numéricas en milímetros (X, Y, Z o radios/alturas).
- Si la descripción no incluye medidas, infiere valores realistas basados en el contexto.
- Incluye traslaciones espaciales, rotaciones y operaciones booleanas (unión, corte/diferencia, intersección).
- Estructura las dependencias: qué pieza se construye sobre qué cara o plano de la pieza anterior.
- No escribas código, solo la planificación matemática y geométrica.
"""

SYSTEM_PROMPT_STEP_3 = """
Eres un experto en diseño paramétrico 3D e ingeniería inversa usando Python y la librería CadQuery. 
Tu única tarea es traducir la especificación geométrica recibida a un script de Python válido, eficiente y bien estructurado utilizando CadQuery (`import cadquery as cq`).

REGLAS ESTRICTAS:
1. Importa SOLAMENTE `cadquery as cq`. ESTÁ ESTRICTAMENTE PROHIBIDO importar `os`, `sys`, `subprocess`, o usar funciones como `open()`, `eval()`, `exec()`.
2. Construye el modelo usando la API fluida de CadQuery y el árbol CSG (ej. `cq.Workplane("XY").box(...)`).
3. Asigna el modelo final y definitivo a una variable llamada exactamente `result`.
4. Evita bucles infinitos, recursión profunda o lógicas matemáticas extremadamente complejas. Mantén la geometría sólida y directa.
5. Al final del script, DEBES exportar `result` a los formatos STEP y STL usando EXACTAMENTE los nombres de archivo que se te indiquen en el prompt del usuario.
6. Devuelve ÚNICAMENTE el código Python dentro de un bloque ```python ... ```. No incluyas explicaciones ni saludos.

--- EJEMPLO ONE-SHOT ---
Entrada:
"Cilindro base de 20 mm de radio y 5 mm de alto, con un agujero pasante de 5 mm de radio en el centro. Guardar como 'modelo1.step' y 'modelo1.stl'."

Salida:
```python
import cadquery as cq

# Parámetros
radio_base = 20
altura_base = 5
radio_agujero = 5

# Modelo: Base cilíndrica con agujero
result = (
    cq.Workplane("XY")
    .cylinder(altura_base, radio_base)
    .faces(">Z")
    .hole(radio_agujero * 2)  # hole() toma el diámetro
)

# Exportar en los formatos solicitados
cq.exporters.export(result, "modelo1.step")
cq.exporters.export(result, "modelo1.stl")
"""

def extract_code(response_text: str) -> str:
    """Limpia la respuesta de la IA para extraer el código Python."""
    match = re.search(r"```python\n(.*?)```", response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response_text.strip()

def is_safe_python_code(code_str: str) -> bool:
    """
    Analiza el AST (Abstract Syntax Tree) para bloquear importaciones
    maliciosas y el uso de funciones integradas peligrosas.
    """
    forbidden_modules = {'os', 'sys', 'subprocess', 'shutil', 'socket', 'requests', 'pathlib'}
    forbidden_functions = {'eval', 'exec', 'open', 'compile', 'import'}

    try:
        tree = ast.parse(code_str)
    except SyntaxError as e:
        print(f"Error de sintaxis en el código generado por IA: {e}")
        return False

    for node in ast.walk(tree):
        # Verifica imports simples (ej: import os)
        if isinstance(node, ast.Import):
            for alias in node.names:
                base_module = alias.name.split('.')[0]
                if base_module in forbidden_modules:
                    print(f"Bloqueo de seguridad: Importación prohibida detectada ({alias.name})")
                    return False

        # Verifica imports desde módulos (ej: from os import system)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                base_module = node.module.split('.')[0]
                if base_module in forbidden_modules:
                    print(f"Bloqueo de seguridad: Importación 'from' prohibida ({node.module})")
                    return False

        # Verifica llamadas a funciones peligrosas
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in forbidden_functions:
                    print(f"Bloqueo de seguridad: Función prohibida detectada ({node.func.id})")
                    return False

    return True

def call_ollama(prompt: str, system_prompt: str) -> str:
    """Función auxiliar para encapsular la llamada a la API de Ollama."""
    payload = {
        "model": MODEL_NAME,
        "system": system_prompt,
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=180)
        response.raise_for_status()
        return response.json().get("response", "")
    except requests.exceptions.RequestException as e:
        print(f"Error en llamada a Ollama: {e}")
        raise


def build_timestamp_base_name() -> str:
    """Genera nombre base con formato año-mes-dia_hora-minuto."""
    return datetime.now().strftime("%Y-%m-%d_%H-%M")


def object_exists_in_minio(object_name: str) -> bool:
    """Verifica si un objeto ya existe en MinIO."""
    try:
        storage.s3_client.head_object(Bucket=storage.BUCKET_NAME, Key=object_name)
        return True
    except ClientError as error:
        status_code = error.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if status_code == 404:
            return False
        print(f"Advertencia verificando objeto en MinIO ({object_name}): {error}")
        return False
    except Exception as error:
        print(f"Advertencia verificando objeto en MinIO ({object_name}): {error}")
        return False


def build_unique_timestamp_base_name() -> str:
    """Genera un nombre base único con formato año-mes-dia_hora-minuto y sufijos _2, _3, ..."""
    base_name = build_timestamp_base_name()
    suffix = 1

    while True:
        candidate = base_name if suffix == 1 else f"{base_name}_{suffix}"
        step_candidate = f"{candidate}.step"
        stl_candidate = f"{candidate}.stl"

        local_exists = (
            os.path.exists(step_candidate)
            or os.path.exists(stl_candidate)
            or os.path.exists(f"{candidate}_script.py")
        )
        remote_exists = object_exists_in_minio(step_candidate) or object_exists_in_minio(stl_candidate)

        if not local_exists and not remote_exists:
            return candidate

        suffix += 1

def process_3d_generation(job_id: str, prompt: str):
    """
    Orquesta el pipeline de agentes en cascada, ejecuta el código CadQuery
    y sube los resultados a MinIO.
    """
    db = SessionLocal()
    job = db.query(models.GenerationJob).filter(models.GenerationJob.id == job_id).first()

    if not job:
        print(f"Trabajo {job_id} no encontrado")
        db.close()
        return

    base_name = build_unique_timestamp_base_name()
    py_filename = f"{base_name}_script.py"
    step_filename = f"{base_name}.step"
    stl_filename = f"{base_name}.stl"

    try:
        job.status = "processing"
        db.commit()

        # --- PIPELINE DE AGENTES ---
        print(f"[{job_id}] Paso 1: Expandiendo concepto (Diseñador Industrial)...")
        design_description = call_ollama(prompt=prompt, system_prompt=SYSTEM_PROMPT_STEP_1)

        print(f"[{job_id}] Paso 2: Análisis geométrico (Analista Geométrico)...")
        geometric_analysis = call_ollama(prompt=design_description, system_prompt=SYSTEM_PROMPT_STEP_2)

        print(f"[{job_id}] Paso 3: Generando código Python (CadQuery)...")

        # Inyectamos los nombres de archivo exactos que la IA debe usar en las exportaciones
        prompt_paso_3 = (
            f"Especificación geométrica:\n{geometric_analysis}\n\n"
            f"REQUISITO FINAL: Debes exportar el modelo a '{step_filename}' y '{stl_filename}'."
        )
        raw_output = call_ollama(prompt=prompt_paso_3, system_prompt=SYSTEM_PROMPT_STEP_3)

        py_code = extract_code(raw_output)

        # --- VALIDACIÓN DE SEGURIDAD ---
        if not is_safe_python_code(py_code):
            raise Exception("El código generado no pasó las verificaciones de seguridad.")

        # Guardar el script Python
        with open(py_filename, "w", encoding="utf-8") as f:
            f.write(py_code)

        # --- EJECUCIÓN DEL SCRIPT (usando WSL) ---
        print(f"[{job_id}] Ejecutando script CadQuery en WSL...")

        # Se asume que WSL tiene Python 3 instalado y el script se ejecuta desde el directorio actual
        # (montado en /mnt/c/...). Ajusta la ruta si es necesario.
        subprocess.run([sys.executable, py_filename], check=True, capture_output=True, text=True)

        # --- SUBIDA A MINIO ---
        print(f"[{job_id}] Subiendo modelos a almacenamiento (MinIO)...")

        if not os.path.exists(step_filename) or not os.path.exists(stl_filename):
            raise Exception("El script se ejecutó pero no generó todos los archivos esperados (.step y .stl).")

        step_url = storage.upload_file(step_filename, step_filename)
        stl_url = storage.upload_file(stl_filename, stl_filename)

        if not step_url or not stl_url:
            raise Exception("Error al subir los archivos a MinIO")

        # Actualizamos la base de datos
        job.status = "completed"

        # Guardamos ambas URLs separadas por coma en la base de datos
        # (El frontend deberá hacer un .split(","))
        job.file_url = f"{step_url},{stl_url}"
        db.commit()
        print(f"✅ [{job_id}] Generación CadQuery completada con éxito!")

    except subprocess.CalledProcessError as e:
        print(f"❌ Error ejecutando script CadQuery:\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}")
        job.status = "failed"
        db.commit()
    except Exception as e:
        print(f"❌ Error general en proceso: {e}")
        job.status = "failed"
        db.commit()
    finally:
        db.close()

    # Limpieza exhaustiva de todos los archivos generados durante el ciclo
    archivos_temporales = [py_filename, step_filename, stl_filename]
    for archivo in archivos_temporales:
        if os.path.exists(archivo):
            try:
                os.remove(archivo)
            except Exception as ex:
                print(f"No se pudo eliminar el archivo temporal {archivo}: {ex}")