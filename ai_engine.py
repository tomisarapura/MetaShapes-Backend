import requests
import subprocess
import os
import re
import ast
import sys
import importlib
from datetime import datetime
from botocore.exceptions import ClientError
import storage
from database import SessionLocal
import models
from dotenv import load_dotenv

# Cargamos las variables de entorno
load_dotenv()

# --- CONFIGURACIÓN ---
AI_PROVIDER = os.getenv("AI_PROVIDER", "ollama").lower()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen2.5-coder:14b")

LOG_PIPELINE_TEXT = os.getenv("LOG_PIPELINE_TEXT", "true").lower() == "true"
LOG_MAX_CHARS = int(os.getenv("LOG_MAX_CHARS", "12000"))

# --- SYSTEM PROMPTS (PIPELINE DE 3 PASOS) ---
SYSTEM_PROMPT_STEP_1 = """
You are an expert Industrial Designer. Deconstruct the user's prompt into a strict mechanical specification.

CRITICAL RULE: DO NOT add extra components unless the user explicitly asks for them. Build ONLY what is requested.

Output a structured report with:
1. OVERALL DIMENSIONS: (Bounding box).
2. COMPONENT LIST: Name every geometric part.
3. EXACT SIZES: Provide realistic dimensions in mm for EVERY component.
4. SHAPE DETAILS: Specifically mention if a shape is a Cylinder, Box, Sphere, Cone, or a Truncated Cone (Frustum).
"""

SYSTEM_PROMPT_STEP_2 = """
You are a Geometric Planner. Convert the industrial design into a strict coordinate system map.
Origin (0,0,0) is the center of mass of the primary base object.

CRITICAL RULE: ALL dimensions and coordinates MUST be in millimeters (mm) ONLY. Convert cm to mm (e.g., 14 cm = 140 mm).

For EVERY object, output a line with:
- Part Name
- Primitive Type (Box, Cylinder, Sphere, Cone, TruncatedCone)
- Dimensions (X, Y, Z or Radius, Height or BottomRadius, TopRadius, Height) in mm
- Translation Coordinates from Origin (Tx, Ty, Tz) in mm
- Boolean Operation (Base, Union, Cut)

Output ONLY the technical list. Do not use conversational text.
"""

SYSTEM_PROMPT_STEP_3 = """
You are an expert CadQuery (Python) developer. Write code to build the 3D model based on the geometric plan.

STRICT RULES:
1. DO NOT import cadquery. It is already imported as `cq`.
2. You MUST assign the final geometric object to a variable named EXACTLY `result`.
3. Combine parts correctly using `.union()` or `.cut()`. 
4. DO NOT union a solid part over a hole you just cut.

CADQUERY CHEAT SHEET:
- Truncated Cone (Frustum): Use lofting. Example:
  `cq.Workplane("XY").workplane(offset=0).circle(bottom_radius).workplane(offset=height).circle(top_radius).loft()`
- Hollowing an object (like a pot/bowl): Create the solid body, then select the top face and use `.shell(thickness)`. Example: 
  `cq.Workplane("XY").cylinder(radius, height).faces(">Z").shell(-5)`

EXAMPLE OUTPUT:
```python
# Main Body (Truncated Cone)
main_body = cq.Workplane("XY").workplane(offset=0).circle(50).workplane(offset=140).circle(75).loft()
# Inner Hole
hole = cq.Workplane("XY").workplane(offset=5).circle(45).workplane(offset=140).circle(70).loft()
# Combine
result = main_body.cut(hole)
OUTPUT ONLY PYTHON CODE inside python blocks.
"""


def extract_code(response_text: str) -> str:
    """Limpia la respuesta de la IA para extraer el código Python."""
    fenced_match = re.search(r"```python\s*(.*?)```", response_text, re.DOTALL | re.IGNORECASE)
    if fenced_match:
        return fenced_match.group(1).strip()
    return response_text.strip()


def validate_cadquery_code(code_str: str) -> tuple[bool, str]:
    """
    Analiza el AST (Abstract Syntax Tree) para validar el código.
    Verifica errores de sintaxis, bloquea imports maliciosos y asegura
    que el resultado final se guarde en la variable 'result'.
    """
    forbidden_modules = {"os", "sys", "subprocess", "shutil", "socket", "requests", "pathlib"}
    forbidden_functions = {"eval", "exec", "open", "compile", "import"}

    try:
        tree = ast.parse(code_str)
    except SyntaxError as e:
        return False, f"Error de sintaxis de Python: {e}"

    has_result_assignment = False

    for node in ast.walk(tree):
        # Verifica imports simples (ej: import os)
        if isinstance(node, ast.Import):
            for alias in node.names:
                base_module = alias.name.split(".")[0]
                if base_module in forbidden_modules:
                    return False, f"Importación de seguridad prohibida detectada ({alias.name})"

        # Verifica imports desde módulos (ej: from os import system)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                base_module = node.module.split(".")[0]
                if base_module in forbidden_modules:
                    return False, f"Importación 'from' prohibida ({node.module})"

        # Verifica llamadas a funciones peligrosas
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in forbidden_functions:
                return False, f"Función prohibida detectada ({node.func.id})"

        # Verifica que exista una asignación a la variable 'result'
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "result":
                    has_result_assignment = True

    if not has_result_assignment:
        return (
            False,
            "No se encontró la asignación obligatoria a la variable 'result'. El código debe terminar asignando el modelo final a 'result'.",
        )

    return True, ""


def call_llm(prompt: str, system_prompt: str) -> str:
    """Llama a la API de Ollama o a Groq dependiendo de la configuración."""
    if AI_PROVIDER == "groq":
        try:
            openai_module = importlib.import_module("openai")
            OpenAI = getattr(openai_module, "OpenAI")

            client = OpenAI(
                api_key=GROQ_API_KEY,
                base_url="https://api.groq.com/openai/v1",
            )
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error en llamada a Groq: {e}", flush=True)
            raise
    else:
        payload = {
            "model": MODEL_NAME,
            "system": system_prompt,
            "prompt": prompt,
            "stream": False,
        }
        try:
            response = requests.post(OLLAMA_URL, json=payload, timeout=600)
            response.raise_for_status()
            return response.json().get("response", "")
        except requests.exceptions.RequestException as e:
            print(f"Error en llamada a Ollama: {e}", flush=True)
            raise


def log_pipeline_block(job_id: str, title: str, text: str) -> None:
    """Imprime bloques largos del pipeline en terminal con truncado configurable."""
    if not LOG_PIPELINE_TEXT:
        return

    clean_text = (text or "").strip()
    if not clean_text:
        clean_text = "<vacío>"

    if len(clean_text) > LOG_MAX_CHARS:
        clean_text = clean_text[:LOG_MAX_CHARS] + "\n...[truncado]"

    separator = "=" * 72
    print(f"\n[{job_id}] {separator}", flush=True)
    print(f"[{job_id}] {title}", flush=True)
    print(f"[{job_id}] {separator}", flush=True)
    print(clean_text, flush=True)
    print(f"[{job_id}] {separator}\n", flush=True)


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
    script_filename = f"{base_name}_script.py"
    step_filename = f"{base_name}.step"
    stl_filename = f"{base_name}.stl"

    try:
        job.status = "processing"
        db.commit()

        # --- PIPELINE DE AGENTES ---
        print(f"[{job_id}] Paso 1: Expandiendo concepto (Diseñador Industrial)...", flush=True)
        log_pipeline_block(job_id, "Prompt inicial del usuario", prompt)
        design_description = call_llm(prompt=prompt, system_prompt=SYSTEM_PROMPT_STEP_1)
        log_pipeline_block(job_id, "Consigna generada - Paso 1 (diseño detallado)", design_description)

        print(f"[{job_id}] Paso 2: Análisis geométrico (Analista Geométrico)...", flush=True)
        geometric_analysis = call_llm(prompt=design_description, system_prompt=SYSTEM_PROMPT_STEP_2)
        log_pipeline_block(job_id, "Consigna generada - Paso 2 (análisis geométrico)", geometric_analysis)

        print(f"[{job_id}] Paso 3: Generando código Python (CadQuery)...", flush=True)

        prompt_paso_3 = (
            f"Especificación geométrica:\n{geometric_analysis}\n\n"
            f"REQUISITO FINAL: Genera el código asignando el modelo final a la variable 'result'."
        )
        log_pipeline_block(job_id, "Consigna enviada al Paso 3 (generación CadQuery)", prompt_paso_3)

        # --- VALIDACIÓN Y REINTENTOS ---
        max_retries = 3
        python_code = ""
        current_prompt = prompt_paso_3

        for attempt in range(max_retries):
            raw_output = call_llm(prompt=current_prompt, system_prompt=SYSTEM_PROMPT_STEP_3)
            log_pipeline_block(job_id, f"Respuesta cruda del modelo en Paso 3 (Intento {attempt + 1})", raw_output)

            python_code = extract_code(raw_output)

            # Validación robusta de AST y reglas de negocio
            is_valid, error_msg = validate_cadquery_code(python_code)

            if is_valid:
                log_pipeline_block(job_id, "Código Python extraído y validado", python_code)
                break

            print(f"[{job_id}] Validación fallida en intento {attempt + 1}: {error_msg}", flush=True)
            if attempt < max_retries - 1:
                # Retroalimentamos al modelo con su propio error
                current_prompt = (
                    f"{prompt_paso_3}\n\n"
                    f"El código que generaste anteriormente falló la validación con este error:\n"
                    f"{error_msg}\n"
                    "Por favor, analiza el error, corrígelo y genera una versión que funcione."
                )
        else:
            # Si agota los intentos y no hay código válido
            raise Exception(
                f"No se pudo generar un código válido tras {max_retries} intentos. Último error: {error_msg}"
            )

        # --- FIN DEL PIPELINE ---

        injected_header = "import cadquery as cq\nimport math\n# --- CODIGO GENERADO POR IA ---\n"
        injected_footer = (
            "\n# --- EXPORTACION AUTOMATICA ---\n"
            "if 'result' not in locals():\n"
            "    raise ValueError('El script no definio result.')\n"
            f"cq.exporters.export(result, '{step_filename}')\n"
            f"cq.exporters.export(result, '{stl_filename}')\n"
        )

        final_script_content = injected_header + python_code + injected_footer

        with open(script_filename, "w", encoding="utf-8") as file:
            file.write(final_script_content)

        print(f"[{job_id}] Ejecutando script CadQuery...", flush=True)
        subprocess.run([sys.executable, script_filename], check=True, capture_output=True, text=True, timeout=45)

        # --- SUBIDA A MINIO ---
        print(f"[{job_id}] Subiendo modelos a almacenamiento (MinIO)...", flush=True)

        if not os.path.exists(step_filename) or not os.path.exists(stl_filename):
            raise Exception("El script se ejecutó pero no generó todos los archivos esperados (.step y .stl).")

        step_url = storage.upload_file(step_filename, step_filename)
        stl_url = storage.upload_file(stl_filename, stl_filename)

        if not step_url or not stl_url:
            raise Exception("Error al subir los archivos a MinIO")

        # Actualizamos la base de datos
        job.status = "completed"
        job.file_url = f"{step_url},{stl_url}"
        db.commit()
        print(f"[{job_id}] Generacion CadQuery completada con exito!", flush=True)

    except subprocess.CalledProcessError as e:
        print(f"Error ejecutando script CadQuery:\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}", flush=True)
        job.status = "failed"
        db.commit()
    except Exception as e:
        print(f"Error general en proceso: {e}", flush=True)
        job.status = "failed"
        db.commit()
    finally:
        db.close()

        # Limpieza exhaustiva de todos los archivos generados durante el ciclo
        archivos_temporales = [script_filename, step_filename, stl_filename]
        for archivo in archivos_temporales:
            if os.path.exists(archivo):
                try:
                    os.remove(archivo)
                except Exception as ex:
                    print(f"No se pudo eliminar el archivo temporal {archivo}: {ex}", flush=True)