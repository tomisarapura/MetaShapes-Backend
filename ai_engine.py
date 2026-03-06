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
LOG_PIPELINE_TEXT = os.getenv("LOG_PIPELINE_TEXT", "true").lower() == "true"
LOG_MAX_CHARS = int(os.getenv("LOG_MAX_CHARS", "12000"))

# --- SYSTEM PROMPTS (PIPELINE DE 3 PASOS) ---
SYSTEM_PROMPT_STEP_1 = """
You are an expert Industrial Designer specializing in digital manufacturing (CNC and 3D Printing). 
Your goal is to transform the user's idea into a rigorous engineering technical specification.

For every design, you must:
1. COMPONENT BREAKDOWN: Deconstruct the object into its primary volumes (boxes, cylinders, spheres, toroids).
2. SPATIAL HIERARCHY: Define the "Base" part and specify which parts are "anchored" to or "subtracted" from it.
3. SIZE RELATIONS: If specific measurements are missing, assign realistic dimensions in millimeters based on the object's intended use.
4. FINISHING DETAILS: Specify if edges should be rounded (fillets) or angled (chamfers) to improve ergonomics or structural integrity.

Provide a structured technical description focused purely on form and function. Do not generate code.
"""

SYSTEM_PROMPT_STEP_2 = """
You are an expert Geometric CAD Analyst. Your task is to translate an industrial design description into a rigorous mathematical construction plan.

MANDATORY RULES:
- ORIGIN POINT: Define the center of the base part at (0, 0, 0).
- PRIMITIVE TABLE: For each part, list:
    - Shape (Box, Cylinder, Sphere, etc.).
    - Exact Dimensions (Length, Width, Height or Radius, Height) in mm.
    - Relative Position: Center-of-mass coordinates (X, Y, Z).
    - Rotation: Angles in degrees if applicable.
- BOOLEAN LOGIC: Explicitly state which parts are JOINED (union) and which are SUBTRACTED (cut).
- FACE REFERENCING: If a part is built on another, specify the reference face (e.g., "on the top face (+Z) of the Base").

Do not write unnecessary prose. Deliver a technical list of construction steps.
"""

SYSTEM_PROMPT_STEP_3 = """
You are a senior Python developer, an expert in CadQuery and 3D parametric design. 
Your sole mission is to translate the received geometric planning into a valid, robust, and executable Python script using CadQuery.

CRITICAL TECHNICAL RULES (TO AVOID ERRORS AND CRASHES):
1) INDEPENDENT SOLIDS:
   - Create each part as an independent 3D solid: p1 = cq.Workplane("XY").box(...), p2 = cq.Workplane("XY").cylinder(...), etc.
   - Do not chain a long flow of operations on a single Workplane; use variables for each part.

2) EXPLICIT FORMAT PRIMITIVES:
   - Always use the full format for primitives:
     * cq.Workplane("XY").box(x, y, z)
     * cq.Workplane("XY").cylinder(height, radius)
     * cq.Workplane("XY").sphere(radius)

3) EXPLICIT TRANSFORMATIONS BEFORE BOOLEANS:
   - Position each solid with .translate((x, y, z)) and/or .rotate((ax, ay, az), (vx, vy, vz), angle) before any .union(), .cut(), or .intersect().

4) BOOLEAN OPERATIONS ONLY BETWEEN SOLIDS:
   - Apply .union(), .cut(), or .intersect() only to 3D SOLIDS.
   - NEVER apply boolean operations to an empty Workplane or an un-extruded 2D sketch.
   - For unions: final_model = part_a.union(part_b)
   - For cuts:   final_model = final_model.cut(part_c)

5) PROGRESSIVE ASSEMBLY:
   - Create and position all parts first.
   - Assemble with secure unions (in the necessary order).
   - Create cutting solids as independent bodies and apply them afterward.

6) MANDATORY EXPORT:
   - Use the environment-injected variables: step_filename and stl_filename.
   - At the end of the script, export with:
     cq.exporters.export(final_model, step_filename)
     cq.exporters.export(final_model, stl_filename)

OUTPUT STYLE:
- Return ONLY the Python code within a ```python ... ``` block.
- No explanations, no prose, and no comments outside the code. The script must be self-contained.

IDEAL STRUCTURE EXAMPLE:
```python
import cadquery as cq

# Parameters (modify as needed)
base_x, base_y, base_z = 50, 50, 10
cyl_radius, cyl_height = 15, 20
hole_radius = 5

# 1) Independent parts (3D solids)
base = cq.Workplane("XY").box(base_x, base_y, base_z)
cylinder = cq.Workplane("XY").cylinder(cyl_height, cyl_radius)

# 2) Positioning before booleans
cylinder = cylinder.translate((0, 0, base_z/2 + cyl_height/2))

# 3) Assembly (secure unions between solids)
final_model = base.union(cylinder)

# 4) Cutting solids
hole = cq.Workplane("XY").cylinder(base_z + cyl_height + 10, hole_radius)

# 5) Apply cuts
final_model = final_model.cut(hole)

# 6) Export (step_filename and stl_filename already exist in the environment)
cq.exporters.export(final_model, step_filename)
cq.exporters.export(final_model, stl_filename)
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
        design_description = call_ollama(prompt=prompt, system_prompt=SYSTEM_PROMPT_STEP_1)
        log_pipeline_block(job_id, "Consigna generada - Paso 1 (diseño detallado)", design_description)

        print(f"[{job_id}] Paso 2: Análisis geométrico (Analista Geométrico)...", flush=True)
        geometric_analysis = call_ollama(prompt=design_description, system_prompt=SYSTEM_PROMPT_STEP_2)
        log_pipeline_block(job_id, "Consigna generada - Paso 2 (análisis geométrico)", geometric_analysis)

        print(f"[{job_id}] Paso 3: Generando código Python (CadQuery)...", flush=True)

        # Inyectamos los nombres de archivo exactos que la IA debe usar en las exportaciones
        prompt_paso_3 = (
            f"Especificación geométrica:\n{geometric_analysis}\n\n"
            f"REQUISITO FINAL: Debes exportar el modelo a '{step_filename}' y '{stl_filename}'."
        )
        log_pipeline_block(job_id, "Consigna enviada al Paso 3 (generación CadQuery)", prompt_paso_3)
        raw_output = call_ollama(prompt=prompt_paso_3, system_prompt=SYSTEM_PROMPT_STEP_3)
        log_pipeline_block(job_id, "Respuesta cruda del modelo en Paso 3", raw_output)

        # --- FIN DEL PIPELINE ---

        # 1. Extraemos el código generado por la IA
        python_code = extract_code(raw_output)
        log_pipeline_block(job_id, "Código Python extraído para ejecución", python_code)

        # 2. Usamos nombres basados en timestamp (definidos al inicio del proceso)

        # --- VALIDACIÓN DE SEGURIDAD ---
        if not is_safe_python_code(python_code):
            raise Exception("El código generado no pasó las verificaciones de seguridad.")

        # 3. INYECCIÓN CRÍTICA: Creamos las variables como código Python
        injected_header = f"""# Variables inyectadas por el backend
step_filename = \"{step_filename}\"
stl_filename = \"{stl_filename}\"
"""

        # 4. Unimos las variables inyectadas con el código de la IA
        final_script_content = injected_header + "\n" + python_code

        # 5. Guardamos el archivo listo para ejecutar
        with open(script_filename, "w", encoding="utf-8") as f:
            f.write(final_script_content)

        # 6. Ejecutamos el script localmente
        print(f"[{job_id}] Ejecutando script CadQuery...", flush=True)
        subprocess.run([sys.executable, script_filename], check=True, capture_output=True, text=True)
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

        # Guardamos ambas URLs separadas por coma en la base de datos
        # (El frontend deberá hacer un .split(","))
        job.file_url = f"{step_url},{stl_url}"
        db.commit()
        print(f"✅ [{job_id}] Generación CadQuery completada con éxito!", flush=True)

    except subprocess.CalledProcessError as e:
        print(f"❌ Error ejecutando script CadQuery:\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}", flush=True)
        job.status = "failed"
        db.commit()
    except Exception as e:
        print(f"❌ Error general en proceso: {e}", flush=True)
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