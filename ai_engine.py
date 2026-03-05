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
Eres un experto en diseño paramétrico 3D usando Python y la librería CadQuery.
Tu única tarea es traducir la especificación geométrica recibida a un script de Python válido y eficiente usando CadQuery.

REGLAS ESTRICTAS DE CADQUERY (¡CRÍTICO PARA EVITAR ERRORES!):
1. Todo objeto debe ser un SÓLIDO 3D antes de aplicarle operaciones booleanas (.union, .cut, .intersect).
2. NUNCA apliques .union() a un Workplane vacío o a un boceto 2D que no haya sido extruido.
3. Para primitivas directas, usa siempre el formato completo: 
   cq.Workplane("XY").box(x, y, z)
   cq.Workplane("XY").cylinder(altura, radio)
   cq.Workplane("XY").sphere(radio)
4. Crea cada parte como una variable independiente y luego únelas al final.
5. Usa el método .translate((x, y, z)) de CadQuery para mover los sólidos a sus posiciones correctas antes de unirlos.
6. AL FINAL DEL SCRIPT, debes exportar el resultado final a los formatos requeridos usando las variables predefinidas `step_filename` y `stl_filename`.
7. Devuelve ÚNICAMENTE el código Python dentro de un bloque ```python ... ```.

--- EJEMPLO IDEAL DE CÓDIGO ---
```python
import cadquery as cq

# Parámetros (usar valores inferidos si no se especifican)
base_x, base_y, base_z = 50, 50, 10
cil_radio, cil_altura = 15, 20
agujero_radio = 5

# 1. Crear las partes como sólidos independientes
base = cq.Workplane("XY").box(base_x, base_y, base_z)

# 2. Posicionar las partes (ej. mover el cilindro arriba de la base)
cilindro = cq.Workplane("XY").cylinder(cil_altura, cil_radio).translate((0, 0, base_z/2 + cil_altura/2))

# 3. Unir los sólidos
modelo_final = base.union(cilindro)

# 4. Crear sólidos para cortes (diferencia)
agujero = cq.Workplane("XY").cylinder(base_z + cil_altura + 10, agujero_radio)

# 5. Aplicar el corte
modelo_final = modelo_final.cut(agujero)

# 6. Exportar (Las variables step_filename y stl_filename serán inyectadas por el sistema)
cq.exporters.export(modelo_final, step_filename)
cq.exporters.export(modelo_final, stl_filename)
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