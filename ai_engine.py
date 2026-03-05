import requests
import subprocess
import os
import re
import datetime
import storage
from database import SessionLocal
import models
from dotenv import load_dotenv

# Cargamos las variables de entorno
load_dotenv()

# --- CONFIGURACIÓN ---
# Busca la URL en el .env, si no la encuentra usa localhost por defecto
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
MODEL_NAME = "qwen2.5-coder:14b"


def build_artifact_basename(created_at: datetime.datetime | None, job_id: str) -> str:
    """Genera un nombre legible tipo YYYY-MM-DD_HH-MM con sufijo corto para evitar colisiones."""
    base_dt = created_at or datetime.datetime.now()
    timestamp = base_dt.strftime("%Y-%m-%d_%H-%M")
    short_id = (job_id or "")[:8]
    return f"{timestamp}_{short_id}" if short_id else timestamp

# --- SYSTEM PROMPTS (PIPELINE DE 3 PASOS) ---
SYSTEM_PROMPT_STEP_1 = """
Eres un diseñador industrial experto. 
Tu tarea es tomar la idea base de un usuario y expandirla en lenguaje natural detallado. 
Describe su función principal, las partes exactas que lo componen, sus proporciones generales y cómo interactúan las piezas entre sí.
"""

SYSTEM_PROMPT_STEP_2 = """
Eres un analista geométrico experto. 
Toma la descripción del diseño industrial proporcionada y tradúcela ESTRICTAMENTE a una lista técnica estructurada de primitivas geométricas 3D (cilindros, cubos, esferas, polígonos).

REQUISITOS OBLIGATORIOS:
- Para cada parte, especifica dimensiones numéricas en milímetros (X, Y, Z o radios/alturas).
- Si la descripción no incluye medidas, infiere valores realistas basados en el contexto (ej. una silla ~400 mm de alto, una taza ~80 mm).
- Incluye traslaciones (X,Y,Z) respecto al origen, rotaciones y operaciones booleanas.
- No escribas código, solo la planificación matemática y geométrica.
- Usa el formato:
  Parte: [nombre]
  Primitiva: [cubo/cilindro/esfera...]
  Dimensiones: [valores en mm]
  Traslación: [x, y, z]
  Rotación: [rx, ry, rz] (grados)
  Operación: [unión/diferencia/intersección]

Ejemplo:
Entrada: "Una base cilíndrica con un agujero en el centro."
Salida:
Parte: Base
Primitiva: cilindro
Dimensiones: radio=50, altura=10
Traslación: [0,0,0]
Rotación: [0,0,0]
Operación: unión

Parte: Agujero
Primitiva: cilindro
Dimensiones: radio=20, altura=12
Traslación: [0,0,-1]
Rotación: [0,0,0]
Operación: diferencia
"""

SYSTEM_PROMPT_STEP_3 = """
Eres un experto en diseño paramétrico 3D usando OpenSCAD. 
Tu única tarea es traducir la especificación geométrica recibida a código OpenSCAD válido, eficiente y bien estructurado.

REGLAS ESTRICTAS:
1. El código debe comenzar con las siguientes líneas exactamente:
   $fn = 50;            // resolución de superficies curvas
   eps = 0.01;           // épsilon para evitar Z-fighting en diferencias

2. Define todas las dimensiones como variables paramétricas al inicio del script (ej: altura_total = 100; radio_base = 20; ancho = 30;). Usa nombres descriptivos en minúsculas y con guiones bajos.

3. Para las operaciones de diferencia (difference()), asegúrate de que las piezas a sustraer sean ligeramente más largas en la dirección del corte usando eps. Por ejemplo:
   translate([0, 0, -eps]) cylinder(r=radio, h=altura+2*eps);

4. Utiliza center=true en las primitivas siempre que sea conveniente para facilitar el posicionamiento.

5. El código debe ser completo y autocontenido: al ejecutarse, debe generar el modelo 3D final sin necesidad de edición manual.

6. Devuelve ÚNICAMENTE el código OpenSCAD dentro de un bloque ```openscad ... ```. No incluyas explicaciones, ni introducciones, ni saludos, ni texto adicional fuera del bloque.

A continuación se muestran dos ejemplos de especificaciones geométricas y sus correspondientes códigos OpenSCAD correctos. Utilízalos como referencia para generar el código solicitado.

--- EJEMPLO 1 ---
Especificación:
"La pieza es un cilindro base de 20 mm de radio y 5 mm de alto, con un agujero pasante de 5 mm de radio en el centro."

Código OpenSCAD:
```openscad
$fn = 50;
eps = 0.01;

// Parámetros
radio_base = 20;
altura_base = 5;
radio_agujero = 5;

// Modelo
difference() {
    cylinder(r = radio_base, h = altura_base);
    // Agujero pasante
    translate([0, 0, -eps])
        cylinder(r = radio_agujero, h = altura_base + 2*eps);
}"""

def extract_code(response_text: str) -> str:
    """Limpia la respuesta de la IA por si incluyó formato Markdown."""
    # Busca código dentro de bloques ```openscad ... ``` o ``` ... ```
    match = re.search(r'```(?:openscad)?\n(.*?)```', response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Si no encuentra bloques, devuelve todo el texto (por si acaso)
    return response_text.strip()

def call_ollama(prompt: str, system_prompt: str) -> str:
    """Función auxiliar para encapsular la llamada a la API de Ollama."""
    payload = {
        "model": MODEL_NAME,
        "system": system_prompt,
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        return response.json().get("response", "")
    except requests.exceptions.RequestException as e:
        print(f"Error en llamada a Ollama: {e}")
        raise

def process_3d_generation(job_id: str, prompt: str):
    """
    Esta es la tarea pesada que correrá en segundo plano ejecutando el pipeline de agentes en cascada.
    """
    db = SessionLocal()
    job = db.query(models.GenerationJob).filter(models.GenerationJob.id == job_id).first()

    if not job:
        print(f"Trabajo {job_id} no encontrado")
        db.close()
        return

    scad_filename = None
    stl_filename = None

    try:
        # Actualizamos estado a procesando
        job.status = "processing"
        db.commit()

        # --- INICIO DEL PIPELINE DE AGENTES EN CASCADA ---

        # Paso 1: Diseñador Industrial
        print(f"[{job_id}] Paso 1: Expandiendo concepto (Diseñador Industrial)...")
        design_description = call_ollama(prompt=prompt, system_prompt=SYSTEM_PROMPT_STEP_1)

        # Paso 2: Analista Geométrico
        print(f"[{job_id}] Paso 2: Análisis geométrico (Analista Geométrico)...")
        geometric_analysis = call_ollama(prompt=design_description, system_prompt=SYSTEM_PROMPT_STEP_2)

        # Paso 3: Compilador OpenSCAD
        print(f"[{job_id}] Paso 3: Generando código (Compilador OpenSCAD)...")
        raw_output = call_ollama(prompt=geometric_analysis, system_prompt=SYSTEM_PROMPT_STEP_3)

        # --- FIN DEL PIPELINE ---

        scad_code = extract_code(raw_output)

        # Guardar el código temporalmente
        artifact_basename = build_artifact_basename(job.created_at, job_id)
        scad_filename = f"{artifact_basename}.scad"
        stl_filename = f"{artifact_basename}.stl"

        with open(scad_filename, "w", encoding="utf-8") as f:
            f.write(scad_code)

        # Compilar usando OpenSCAD (ajusta el comando según tu sistema)
        print(f"[{job_id}] Compilando código OpenSCAD a STL...")
        # Si estás en Windows con WSL:
        subprocess.run(["wsl", "openscad", "-o", stl_filename, scad_filename], check=True, capture_output=True)
        
        # Subir a MinIO (o almacenamiento local)
        print(f"[{job_id}] Subiendo modelo a almacenamiento...")
        file_url = storage.upload_file(stl_filename, stl_filename)

        if not file_url:
            raise Exception("Error al subir el archivo a MinIO")

        # Actualizar la base de datos con éxito
        job.status = "completed"
        job.file_url = file_url
        db.commit()
        print(f"✅ [{job_id}] Generación completada con éxito!")

    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode() if e.stderr else str(e)
        print(f"❌ Error compilando OpenSCAD: {error_msg}")
        job.status = "failed"
        db.commit()
    except Exception as e:
        print(f"❌ Error general en proceso: {e}")
        job.status = "failed"
        db.commit()
    finally:
        db.close()
        # Limpiamos los archivos temporales
        if scad_filename and os.path.exists(scad_filename):
            os.remove(scad_filename)
        if stl_filename and os.path.exists(stl_filename):
            os.remove(stl_filename)
