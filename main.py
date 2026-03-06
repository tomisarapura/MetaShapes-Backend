from fastapi import FastAPI, BackgroundTasks, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from database import engine, get_db, Base
import models
import storage
import ai_engine 
from typing import List 

# Esto crea las tablas en la base de datos de Docker automáticamente al iniciar
Base.metadata.create_all(bind=engine)

# Inicializa el bucket en MinIO y le da permisos públicos
storage.init_storage() # <-- Ejecutamos la inicialización

app = FastAPI(
    title="MetaShapes API",
    description="API para orquestar la generación de modelos 3D a partir de texto",
    version="0.1.0"
)

# --- ESQUEMAS (Pydantic) ---
class GenerationRequest(BaseModel):
    prompt: str
    
class JobResponse(BaseModel):
    job_id: str
    status: str
    message: str

class JobStatusResponse(BaseModel):
    job_id: str
    prompt: str
    status: str
    file_url: str | None

# --- ENDPOINTS ---

@app.get("/")
def read_root():
    return {"status": "online", "message": "MetaShapes Backend funcionando 🚀"}

@app.post("/generate", response_model=JobResponse)
async def create_generation_job(
    request: GenerationRequest, 
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db) # Inyectamos la conexión a la BD
):
    # 1. Creamos el registro en la base de datos
    new_job = models.GenerationJob(prompt=request.prompt)
    db.add(new_job)
    db.commit()
    db.refresh(new_job) # Obtenemos el ID autogenerado
    
    # 2. Mandamos la tarea pesada al fondo para no bloquear la respuesta
    background_tasks.add_task(ai_engine.process_3d_generation, new_job.id, request.prompt)
    
    return JobResponse(
        job_id=new_job.id,
        status=new_job.status,
        message="Trabajo guardado y en proceso"
    )

@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str, db: Session = Depends(get_db)):
    # Buscamos el trabajo en la base de datos
    job = db.query(models.GenerationJob).filter(models.GenerationJob.id == job_id).first()
    
    if not job:
        raise HTTPException(status_code=404, detail="Trabajo no encontrado")
        
    return JobStatusResponse(
        job_id=job.id,
        prompt=job.prompt,
        status=job.status,
        file_url=job.file_url
    )

# <-- AGREGAR EL NUEVO ENDPOINT AQUÍ
@app.get("/jobs", response_model=List[JobStatusResponse])
def get_all_jobs(db: Session = Depends(get_db)):
    # Devuelve todos los trabajos ordenados por el más reciente
    jobs = db.query(models.GenerationJob).order_by(models.GenerationJob.created_at.desc()).all()
    return jobs