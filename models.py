from sqlalchemy import Column, String, DateTime
from database import Base
import datetime
import uuid

class GenerationJob(Base):
    __tablename__ = "generation_jobs"

    # Usamos UUID como String para que sea compatible y fácil de manejar
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    prompt = Column(String, nullable=False)
    status = Column(String, default="pending") # Puede ser: pending, processing, completed, failed
    file_url = Column(String, nullable=True)   # Aquí guardaremos el link de MinIO cuando termine
    created_at = Column(DateTime, default=datetime.datetime.utcnow)