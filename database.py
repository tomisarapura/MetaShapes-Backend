import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

load_dotenv()

# URL de conexión a tu contenedor Docker (usuario:clave@host:puerto/basededatos)
SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL")

# El "engine" es el motor que maneja la comunicación real con Postgres
engine = create_engine(SQLALCHEMY_DATABASE_URL)

# SessionLocal será la clase que instanciemos para crear sesiones de base de datos en cada petición
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base es la clase de la que heredarán nuestros modelos de datos
Base = declarative_base()

# Dependencia para inyectar la sesión en los endpoints de FastAPI
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()