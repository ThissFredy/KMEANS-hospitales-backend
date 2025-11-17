# api.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from model import Kmeans
import numpy as np

def lifespan(app: FastAPI):
    """Crea una instancia del modelo Kmeans al iniciar la app."""
    global model
    model = Kmeans(m=10)
    yield

# --- Configuración de la App ---
app = FastAPI(
    title="K-Means API",
    description="API para ejecutar el algoritmo K-Means paso a paso."
)


app = FastAPI(
    title="API de Intención de Voto (k-NN Puro)",
    description="API que entrena un modelo k-NN desde cero al iniciar.",
    lifespan=lifespan
)

allowed_origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://localhost:3000",
    "http://127.0.0.1:8000",
    "http://127.0.0.1:3000",
    "https://knn-votantes-frontend.onrender.com" # TODO: Cambiar al dominio real cuando esté desplegado
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Modelos de Datos (para Pydantic) ---
# Esto asegura que los datos que llegan del frontend tienen el tipo correcto
class ModelConfigM(BaseModel):
    m: int = Field(..., description="Tamaño de la cuadrícula")

class ModelConfigN(BaseModel):
    n: int = Field(..., description="Número de casas")

class ModelConfigA(BaseModel):
    A: int = Field(..., description="Número de hospitales")



# --- Endpoints de la API ---

@app.post("/config/set-grid")
def set_grid_size(config: ModelConfigM):
    """
    Define el tamaño 'm' de la cuadrícula. 
    Esto resetea todo el modelo.
    """
    global model # Usamos la instancia global
    # Creamos un nuevo modelo con el tamaño 'm'
    model = Kmeans(m=config.m)
    return {"message": f"Grid configurado a {config.m}x{config.m}.", "m": config.m}

@app.post("/init/generate-data")
def generate_data(config: ModelConfigN):
    """
    Genera 'n' puntos de datos (casas) aleatorios.
    Devuelve la lista de coordenadas de las casas.
    """
    try:
        data = model.random_data_init(n=config.n)
        # Convertimos de numpy a lista para que sea compatible con JSON
        return {"data": data.tolist(), "n": config.n}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/init/generate-hospitals")
def generate_hospitals(config: ModelConfigA):
    """
    Genera 'A' hospitales (centroides) aleatorios.
    Devuelve la lista de coordenadas de los hospitales.
    """

    if config.A is None:
        raise HTTPException(status_code=400, detail="Debe especificar el número de hospitales (A).")

    if config.A > model.m * model.m:
        raise HTTPException(status_code=400, detail="El número de hospitales (A) no puede ser mayor que m*m.")

    if config.A <= 0:
        raise HTTPException(status_code=400, detail="El número de hospitales (A) debe ser un entero positivo.")
         
    if config.A > model.n:
        raise HTTPException(status_code=400, detail="El número de hospitales (A) no puede ser mayor que el número de casas (n).")

    if config.A < 1:
        raise HTTPException(status_code=400, detail="El número de hospitales (A) debe ser mayor que cero.")
    

    try:
        hospitals = model.random_hospitals_init(A=config.A)
        # Convertimos de numpy a lista para que sea compatible con JSON
        return {"hospitals": hospitals.tolist(), "A": config.A}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/run/assign-clusters")
def assign_clusters_step():
    """
    Paso 1 del bucle K-Means: Asigna cada casa al hospital más cercano.
    Devuelve la lista de asignaciones (clusters).
    """
    try:
        clusters = model.assign_clusters()
        return {"clusters": clusters.tolist()}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/run/update-hospitals")
def update_hospitals_step():
    """
    Paso 2 del bucle K-Means: Mueve cada hospital al centro de las casas asignadas.
    Devuelve las nuevas coordenadas de los hospitales.
    """
    try:
        new_hospitals = model.update_hospitals()
        return {"hospitals": new_hospitals.tolist()}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/state")
def get_current_state():
    """Endpoint útil para que el frontend obtenga el estado actual completo."""
    if model.data is None:
        return {"state": False}
        
    return {
        "state": True,
        "m": model.m,
        "n": model.n,
        "A": model.A,
        "data": model.data.tolist() if model.data is not None else None,
        "hospitals": model.hospitals.tolist() if model.hospitals is not None else None,
        "clusters": model.clusters.tolist() if model.clusters is not None else None
    }