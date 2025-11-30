from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Crear la aplicación FastAPI
app = FastAPI(
    title="Audio ML/DL API",
    description="API para clasificación de audio usando modelos de Machine Learning y Deep Learning",
    version="1.0.0"
)

# Configuración CORS para permitir peticiones desde el frontend Flask
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especifica el dominio de tu frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": "API de clasificación de audio y conversión",
        "status": "online",
        "endpoints": {
            "prediction": "/api/dl/predict",
            "conversion": "/api/dl/convert-to-wav",
            "convert_and_predict": "/api/dl/convert-and-predict",
            "model_info": "/api/dl/model-info",
            "health": "/api/dl/health",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Importar y registrar el router después de crear la app
# Esto evita problemas de importación circular con TensorFlow en Windows
from API.apiML import router as ml_router
app.include_router(ml_router, prefix="/api/dl", tags=["Deep Learning"])

if __name__ == "__main__":
    # Usar workers=1 para evitar problemas con TensorFlow en Windows
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False, workers=1)