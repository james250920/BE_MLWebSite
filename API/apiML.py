from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import librosa
import numpy as np
import os
import tempfile
from tensorflow import keras
from typing import Dict, Any

router = APIRouter()

# Cargar el modelo al iniciar
MODEL_PATH = "resources/modelo1.h5"
model = None

try:
    model = keras.models.load_model(MODEL_PATH)
    print(f"Modelo cargado exitosamente desde {MODEL_PATH}")
except Exception as e:
    print(f"Error al cargar el modelo: {str(e)}")

# Configuración de preprocesamiento de audio
SAMPLE_RATE = 22050
DURATION = 3  # segundos
N_MFCC = 40

def preprocess_audio(file_path: str) -> np.ndarray:
    """
    Preprocesa el archivo de audio para la predicción.
    Extrae características MFCC del audio.
    """
    try:
        # Cargar audio
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
        
        # Extraer características MFCC
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
        
        # Calcular estadísticas para reducir dimensionalidad
        mfccs_mean = np.mean(mfccs, axis=1)
        mfccs_std = np.std(mfccs, axis=1)
        
        # Concatenar características
        features = np.concatenate([mfccs_mean, mfccs_std])
        
        return features
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al procesar audio: {str(e)}")

def preprocess_audio_cnn(file_path: str) -> np.ndarray:
    """
    Preprocesa el audio para modelos CNN (espectrograma o MFCC 2D).
    """
    try:
        # Cargar audio
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
        
        # Extraer MFCC como imagen 2D
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
        
        # Normalizar
        mfccs = (mfccs - np.mean(mfccs)) / np.std(mfccs)
        
        # Ajustar forma para CNN (agregar canal)
        mfccs = np.expand_dims(mfccs, axis=-1)
        
        return mfccs
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al procesar audio: {str(e)}")

@router.post("/predict")
async def predict_audio(
    file: UploadFile = File(..., description="Archivo de audio (wav, mp3, etc.)")
) -> Dict[str, Any]:
    """
    Endpoint para clasificar un archivo de audio.
    
    Args:
        file: Archivo de audio subido por el cliente
    
    Returns:
        Diccionario con la predicción y probabilidades
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="El modelo no está disponible. Verifica que el archivo modelo1.h5 existe."
        )
    
    # Validar tipo de archivo
    allowed_extensions = [".wav", ".mp3", ".ogg", ".flac", ".m4a"]
    file_extension = os.path.splitext(file.filename)[1].lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Formato de archivo no soportado. Use: {', '.join(allowed_extensions)}"
        )
    
    # Guardar archivo temporal
    temp_file = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
            content = await file.read()
            tmp.write(content)
            temp_file = tmp.name
        
        # Preprocesar audio - ajusta según tu modelo
        # Si tu modelo usa CNN, usa preprocess_audio_cnn
        # Si es un modelo denso/LSTM, usa preprocess_audio
        features = preprocess_audio(temp_file)
        
        # Preparar para predicción
        features = np.expand_dims(features, axis=0)
        
        # Realizar predicción
        prediction = model.predict(features, verbose=0)
        
        # Procesar resultados
        predicted_class = int(np.argmax(prediction[0]))
        confidence = float(np.max(prediction[0]))
        
        # Etiquetas de clases - AJUSTA SEGÚN TU MODELO
        class_labels = {
            0: "Clase 0",
            1: "Clase 1",
            2: "Clase 2",
            # Agrega más clases según tu modelo
        }
        
        predicted_label = class_labels.get(predicted_class, f"Clase {predicted_class}")
        
        # Crear diccionario de probabilidades
        probabilities = {
            class_labels.get(i, f"Clase {i}"): float(prob)
            for i, prob in enumerate(prediction[0])
        }
        
        return JSONResponse(content={
            "success": True,
            "prediction": {
                "class": predicted_class,
                "label": predicted_label,
                "confidence": confidence
            },
            "probabilities": probabilities,
            "filename": file.filename
        })
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error durante la predicción: {str(e)}"
        )
    
    finally:
        # Limpiar archivo temporal
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
            except:
                pass

@router.get("/model-info")
async def model_info() -> Dict[str, Any]:
    """
    Obtiene información sobre el modelo cargado.
    """
    if model is None:
        return {
            "loaded": False,
            "message": "Modelo no disponible"
        }
    
    try:
        return {
            "loaded": True,
            "input_shape": str(model.input_shape),
            "output_shape": str(model.output_shape),
            "layers": len(model.layers)
        }
    except Exception as e:
        return {
            "loaded": True,
            "error": str(e)
        }