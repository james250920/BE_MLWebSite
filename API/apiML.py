from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
import librosa
import numpy as np
import os
import tempfile
from tensorflow import keras
from typing import Dict, Any
from pydub import AudioSegment

router = APIRouter()

MODEL_PATH = "resources/deepfake_detector_cnn.h5"
model = None

try:
    model = keras.models.load_model(MODEL_PATH)
    print(f"Modelo cargado exitosamente desde {MODEL_PATH}")
except Exception as e:
    print(f"Error al cargar el modelo: {str(e)}")

SR = 16000
DURATION = 3.0
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128

def preprocess_audio(file_path: str) -> np.ndarray:
    try:
        audio, sample_rate = librosa.load(file_path, sr=SR, duration=DURATION)
        
        target_length = int(SR * DURATION)
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
        else:
            audio = audio[:target_length]
        
        mel_spec = librosa.feature.melspectrogram(
            y=audio, 
            sr=SR, 
            n_fft=N_FFT, 
            hop_length=HOP_LENGTH,
            n_mels=N_MELS
        )
        
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        mel_spec_db = np.nan_to_num(mel_spec_db)
        
        mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
        
        mel_spec_cnn = np.expand_dims(mel_spec_norm, axis=-1)
        
        return mel_spec_cnn
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al procesar audio: {str(e)}")

@router.post("/predict")
async def predict_audio(
    file: UploadFile = File(..., description="Archivo de audio (wav, mp3, flac, ogg)")
) -> Dict[str, Any]:
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="El modelo no está disponible. Verifica que el archivo deepfake_detector_cnn.h5 existe."
        )
    
    allowed_extensions = [".wav", ".mp3", ".ogg", ".flac"]
    file_extension = os.path.splitext(file.filename)[1].lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Formato de archivo no soportado. Use: {', '.join(allowed_extensions)}"
        )
    
    temp_file = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
            content = await file.read()
            tmp.write(content)
            temp_file = tmp.name
        
        features = preprocess_audio(temp_file)
        
        features = np.expand_dims(features, axis=0)
        
        prediction = model.predict(features, verbose=0)
        
        probability_fake = float(prediction[0][0])
        probability_real = 1.0 - probability_fake
        
        is_fake = probability_fake > 0.5
        predicted_class = 1 if is_fake else 0
        predicted_label = "DEEPFAKE" if is_fake else "REAL"
        confidence = probability_fake if is_fake else probability_real
        
        return JSONResponse(content={
            "success": True,
            "prediction": {
                "class": predicted_class,
                "label": predicted_label,
                "confidence": round(confidence * 100, 2),
                "is_deepfake": is_fake
            },
            "probabilities": {
                "REAL": round(probability_real * 100, 2),
                "DEEPFAKE": round(probability_fake * 100, 2)
            },
            "audio_info": {
                "filename": file.filename,
                "duration_seconds": DURATION,
                "sample_rate": SR
            }
        })
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error durante la predicción: {str(e)}"
        )
    
    finally:
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
            except:
                pass

@router.get("/model-info")
async def model_info() -> Dict[str, Any]:
    if model is None:
        return {
            "loaded": False,
            "message": "Modelo no disponible"
        }
    
    try:
        return {
            "loaded": True,
            "model_type": "CNN para detección de deepfakes",
            "input_shape": str(model.input_shape),
            "output_shape": str(model.output_shape),
            "layers": len(model.layers),
            "parameters": {
                "sample_rate": SR,
                "duration": DURATION,
                "n_mels": N_MELS,
                "n_fft": N_FFT,
                "hop_length": HOP_LENGTH
            },
            "classes": {
                "0": "REAL",
                "1": "DEEPFAKE"
            }
        }
    except Exception as e:
        return {
            "loaded": True,
            "error": str(e)
        }

@router.get("/health")
async def health_check() -> Dict[str, Any]:
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "version": "1.0.0"
    }

@router.post("/convert-to-wav")
async def convert_to_wav(
    file: UploadFile = File(..., description="Archivo de audio a convertir a WAV")
):
    """
    Convierte cualquier archivo de audio a formato WAV.
    
    Formatos soportados: mp3, opus, m4a, flac, ogg, aac, webm, etc.
    
    Returns:
        Archivo WAV convertido para descargar
    """
    allowed_extensions = [".mp3", ".opus", ".m4a", ".flac", ".ogg", ".aac", ".webm", ".wma", ".wav"]
    file_extension = os.path.splitext(file.filename)[1].lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Formato no soportado. Formatos válidos: {', '.join(allowed_extensions)}"
        )
    
    temp_input = None
    temp_output = None
    
    try:
        # Guardar archivo de entrada temporal
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_in:
            content = await file.read()
            tmp_in.write(content)
            temp_input = tmp_in.name
        
        # Crear archivo de salida temporal
        temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.wav').name
        
        # Si ya es WAV, solo copiarlo
        if file_extension == '.wav':
            import shutil
            shutil.copy(temp_input, temp_output)
            output_filename = file.filename
        else:
            # Convertir a WAV usando pydub
            audio = AudioSegment.from_file(temp_input)
            audio.export(temp_output, format='wav')
            
            # Generar nombre de salida
            base_name = os.path.splitext(file.filename)[0]
            output_filename = f"{base_name}.wav"
        
        # Obtener información del archivo convertido
        file_size = os.path.getsize(temp_output)
        
        # Retornar el archivo como descarga
        return FileResponse(
            path=temp_output,
            media_type='audio/wav',
            filename=output_filename,
            headers={
                "Content-Disposition": f"attachment; filename={output_filename}",
                "X-Original-Filename": file.filename,
                "X-File-Size": str(file_size)
            },
            background=lambda: cleanup_files([temp_input, temp_output])
        )
    
    except Exception as e:
        # Limpiar archivos en caso de error
        if temp_input and os.path.exists(temp_input):
            try:
                os.unlink(temp_input)
            except:
                pass
        if temp_output and os.path.exists(temp_output):
            try:
                os.unlink(temp_output)
            except:
                pass
        
        raise HTTPException(
            status_code=500,
            detail=f"Error al convertir el audio: {str(e)}"
        )

def cleanup_files(file_paths: list):
    """Función auxiliar para limpiar archivos temporales después de la descarga."""
    for file_path in file_paths:
        if file_path and os.path.exists(file_path):
            try:
                os.unlink(file_path)
            except:
                pass

@router.post("/convert-and-predict")
async def convert_and_predict(
    file: UploadFile = File(..., description="Archivo de audio (cualquier formato)")
) -> Dict[str, Any]:
    """
    Convierte el audio a WAV (si es necesario) y luego realiza la predicción.
    
    Este endpoint combina la conversión y predicción en un solo paso.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="El modelo no está disponible. Verifica que el archivo deepfake_detector_cnn.h5 exists."
        )
    
    allowed_extensions = [".wav", ".mp3", ".opus", ".m4a", ".flac", ".ogg", ".aac", ".webm"]
    file_extension = os.path.splitext(file.filename)[1].lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Formato no soportado. Use: {', '.join(allowed_extensions)}"
        )
    
    temp_input = None
    temp_wav = None
    
    try:
        # Guardar archivo original
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
            content = await file.read()
            tmp.write(content)
            temp_input = tmp.name
        
        # Convertir a WAV si es necesario
        if file_extension != '.wav':
            temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix='.wav').name
            audio = AudioSegment.from_file(temp_input)
            audio.export(temp_wav, format='wav')
            processing_file = temp_wav
            converted = True
        else:
            processing_file = temp_input
            converted = False
        
        # Procesar el audio
        features = preprocess_audio(processing_file)
        features = np.expand_dims(features, axis=0)
        
        # Realizar predicción
        prediction = model.predict(features, verbose=0)
        
        probability_fake = float(prediction[0][0])
        probability_real = 1.0 - probability_fake
        
        is_fake = probability_fake > 0.5
        predicted_class = 1 if is_fake else 0
        predicted_label = "DEEPFAKE" if is_fake else "REAL"
        confidence = probability_fake if is_fake else probability_real
        
        return JSONResponse(content={
            "success": True,
            "prediction": {
                "class": predicted_class,
                "label": predicted_label,
                "confidence": round(confidence * 100, 2),
                "is_deepfake": is_fake
            },
            "probabilities": {
                "REAL": round(probability_real * 100, 2),
                "DEEPFAKE": round(probability_fake * 100, 2)
            },
            "audio_info": {
                "filename": file.filename,
                "original_format": file_extension,
                "converted_to_wav": converted,
                "duration_seconds": DURATION,
                "sample_rate": SR
            }
        })
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error durante el proceso: {str(e)}"
        )
    
    finally:
        # Limpiar archivos temporales
        for temp_file in [temp_input, temp_wav]:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except:
                    pass