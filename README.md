# Backend API - Clasificación de Audio ML/DL

API REST desarrollada con FastAPI para clasificación de audio usando modelos de Machine Learning y Deep Learning.

## Estructura del Proyecto

```
BE_ML/
├── main.py                 # Aplicación principal FastAPI
├── requirements.txt        # Dependencias del proyecto
├── .env.example           # Variables de entorno de ejemplo
├── API/
│   ├── __init__.py
│   └── apiML.py           # Endpoints de ML
└── resources/
    └── modelo1.h5         # Modelo entrenado
```

## Instalación

1. Crear entorno virtual:
```bash
python -m venv .venv
.venv\Scripts\activate
```

2. Instalar dependencias:
```bash
pip install -r requirements.txt
```

## Uso

### Iniciar el servidor

```bash
python main.py
```

El servidor estará disponible en: `http://localhost:8000`

### Endpoints disponibles

#### 1. Health Check
```
GET /health
```

#### 2. Información del modelo
```
GET /api/ml/model-info
```

#### 3. Predicción de audio
```
POST /api/ml/predict
Content-Type: multipart/form-data

Parámetros:
- file: Archivo de audio (wav, mp3, ogg, flac, m4a)
```

**Respuesta:**
```json
{
  "success": true,
  "prediction": {
    "class": 0,
    "label": "Clase 0",
    "confidence": 0.95
  },
  "probabilities": {
    "Clase 0": 0.95,
    "Clase 1": 0.03,
    "Clase 2": 0.02
  },
  "filename": "audio.wav"
}
```

## Integración con Frontend Flask

### Ejemplo de código para consumir la API desde Flask:

```python
import requests

@app.route('/classify', methods=['POST'])
def classify_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file'}), 400
    
    audio_file = request.files['audio']
    
    # Enviar al backend FastAPI
    files = {'file': (audio_file.filename, audio_file.stream, audio_file.mimetype)}
    response = requests.post('http://localhost:8000/api/ml/predict', files=files)
    
    return jsonify(response.json())
```

## Configuración del Modelo

### IMPORTANTE: Ajustar según tu modelo

En `API/apiML.py`, ajusta los siguientes parámetros según tu modelo:

1. **Etiquetas de clases** (línea ~115):
```python
class_labels = {
    0: "Clase 0",
    1: "Clase 1",
    2: "Clase 2",
}
```

2. **Función de preprocesamiento**:
   - Si tu modelo es **CNN**: usa `preprocess_audio_cnn()`
   - Si tu modelo es **denso/LSTM**: usa `preprocess_audio()`

3. **Parámetros de audio** (líneas 21-23):
```python
SAMPLE_RATE = 22050  # Frecuencia de muestreo
DURATION = 3         # Duración en segundos
N_MFCC = 40          # Número de coeficientes MFCC
```

## Documentación Interactiva

Una vez iniciado el servidor, accede a:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Notas

- El modelo debe estar en formato `.h5` (Keras/TensorFlow)
- Los archivos de audio se procesan temporalmente y se eliminan después de la predicción
- CORS está configurado para permitir todas las origines en desarrollo. En producción, especifica los dominios permitidos.

## Próximos pasos

1. Ajustar las etiquetas de clases según tu modelo
2. Verificar que los parámetros de preprocesamiento coincidan con los usados en el entrenamiento
3. Configurar CORS para producción
4. Agregar autenticación si es necesario
