"""
Ejemplo de cómo integrar el backend FastAPI con tu frontend Flask.
Copia estas funciones a tu aplicación Flask existente.
"""
from flask import Flask, request, jsonify, render_template
import requests
import os

app = Flask(__name__)

# URL del backend FastAPI
FASTAPI_BACKEND_URL = "http://localhost:8000"

@app.route('/')
def index():
    """Página principal del frontend."""
    return render_template('index.html')

@app.route('/upload-audio', methods=['POST'])
def upload_audio():
    """
    Endpoint del frontend Flask que recibe el audio del usuario
    y lo envía al backend FastAPI para clasificación.
    """
    # Verificar que se envió un archivo
    if 'audio' not in request.files:
        return jsonify({
            'error': 'No se envió ningún archivo de audio'
        }), 400
    
    audio_file = request.files['audio']
    
    # Verificar que el archivo tiene nombre
    if audio_file.filename == '':
        return jsonify({
            'error': 'Archivo sin nombre'
        }), 400
    
    try:
        # Preparar el archivo para enviarlo al backend
        files = {
            'file': (
                audio_file.filename,
                audio_file.stream,
                audio_file.mimetype or 'audio/wav'
            )
        }
        
        # Enviar al backend FastAPI
        response = requests.post(
            f"{FASTAPI_BACKEND_URL}/api/ml/predict",
            files=files,
            timeout=30  # timeout de 30 segundos
        )
        
        # Verificar respuesta
        if response.status_code == 200:
            result = response.json()
            return jsonify({
                'success': True,
                'prediction': result['prediction'],
                'probabilities': result['probabilities']
            })
        else:
            return jsonify({
                'success': False,
                'error': response.json().get('detail', 'Error desconocido')
            }), response.status_code
    
    except requests.exceptions.ConnectionError:
        return jsonify({
            'success': False,
            'error': 'No se pudo conectar con el servidor de ML. Asegúrate de que el backend esté corriendo.'
        }), 503
    
    except requests.exceptions.Timeout:
        return jsonify({
            'success': False,
            'error': 'La predicción tardó demasiado. Intenta con un archivo más pequeño.'
        }), 504
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error del servidor: {str(e)}'
        }), 500

@app.route('/model-status', methods=['GET'])
def model_status():
    """
    Verifica el estado del modelo en el backend.
    """
    try:
        response = requests.get(
            f"{FASTAPI_BACKEND_URL}/api/ml/model-info",
            timeout=5
        )
        
        if response.status_code == 200:
            return jsonify({
                'backend_online': True,
                'model_info': response.json()
            })
        else:
            return jsonify({
                'backend_online': False,
                'error': 'Backend no responde correctamente'
            }), 503
    
    except:
        return jsonify({
            'backend_online': False,
            'error': 'Backend no disponible'
        }), 503

@app.route('/health')
def health():
    """Health check del frontend."""
    return jsonify({
        'status': 'online',
        'service': 'frontend'
    })

if __name__ == '__main__':
    # En desarrollo
    app.run(debug=True, port=5000)
    
    # En producción, usa:
    # app.run(host='0.0.0.0', port=5000, debug=False)
