"""
Script de prueba para consumir la API desde el frontend Flask.
Ejemplo de cómo enviar un archivo de audio al backend.
"""
import requests

def test_audio_prediction(audio_file_path: str):
    """
    Prueba el endpoint de predicción de audio.
    
    Args:
        audio_file_path: Ruta al archivo de audio
    """
    url = "http://localhost:8000/api/ml/predict"
    
    try:
        # Abrir el archivo de audio
        with open(audio_file_path, 'rb') as audio_file:
            files = {'file': (audio_file_path.split('\\')[-1], audio_file, 'audio/wav')}
            
            # Enviar petición POST
            response = requests.post(url, files=files)
            
            # Mostrar resultados
            if response.status_code == 200:
                result = response.json()
                print("✅ Predicción exitosa!")
                print(f"\nClase predicha: {result['prediction']['label']}")
                print(f"Confianza: {result['prediction']['confidence']:.2%}")
                print("\nProbabilidades:")
                for clase, prob in result['probabilities'].items():
                    print(f"  {clase}: {prob:.2%}")
            else:
                print(f"❌ Error: {response.status_code}")
                print(response.json())
    
    except FileNotFoundError:
        print(f"❌ Archivo no encontrado: {audio_file_path}")
    except requests.exceptions.ConnectionError:
        print("❌ No se pudo conectar al servidor. Asegúrate de que esté corriendo.")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def test_health():
    """Prueba el endpoint de health check."""
    url = "http://localhost:8000/health"
    
    try:
        response = requests.get(url)
        print(f"Health check: {response.json()}")
    except:
        print("❌ Servidor no disponible")

def test_model_info():
    """Obtiene información del modelo."""
    url = "http://localhost:8000/api/ml/model-info"
    
    try:
        response = requests.get(url)
        print(f"Info del modelo: {response.json()}")
    except:
        print("❌ No se pudo obtener info del modelo")

if __name__ == "__main__":
    print("=== Pruebas del API de Audio ML/DL ===\n")
    
    # Probar health check
    test_health()
    print()
    
    # Probar info del modelo
    test_model_info()
    print()
    
    # Probar predicción (ajusta la ruta al audio)
    # test_audio_prediction("ruta/al/audio.wav")
    print("Para probar la predicción, descomenta la última línea y proporciona la ruta de un archivo de audio.")
