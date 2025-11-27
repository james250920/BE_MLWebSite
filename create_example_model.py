"""
Script para crear un modelo de ejemplo para probar la API.
Este modelo es solo para demostración. Reemplázalo con tu modelo real entrenado.
"""
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

def create_example_model_dense():
    """
    Crea un modelo denso simple para clasificación de audio.
    Asume que las características de audio son vectores 1D (por ejemplo, MFCC agregados).
    """
    model = keras.Sequential([
        layers.Input(shape=(80,)),  # 40 MFCC mean + 40 MFCC std
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dense(3, activation='softmax')  # 3 clases de ejemplo
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_example_model_cnn():
    """
    Crea un modelo CNN simple para clasificación de audio.
    Asume que las características son espectrogramas o MFCC 2D.
    """
    model = keras.Sequential([
        layers.Input(shape=(40, 130, 1)),  # MFCC como imagen
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(3, activation='softmax')  # 3 clases de ejemplo
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

if __name__ == "__main__":
    import os
    
    print("Creando modelo de ejemplo...")
    
    # Elegir el tipo de modelo (dense o cnn)
    model_type = "dense"  # Cambia a "cnn" si tu modelo es CNN
    
    if model_type == "dense":
        model = create_example_model_dense()
        print("✅ Modelo denso creado")
    else:
        model = create_example_model_cnn()
        print("✅ Modelo CNN creado")
    
    # Mostrar resumen
    print("\nResumen del modelo:")
    model.summary()
    
    # Guardar el modelo
    save_path = "resources/modelo1.h5"
    os.makedirs("resources", exist_ok=True)
    model.save(save_path)
    
    print(f"\n✅ Modelo guardado en: {save_path}")
    print("\n⚠️  IMPORTANTE: Este es un modelo de ejemplo NO ENTRENADO.")
    print("   Las predicciones serán aleatorias.")
    print("   Reemplaza este modelo con tu modelo entrenado real.")
    
    # Probar que el modelo se puede cargar
    print("\nVerificando que el modelo se puede cargar...")
    loaded_model = keras.models.load_model(save_path)
    print("✅ Modelo cargado exitosamente")
    
    # Probar predicción con datos aleatorios
    print("\nProbando predicción con datos aleatorios...")
    if model_type == "dense":
        dummy_input = np.random.randn(1, 80)
    else:
        dummy_input = np.random.randn(1, 40, 130, 1)
    
    prediction = loaded_model.predict(dummy_input, verbose=0)
    print(f"Salida del modelo: {prediction}")
    print(f"Clase predicha: {np.argmax(prediction[0])}")
    print(f"Confianza: {np.max(prediction[0]):.2%}")
