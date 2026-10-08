import os
import pandas as pd
import torch
from transformers import AutoTokenizer

# 1. Configurar la aceleración por hardware
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo activo: {device}")

# 2. Rutas a los datasets locales
# Asegúrate de que el nombre del archivo coincida exactamente con el que se descargo
ruta_plos = os.path.join("datasets", "PLOS", "train.parquet") 

try:
    # Leer el dataset usando el motor optimizado pyarrow
    df_plos = pd.read_parquet(ruta_plos, engine='pyarrow')
    
    # Extraer una pequeña muestra (batch) para verificar el flujo
    # Reemplaza 'article' y 'lay_summary' si los nombres de columna varían en el dataset 202X (en 2025 deverian ser estos)
    textos_tecnicos = df_plos['article'].head(3).tolist()
    textos_divulgativos = df_plos['lay_summary'].head(3).tolist()
    
    print(f"\nSe cargaron {len(textos_tecnicos)} pares de texto para la prueba.")

    # 3. Inicializar el tokenizador (mT5 es excelente para transferencia de estilo multilingüe)
    nombre_modelo = "google/mt5-small"
    print(f"Descargando/Cargando tokenizador: {nombre_modelo}...")
    tokenizer = AutoTokenizer.from_pretrained(nombre_modelo)

    # 4. Tokenización y envío a la GPU
    # max_length recorta los textos muy largos para evitar desbordar la VRAM de la GPU
    inputs = tokenizer(
        textos_tecnicos, 
        padding=True, 
        truncation=True, 
        max_length=512, 
        return_tensors="pt"
    )
    
    # Transferir los tensores de la RAM a la memoria de la RTX 5060
    inputs = inputs.to(device)

    print("\n--- Resultado de la Tokenización ---")
    print(f"Dimensiones del tensor de entrada (Batch Size, Sequence Length): {inputs['input_ids'].shape}")
    print(f"Ubicación actual del tensor: {inputs['input_ids'].device}")

except FileNotFoundError:
    print(f"No se encontró el archivo en la ruta: {ruta_plos}")
except KeyError as e:
    print(f"Error de columna: Verifica los nombres de las columnas en el dataframe. Detalle: {e}")