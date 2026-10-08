import os
import pandas as pd
import torch
from transformers import AutoTokenizer

# 1. Configurar la aceleración por hardware
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo activo: {device}")

ruta_plos = os.path.join("datasets", "PLOS", "train-00000-of-00003.parquet") 

try:
    # Leer el dataset usando el motor optimizado pyarrow
    df_plos = pd.read_parquet(ruta_plos, engine='pyarrow')
    
    # 2. Imprimir las columnas disponibles para depuración
    print("\nColumnas detectadas en el archivo Parquet:")
    print(df_plos.columns.tolist())
    
    # Intento de adivinar posibles nombres de columna (puedes ajustar esto luego)
    col_texto = 'article' if 'article' in df_plos.columns else df_plos.columns[0]
    # Busca alguna columna que contenga 'summary' o 'lay'
    col_resumen = next((col for col in df_plos.columns if 'summary' in col.lower() or 'lay' in col.lower()), df_plos.columns[1])
    
    print(f"\nUsando '{col_texto}' como texto original y '{col_resumen}' como resumen.")

    # Extraer la muestra
    textos_tecnicos = df_plos[col_texto].head(3).tolist()
    textos_divulgativos = df_plos[col_resumen].head(3).tolist()

    # 3. Inicializar el tokenizador
    nombre_modelo = "google/mt5-small"
    print(f"\nDescargando/Cargando tokenizador: {nombre_modelo}...")
    tokenizer = AutoTokenizer.from_pretrained(nombre_modelo)

    # 4. Tokenización y envío a la GPU
    inputs = tokenizer(
        textos_tecnicos, 
        padding=True, 
        truncation=True, 
        max_length=512, 
        return_tensors="pt"
    )
    
    inputs = inputs.to(device)

    print("\n--- Resultado de la Tokenización ---")
    print(f"Dimensiones del tensor de entrada: {inputs['input_ids'].shape}")
    print(f"Ubicación actual del tensor: {inputs['input_ids'].device}")

except FileNotFoundError:
    print(f"No se encontró el archivo en la ruta: {ruta_plos}")
except Exception as e:
    print(f"Error inesperado: {e}")