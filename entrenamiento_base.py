import os
import pandas as pd
import torch
import mlflow
from transformers import AutoTokenizer, MT5ForConditionalGeneration

# 1. Configuración de Hardware
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Hardware activo: {device}")

# 2. Carga de Datos
ruta_plos = os.path.join("datasets", "PLOS", "train-00000-of-00003.parquet") 
df_plos = pd.read_parquet(ruta_plos, engine='pyarrow')

textos_tecnicos = df_plos['article'].head(3).tolist()
textos_divulgativos = df_plos['summary'].head(3).tolist()

# 3. Inicialización del Tokenizador y del Modelo
nombre_modelo = "google/mt5-small"
tokenizer = AutoTokenizer.from_pretrained(nombre_modelo)

print("Cargando el modelo a la GPU (esto puede tomar unos segundos)...")
modelo = MT5ForConditionalGeneration.from_pretrained(nombre_modelo).to(device)

# 4. Tokenización Dual (Entradas y Etiquetas)
inputs = tokenizer(textos_tecnicos, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
labels = tokenizer(textos_divulgativos, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)

# Ajuste crítico: Reemplazar los tokens de padding en las etiquetas con -100
# Esto indica a PyTorch que ignore el padding al calcular la función de pérdida
labels_ids = labels["input_ids"].clone()
labels_ids[labels_ids == tokenizer.pad_token_id] = -100

# 5. Configurar MLflow para el Tracking
mlflow.set_experiment("Transferencia_Estilo_Academico")

with mlflow.start_run(run_name="Prueba_Inicial_mT5"):
    # Registrar parámetros del experimento
    mlflow.log_param("modelo", nombre_modelo)
    mlflow.log_param("batch_size", len(textos_tecnicos))
    mlflow.log_param("max_length_input", 512)
    
    print("\nEjecutando Forward Pass...")
    # Pasada hacia adelante
    outputs = modelo(
        input_ids=inputs["input_ids"], 
        attention_mask=inputs["attention_mask"], 
        labels=labels_ids
    )
    
    loss = outputs.loss
    
    # Registrar la métrica resultante
    mlflow.log_metric("loss_inicial", loss.item())
    
    print(f"Pérdida (Loss) calculada: {loss.item():.4f}")
    print("¡Ciclo completado y registrado exitosamente en MLflow!")