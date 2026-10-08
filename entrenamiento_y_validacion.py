import os
import pandas as pd
import torch
import mlflow
from transformers import AutoTokenizer, MT5ForConditionalGeneration
from torch.optim import AdamW

# Importar las métricas de preservación
from metricas_preservacion import calcular_similitud_tfidf, calcular_solapamiento_ner

# 1. Configuración de Hardware
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Hardware activo: {device}")

# 2. Carga de Datos (Usando una muestra pequeña para la prueba)
ruta_plos = os.path.join("datasets", "PLOS", "train-00000-of-00003.parquet") 
df_plos = pd.read_parquet(ruta_plos, engine='pyarrow')

batch_size = 3
textos_tecnicos = df_plos['article'].head(batch_size).tolist()
textos_divulgativos = df_plos['summary'].head(batch_size).tolist()

# 3. Inicialización del Tokenizador y del Modelo
nombre_modelo = "google/mt5-small"
tokenizer = AutoTokenizer.from_pretrained(nombre_modelo)

print("Cargando el modelo a la GPU...")
# Añadimos tie_word_embeddings=False para silenciar la advertencia de Hugging Face
modelo = MT5ForConditionalGeneration.from_pretrained(nombre_modelo, tie_word_embeddings=False).to(device)

# Configurar el Optimizador (AdamW es el estándar para transformadores)
optimizador = AdamW(modelo.parameters(), lr=5e-5)

# 4. Configurar MLflow
mlflow.set_experiment("Transferencia_Estilo_Academico")

with mlflow.start_run(run_name="Entrenamiento_y_Validacion_mT5"):
    mlflow.log_param("modelo", nombre_modelo)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("learning_rate", 5e-5)
    
    print("\n--- Fase de Entrenamiento ---")
    modelo.train() # Poner el modelo en modo entrenamiento
    optimizador.zero_grad() # Limpiar gradientes anteriores
    
    # Tokenización para entrenamiento
    inputs = tokenizer(textos_tecnicos, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    labels = tokenizer(textos_divulgativos, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
    
    # Reemplazar padding por -100 para la función de pérdida
    labels_ids = labels["input_ids"].clone()
    labels_ids[labels_ids == tokenizer.pad_token_id] = -100
    
    # Pasada hacia adelante (Forward Pass)
    outputs = modelo(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], labels=labels_ids)
    loss = outputs.loss
    
    # Pasada hacia atrás (Backward Pass) y actualización de pesos
    loss.backward()
    optimizador.step()
    
    mlflow.log_metric("loss_entrenamiento", loss.item())
    print(f"Pérdida (Loss) calculada y pesos actualizados: {loss.item():.4f}")
    
    print("\n--- Fase de Generación y Validación Semántica ---")
    modelo.eval() # Poner el modelo en modo evaluación (apaga dropout, etc.)
    with torch.no_grad(): # No calcular gradientes para ahorrar memoria
        generados_ids = modelo.generate(
            input_ids=inputs["input_ids"], 
            attention_mask=inputs["attention_mask"], 
            max_length=128, 
            num_beams=2 # Búsqueda de haz ligero para mejorar la coherencia
        )
    
    # Decodificar los tensores a texto legible
    textos_generados = tokenizer.batch_decode(generados_ids, skip_special_tokens=True)
    
    # Calcular métricas para el batch
    puntajes_tfidf = []
    puntajes_ner = []
    
    for original, generado in zip(textos_tecnicos, textos_generados):
        puntajes_tfidf.append(calcular_similitud_tfidf(original, generado))
        puntajes_ner.append(calcular_solapamiento_ner(original, generado))
        
    avg_tfidf = sum(puntajes_tfidf) / len(puntajes_tfidf)
    avg_ner = sum(puntajes_ner) / len(puntajes_ner)
    
    # Registrar métricas en MLflow
    mlflow.log_metric("avg_similitud_tfidf", avg_tfidf)
    mlflow.log_metric("avg_solapamiento_ner", avg_ner)
    
    print(f"Similitud TF-IDF Promedio: {avg_tfidf:.4f}")
    print(f"Solapamiento NER Promedio: {avg_ner:.4f}")
    
    print("\nEjemplo de texto generado en esta época:")
    print(textos_generados[0][:200] + "...") # Mostrar los primeros 200 caracteres
    
    print("\n¡Bucle de entrenamiento y validación completado con éxito!")