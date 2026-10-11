import os
import pandas as pd
import torch
import mlflow
from transformers import AutoTokenizer, MT5ForConditionalGeneration
from torch.optim import AdamW
from metricas_preservacion import calcular_similitud_tfidf, calcular_solapamiento_ner

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Hardware activo: {device}")

ruta_plos = os.path.join("datasets", "PLOS", "train-00000-of-00003.parquet") 
df_plos = pd.read_parquet(ruta_plos, engine='pyarrow')

batch_size = 3
# Agregamos un prefijo de tarea. Esto es CRÍTICO para los modelos de la familia T5
prefijo = "simplificar texto academico: " 
textos_tecnicos = [prefijo + texto for texto in df_plos['article'].head(batch_size).tolist()]
textos_divulgativos = df_plos['summary'].head(batch_size).tolist()

nombre_modelo = "google/mt5-small"
tokenizer = AutoTokenizer.from_pretrained(nombre_modelo)
modelo = MT5ForConditionalGeneration.from_pretrained(nombre_modelo, tie_word_embeddings=False).to(device)

optimizador = AdamW(modelo.parameters(), lr=3e-4) # Subimos un poco el learning rate para ver cambios rápidos
num_epocas = 80

mlflow.set_experiment("Transferencia_Estilo_Academico")

with mlflow.start_run(run_name="Overfitting_Sanity_Check"):
    mlflow.log_param("modelo", nombre_modelo)
    mlflow.log_param("epocas", num_epocas)
    
    # Tokenización fuera del bucle (los datos no cambian en esta prueba)
    inputs = tokenizer(textos_tecnicos, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    labels = tokenizer(textos_divulgativos, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
    
    labels_ids = labels["input_ids"].clone()
    labels_ids[labels_ids == tokenizer.pad_token_id] = -100

    print(f"\nIniciando entrenamiento por {num_epocas} épocas...")
    
    for epoca in range(num_epocas):
        modelo.train()
        optimizador.zero_grad()
        
        outputs = modelo(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], labels=labels_ids)
        loss = outputs.loss
        
        loss.backward()
        optimizador.step()
        
        # Validación cada 5 épocas y en la última
        if (epoca + 1) % 5 == 0 or epoca == num_epocas - 1:
            modelo.eval()
            with torch.no_grad():
                generados_ids = modelo.generate(
                    input_ids=inputs["input_ids"], 
                    attention_mask=inputs["attention_mask"], 
                    max_length=128, 
                    num_beams=4
                    repetition_penalty=2.5
                )
            
            textos_generados = tokenizer.batch_decode(generados_ids, skip_special_tokens=True)
            
            puntajes_tfidf = [calcular_similitud_tfidf(orig, gen) for orig, gen in zip(textos_tecnicos, textos_generados)]
            puntajes_ner = [calcular_solapamiento_ner(orig, gen) for orig, gen in zip(textos_tecnicos, textos_generados)]
            
            avg_tfidf = sum(puntajes_tfidf) / len(puntajes_tfidf)
            avg_ner = sum(puntajes_ner) / len(puntajes_ner)
            
            mlflow.log_metric("loss_entrenamiento", loss.item(), step=epoca)
            mlflow.log_metric("avg_similitud_tfidf", avg_tfidf, step=epoca)
            
            print(f"Época {epoca + 1}/{num_epocas} | Loss: {loss.item():.4f} | TF-IDF: {avg_tfidf:.4f} | NER: {avg_ner:.4f}")
            if (epoca + 1) % 10 == 0 or epoca == num_epocas - 1:
                print(f"\nEjemplo generado final:\n{textos_generados[0][:200]}...")