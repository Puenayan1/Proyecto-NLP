import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Cargar el modelo de NLP para extracción de entidades (NER)
# Nota: Para la fase en español del proyecto, cambiar a "es_core_news_md"
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Error: El modelo de spaCy no está instalado. Ejecuta: python -m spacy download en_core_web_sm")

def calcular_similitud_tfidf(texto_original: str, texto_generado: str) -> float:
    """
    Calcula la similitud coseno entre dos textos utilizando representaciones vectoriales TF-IDF.
    Devuelve un valor entre 0.0 (completamente distintos) y 1.0 (idénticos).
    """
    # Si alguno de los textos está vacío, la similitud es 0
    if not texto_original.strip() or not texto_generado.strip():
        return 0.0

    vectorizer = TfidfVectorizer()
    try:
        # Vectorizar ambos textos en el mismo espacio
        tfidf_matrix = vectorizer.fit_transform([texto_original, texto_generado])
        # Calcular la similitud coseno entre el vector 0 (original) y el vector 1 (generado)
        similitud = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return float(similitud)
    except ValueError:
        # Captura errores si los textos no contienen vocabulario válido
        return 0.0

def calcular_solapamiento_ner(texto_original: str, texto_generado: str) -> float:
    """
    Calcula la proporción de entidades nombradas (NER) del texto original 
    que se lograron preservar en el texto generado.
    Devuelve un valor entre 0.0 (ninguna preservada) y 1.0 (todas preservadas).
    """
    doc_orig = nlp(texto_original)
    doc_gen = nlp(texto_generado)

    # Extraer entidades en minúsculas para evitar falsos negativos por capitalización
    entidades_orig = set([ent.text.lower() for ent in doc_orig.ents])
    entidades_gen = set([ent.text.lower() for ent in doc_gen.ents])

    # Si el texto original no tenía entidades, asumimos preservación perfecta
    if not entidades_orig:
        return 1.0

    # Calcular la intersección (entidades que sobrevivieron a la transformación)
    entidades_preservadas = entidades_orig.intersection(entidades_gen)
    
    # Calcular el ratio de solapamiento
    ratio = len(entidades_preservadas) / len(entidades_orig)
    return float(ratio)

# ==========================================
# Bloque de prueba local
# ==========================================
if __name__ == "__main__":
    # Textos de prueba simulando una transformación ("Despaperizar")
    texto_tecnico = "The objective of this randomized controlled trial is to evaluate the efficacy of the modified pedagogical intervention."
    texto_simple = "This trial aims to check if the new teaching intervention works well."

    similitud = calcular_similitud_tfidf(texto_tecnico, texto_simple)
    solapamiento = calcular_solapamiento_ner(texto_tecnico, texto_simple)

    print("--- Resultados de Validación Semántica ---")
    print(f"Texto Original: {texto_tecnico}")
    print(f"Texto Generado: {texto_simple}")
    print("-" * 40)
    print(f"Similitud Coseno (TF-IDF): {similitud:.4f} (Ideal: > 0.5 para estilos distintos)")
    print(f"Solapamiento NER:        {solapamiento:.4f} (Ideal: > 0.7 para conservar conceptos)")