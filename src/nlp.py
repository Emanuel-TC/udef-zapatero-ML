import pandas as pd
from pysentimiento import create_analyzer
import warnings
import os

# Suprimimos warnings de Hugging Face
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

class TextEmotionAnalyzer:
    """
    Clase que implementa Deep Learning (Transformers - RoBERTa) 
    para analizar el sentimiento y la emoción de los textos forenses en español.
    """
    def __init__(self):
        print("Cargando modelos Transformer (RoBERTa) en memoria... Esto puede tardar un poco la primera vez.")
        self.sentiment_analyzer = create_analyzer(task="sentiment", lang="es")
        self.emotion_analyzer = create_analyzer(task="emotion", lang="es")
        print("Modelos NLP listos.")

    def analyze_dataframe(self, df: pd.DataFrame, text_column: str = 'Mensaje') -> pd.DataFrame:
        """
        [Método Público] Aplica inferencia de Deep Learning a todo el dataset.
        """
        df_enriched = df.copy()
        
        sentimientos = []
        emociones = []
        
        print(f"Analizando {len(df_enriched)} mensajes con Deep Learning...")
        
        for idx, row in df_enriched.iterrows():
            texto = str(row[text_column])
            
            # Si el texto es muy corto o nulo, asignamos neutro por defecto
            if len(texto) < 2 or texto == "nan":
                sentimientos.append("NEU")
                emociones.append("others")
                continue
                
            # Inferencia del Transformer
            res_sent = self.sentiment_analyzer.predict(texto)
            res_emo = self.emotion_analyzer.predict(texto)
            
            sentimientos.append(res_sent.output)
            emociones.append(res_emo.output)
            
        df_enriched['NLP_Sentimiento'] = sentimientos
        df_enriched['NLP_Emocion'] = emociones
        
        return df_enriched