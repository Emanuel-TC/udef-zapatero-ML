import pandas as pd
import spacy
import warnings

class HeterogeneousGraphBuilder:
    """
    Clase responsable de construir la topología base del grafo (Nodos y Aristas)
    a partir de las conversaciones estructuradas.
    Construye grafos Multiplex (Múltiples tipos de relaciones: COMUNICA_CON, MENCIONA_A).
    """
    
    def __init__(self, spacy_model: str = "es_core_news_md"):
        warnings.filterwarnings("ignore")
        try:
            # Deshabilitamos componentes innecesarios para optimizar la velocidad del pipeline NER
            self.nlp = spacy.load(spacy_model, disable=["parser", "attribute_ruler", "lemmatizer"])
            print(f"Modelo NLP '{spacy_model}' cargado correctamente para extracción de entidades.")
        except OSError:
            print(f"Error: No se encontró el modelo {spacy_model}. Ejecuta en terminal: python -m spacy download {spacy_model}")
            self.nlp = None

    def build_raw_graph(self, df_chats: pd.DataFrame) -> pd.DataFrame:
        """
        [Método Público]
        Recorre el DataFrame cronológico de chats y extrae las relaciones matemáticas directas e indirectas.
        """
        aristas = []
        
        # 1. Limpieza y preparación de la copia de trabajo
        df = df_chats.dropna(subset=['Emisor', 'Mensaje']).copy()
        df['Emisor_Limpio'] = df['Emisor'].str.upper().str.strip()
        df = df.reset_index(drop=True)
        
        print("Analizando dinámicas conversacionales y construyendo aristas...")
        
        # Recorrido secuencial
        for i in range(len(df)):
            emisor_actual = df.loc[i, 'Emisor_Limpio']
            mensaje = str(df.loc[i, 'Mensaje'])
            pagina = df.loc[i, 'Pagina_PDF']
            
            # --- LÓGICA 1: ARISTAS DIRECTAS (Comunicación Secuencial) ---
            # Ventana de conversación: miramos hasta 3 mensajes hacia adelante
            for j in range(i + 1, min(i + 4, len(df))):
                emisor_siguiente = df.loc[j, 'Emisor_Limpio']
                
                # Inferencia de respuesta
                if emisor_actual != emisor_siguiente:
                    aristas.append({
                        "Origen": emisor_actual,
                        "Destino": emisor_siguiente,
                        "Tipo_Relacion": "COMUNICA_CON",
                        "Pagina": pagina
                    })
                    break 
                    
            # --- LÓGICA 2: ARISTAS INDIRECTAS (Menciones / NLP) ---
            if self.nlp:
                doc = self.nlp(mensaje)
                for ent in doc.ents:
                    # Filtro de ruido: Solo personas con nombres mayores a 3 caracteres
                    if ent.label_ == "PER" and len(ent.text) > 3:
                        mencionado = ent.text.upper().strip()
                        
                        # Evitar auto-bucles (ej. "Yo, Rodolfo...")
                        if emisor_actual != mencionado:
                            aristas.append({
                                "Origen": emisor_actual,
                                "Destino": mencionado,
                                "Tipo_Relacion": "MENCIONA_A",
                                "Pagina": pagina
                            })
                            
        # Consolidación inicial de pesos
        df_aristas_raw = pd.DataFrame(aristas)
        if not df_aristas_raw.empty:
            df_grafo = df_aristas_raw.groupby(['Origen', 'Destino', 'Tipo_Relacion']).size().reset_index(name='Peso')
            return df_grafo.sort_values(by='Peso', ascending=False).reset_index(drop=True)
        
        return pd.DataFrame(columns=["Origen", "Destino", "Tipo_Relacion", "Peso"])