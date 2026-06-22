import pandas as pd
import spacy
import warnings
import networkx as nx

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
    

class GraphFeatureExtractor:
    """
    Clase responsable de convertir un DataFrame de aristas en un grafo de NetworkX
    y extraer métricas topológicas (Centralidad, PageRank) para cada nodo.
    """
    def __init__(self):
        # Usaremos un grafo dirigido (DiGraph) porque el flujo de información importa
        self.G = nx.DiGraph()
        
    def _build_networkx_graph(self, df_grafo: pd.DataFrame):
        """[Método Privado] Construye el objeto grafo en memoria."""
        self.G.clear()
        
        for _, row in df_grafo.iterrows():
            origen = row['Origen']
            destino = row['Destino']
            peso = row['Peso']
            tipo = row['Tipo_Relacion']
            
            # Si la arista ya existe, sumamos el peso al existente y actualizamos el tipo principal
            if self.G.has_edge(origen, destino):
                self.G[origen][destino]['weight'] += peso
            else:
                self.G.add_edge(origen, destino, weight=peso, type=tipo)

    def extract_features(self, df_grafo: pd.DataFrame) -> pd.DataFrame:
        """
        [Método Público] 
        Calcula las métricas de red y devuelve un DataFrame tabular (Nodos x Features).
        """
        print("Calculando métricas topológicas (Feature Engineering)...")
        self._build_networkx_graph(df_grafo)
        
        # 1. Degree Centrality (Popularidad base)
        # Usamos out_degree y in_degree porque es dirigido
        in_degree = nx.in_degree_centrality(self.G)
        out_degree = nx.out_degree_centrality(self.G)
        
        # 2. Betweenness Centrality (El perfil del Broker/Intermediario)
        # Usamos el peso invertido (1/peso) porque en NetworkX 'peso' suele significar 'coste' o 'distancia'
        # Para nosotros, mayor peso = más cercanos. Así que invertimos.
        # Aseguramos que el peso no sea 0 para evitar divisiones por cero.
        for u, v, d in self.G.edges(data=True):
             d['distance'] = 1.0 / d['weight'] if d['weight'] > 0 else 1.0
             
        betweenness = nx.betweenness_centrality(self.G, weight='distance')
        
        # 3. PageRank (Importancia jerárquica)
        pagerank = nx.pagerank(self.G, weight='weight')
        
        # 4. HITS (Hubs and Authorities)
        # Puede fallar si la red no converge, lo envolvemos en un try/except
        try:
            hubs, authorities = nx.hits(self.G)
        except nx.PowerIterationFailedConvergence:
            print("Advertencia: HITS no convergió. Rellenando con ceros.")
            hubs = {node: 0.0 for node in self.G.nodes()}
            authorities = {node: 0.0 for node in self.G.nodes()}

        # Consolidamos todo en un DataFrame
        features_list = []
        for node in self.G.nodes():
            features_list.append({
                "Nodo": node,
                "In_Degree": in_degree.get(node, 0.0),
                "Out_Degree": out_degree.get(node, 0.0),
                "Betweenness": betweenness.get(node, 0.0),
                "PageRank": pagerank.get(node, 0.0),
                "Hub_Score": hubs.get(node, 0.0),
                "Authority_Score": authorities.get(node, 0.0)
            })
            
        df_features = pd.DataFrame(features_list)
        return df_features.sort_values(by="PageRank", ascending=False).reset_index(drop=True)