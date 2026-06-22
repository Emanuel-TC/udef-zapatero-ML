import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import Tuple

class NetworkRoleClassifier:
    """
    Clase para aplicar aprendizaje no supervisado (K-Means) sobre métricas topológicas 
    y clasificar automáticamente los roles de una red forense.
    """
    def __init__(self, n_clusters: int = 4, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        self.scaler = StandardScaler()
        
    def fit_predict(self, df_features: pd.DataFrame, feature_cols: list) -> Tuple[pd.DataFrame, float, pd.DataFrame]:
        """
        [Método Público] 
        Entrena el modelo, calcula métricas de validación y asigna etiquetas heurísticas.
        """
        # Filtramos nodos residuales que no aportan a la estructura principal
        df_model = df_features[(df_features['PageRank'] > 0.001) | (df_features['Betweenness'] > 0.001)].copy()
        
        # Escalado Crítico
        X = df_model[feature_cols]
        X_scaled = self.scaler.fit_transform(X)
        
        # Predicción
        df_model['Cluster_Role'] = self.kmeans.fit_predict(X_scaled)
        
        # Validación
        sil_score = silhouette_score(X_scaled, df_model['Cluster_Role'])
        
        # Perfilado para Etiquetado Heurístico
        perfil_clusters = df_model.groupby('Cluster_Role')[feature_cols].mean()
        
        # Asignación de Roles Automática (Lógica de Negocio)
        id_cupula = perfil_clusters['PageRank'].idxmax()
        id_brokers = perfil_clusters['Betweenness'].idxmax()
        
        def nombrar_rol(cluster_id):
            if cluster_id == id_cupula:
                return "Cúpula Directiva (Líderes)"
            elif cluster_id == id_brokers and cluster_id != id_cupula:
                return "Brokers / Intermediarios Clave"
            elif perfil_clusters.loc[cluster_id, 'In_Degree'] > perfil_clusters.loc[cluster_id, 'Out_Degree']:
                return "Receptores / Nodos de Influencia Pasiva"
            else:
                return "Operadores Periféricos / Ejecutores"
                
        df_model['Etiqueta_Forense'] = df_model['Cluster_Role'].apply(nombrar_rol)
        
        return df_model.reset_index(drop=True), sil_score, perfil_clusters