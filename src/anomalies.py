import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

class TemporalAnomalyDetector:
    """
    Clase para la detección no supervisada de anomalías en series de tiempo y comportamiento 
    utilizando Isolation Forest.
    """
    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        # Contamination define el % estimado de anomalías (ej. 5% de los chats)
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(
            contamination=self.contamination, 
            random_state=self.random_state,
            n_jobs=-1 # Paralelización máxima
        )
        self.scaler = StandardScaler()

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        [Método Público] 
        Transforma los metadatos en crudo en variables matemáticas de comportamiento.
        """
        print("Realizando Feature Engineering temporal...")
        df_clean = df.dropna(subset=['Fecha', 'Hora', 'Mensaje']).copy()
        
        # 1. Parseo Robusto de Datetime
        # Se asume formato europeo (dayfirst=True)
        df_clean['Datetime'] = pd.to_datetime(
            df_clean['Fecha'] + ' ' + df_clean['Hora'], 
            dayfirst=True, 
            errors='coerce'
        )
        # Descartamos errores de parseo y ordenamos cronológicamente
        df_clean = df_clean.dropna(subset=['Datetime']).sort_values('Datetime').reset_index(drop=True)
        
        # 2. Variables de Comportamiento Básico
        df_clean['Hour'] = df_clean['Datetime'].dt.hour
        df_clean['DayOfWeek'] = df_clean['Datetime'].dt.dayofweek
        df_clean['Is_Weekend'] = df_clean['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
        df_clean['Message_Length'] = df_clean['Mensaje'].apply(lambda x: len(str(x)))
        
        # 3. Variable de Tensión / Frenesí (Delta temporal en segundos)
        # Calcula cuánto tardó en enviarse este mensaje respecto al anterior en la red global
        df_clean['Time_Delta_Sec'] = df_clean['Datetime'].diff().dt.total_seconds().fillna(0)
        
        return df_clean

    def fit_predict(self, df_features: pd.DataFrame, features_cols: list) -> pd.DataFrame:
        """
        [Método Público] 
        Escala las variables y ejecuta el modelo Isolation Forest.
        """
        print(f"Entrenando Isolation Forest sobre {len(df_features)} eventos...")
        X = df_features[features_cols]
        
        # El escalado es fundamental porque 'Time_Delta_Sec' y 'Hour' tienen magnitudes distintas
        X_scaled = self.scaler.fit_transform(X)
        
        # Predicción: -1 indica Anomalía, 1 indica Normal
        preds = self.model.fit_predict(X_scaled)
        df_features['Is_Anomaly'] = preds
        
        # Calculamos el Score de Anomalía (valores más negativos = más anómalo/raro)
        df_features['Anomaly_Score'] = self.model.decision_function(X_scaled)
        
        return df_features