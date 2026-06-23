import pandas as pd
import numpy as np
import re
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
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(
            contamination=self.contamination, 
            random_state=self.random_state,
            n_jobs=-1
        )
        self.scaler = StandardScaler()

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        print("Realizando Feature Engineering temporal...")
        df_clean = df.dropna(subset=['Fecha', 'Hora', 'Mensaje']).copy()
        
        df_clean['Datetime'] = pd.to_datetime(
            df_clean['Fecha'] + ' ' + df_clean['Hora'], 
            dayfirst=True, 
            errors='coerce'
        )
        df_clean = df_clean.dropna(subset=['Datetime']).sort_values('Datetime').reset_index(drop=True)
        
        df_clean['Hour'] = df_clean['Datetime'].dt.hour
        df_clean['DayOfWeek'] = df_clean['Datetime'].dt.dayofweek
        df_clean['Is_Weekend'] = df_clean['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
        
        # Ahora calculamos la longitud directamente sobre el mensaje real
        df_clean['Message_Length'] = df_clean['Mensaje'].apply(lambda x: len(str(x)))
        df_clean['Time_Delta_Sec'] = df_clean['Datetime'].diff().dt.total_seconds().fillna(0)
        
        return df_clean

    def fit_predict(self, df_features: pd.DataFrame, features_cols: list) -> pd.DataFrame:
        """
        [Método Público] 
        Escala las variables y ejecuta el modelo Isolation Forest.
        """
        print(f"Entrenando Isolation Forest sobre {len(df_features)} eventos...")
        X = df_features[features_cols]
        
        X_scaled = self.scaler.fit_transform(X)
        
        preds = self.model.fit_predict(X_scaled)
        df_features['Is_Anomaly'] = preds
        
        df_features['Anomaly_Score'] = self.model.decision_function(X_scaled)
        
        return df_features