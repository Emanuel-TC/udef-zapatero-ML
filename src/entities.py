import pandas as pd
import re
from typing import Dict, List, Optional

class EntityResolver:
    """
    Clase para la Desambiguación y Resolución de Entidades (Entity Resolution).
    Unifica diferentes alias, apodos y errores tipográficos en un único nodo maestro.
    """
    
    def __init__(self, alias_mapping: Optional[Dict[str, str]] = None):
        # Diccionario de desambiguación (Knowledge Base forense)
        # Si no se le pasa uno al instanciar, usa este por defecto.
        self.alias_mapping = alias_mapping or {
            "ZP": "ZAPATERO",
            "EL ZAPA": "ZAPATERO",
            "ZAPA": "ZAPATERO",
            "AVALOS": "ABALOS", # Corrige la ortografía de los emisores
            "EL TOCAYO": "JULIO MARTINEZ MARTINEZ",
            "JULITO": "JULIO MARTINEZ MARTINEZ",
            "EL MORITO": "NOURREDINE OUABDESSELAM",
            "RODOMAN": "RODOLFO REYES",
            "SANTI": "SANTIAGO FERNANDEZ LENA",
            "DELCY": "DELCY RODRIGUEZ",
            "PEDRO SAURA": "PEDRO SAURA", 
            "SAURA": "PEDRO SAURA"
        }

    def _limpiar_texto_general(self, texto: str) -> str:
        """
        Normaliza un string eliminando ruido, tildes y espacios redundantes.
        """
        if not isinstance(texto, str):
            return "DESCONOCIDO"
        
        texto = texto.upper().strip()
        
        # Eliminación de tildes para evitar duplicados (ej. Ábalos vs Abalos)
        tildes = {'Á': 'A', 'É': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U'}
        for acento, sin_acento in tildes.items():
            texto = texto.replace(acento, sin_acento)
            
        # Reemplazar múltiples espacios internos por uno solo
        texto = re.sub(r'\s+', ' ', texto)
        return texto

    def resolver_entidad_especifica(self, entidad: str) -> str:
        """
        Aplica el motor de reglas sobre una entidad para devolver su Nombre Maestro.
        """
        entidad_limpia = self._limpiar_texto_general(entidad)
        
        # 1. Búsqueda exacta en el diccionario de alias
        if entidad_limpia in self.alias_mapping:
            return self.alias_mapping[entidad_limpia]
            
        # 2. Heurística parcial (Si contiene la palabra clave, lo asimilamos al nodo principal)
        if "ZAPATERO" in entidad_limpia:
            return "ZAPATERO"
        if "ABALOS" in entidad_limpia:
            return "ABALOS"
            
        return entidad_limpia

    def consolidar_grafo(self, df_grafo: pd.DataFrame, columnas_entidades: List[str]) -> pd.DataFrame:
        """
        Recibe un DataFrame de aristas, resuelve las entidades y recalcula los pesos.
        """
        df_resuelto = df_grafo.copy()
        
        # Aplicamos la resolución a las columnas de Origen y Destino
        for col in columnas_entidades:
            if col in df_resuelto.columns:
                df_resuelto[col] = df_resuelto[col].apply(self.resolver_entidad_especifica)
        
        # Al unificar nombres, se generan aristas duplicadas.
        # (Ej: Rodolfo->ZP (peso 2) y Rodolfo->ZAPATERO (peso 3) ahora son Rodolfo->ZAPATERO (peso 5))
        if 'Peso' in df_resuelto.columns and 'Tipo_Relacion' in df_resuelto.columns:
            df_resuelto = df_resuelto.groupby(
                ['Origen', 'Destino', 'Tipo_Relacion']
            )['Peso'].sum().reset_index()
            
        # Filtramos auto-bucles residuales (ej. Si alguien llamaba "Rodoman" a Rodolfo en el mismo chat)
        df_resuelto = df_resuelto[df_resuelto['Origen'] != df_resuelto['Destino']]
            
        return df_resuelto.sort_values(by='Peso', ascending=False).reset_index(drop=True)