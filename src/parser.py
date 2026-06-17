import re
import json
import fitz  # PyMuPDF
import pandas as pd
from typing import List, Dict, Union
import os
import time
from dotenv import load_dotenv, find_dotenv
from groq import Groq

# find_dotenv() busca el .env subiendo por los directorios hasta la raíz
load_dotenv(find_dotenv())

# Compilamos las regex a nivel de módulo para mayor eficiencia
REGEX_F1 = re.compile(
    r'\[(\d{1,2}/\d{1,2}/\d{2,4})[,\s]*(\d{1,2}:\d{2}:\d{2})\]\s*([^:]+?):\s*(.*?)(?=\n\s*\[\d{1,2}/\d{1,2}/\d{2,4}|\Z)', 
    re.DOTALL
)

REGEX_F2 = re.compile(
    r'\s*(?:([A-Za-zÁÉÍÓÚÑáéíóúñ0-9\+\-\.\(\) ]+)\s*\n)?' 
    r'(?:Timestamp|Fecha|Date)\s*:\s*(\d{1,2}/\d{1,2}/\d{2,4})\s+(\d{1,2}:\d{2}:\d{2})(?:\s*\(UTC[^)]*\))?\s*\n'
    r'(.*?)'
    r'(?=\n\s*(?:[A-Za-zÁÉÍÓÚÑáéíóúñ0-9\+\-\.\(\) ]+\s*\n)?(?:Timestamp|Fecha|Date)\s*:|\n\s*\[\d{1,2}/|\Z)',
    re.DOTALL
)

def extraer_con_regex(texto: str, num_pag: int) -> List[Dict[str, Union[str, int]]]:
    """Extrae chats usando expresiones regulares (Reglas deterministas)."""
    matches = []
    
    # Extraer F1
    for match in REGEX_F1.finditer(texto):
        fecha, hora, emisor, mensaje = match.groups()
        matches.append({
            "Pagina_PDF": num_pag,
            "Formato": "RegEx_F1",
            "Fecha": fecha.strip(),
            "Hora": hora.strip(),
            "Emisor": emisor.strip(),
            "Mensaje": " ".join(mensaje.split())
        })
        
    # Extraer F2
    ultimo_emisor = "DESCONOCIDO"
    for match in REGEX_F2.finditer(texto):
        emisor, fecha, hora, mensaje = match.groups()
        if emisor and emisor.strip():
            ultimo_emisor = emisor.strip()
            
        matches.append({
            "Pagina_PDF": num_pag,
            "Formato": "RegEx_F2",
            "Fecha": fecha.strip(),
            "Hora": hora.strip(),
            "Emisor": ultimo_emisor,
            "Mensaje": " ".join(mensaje.split())
        })
        
    return matches

def extraer_con_llm(texto: str, num_pag: int, groq_client: Groq) -> List[Dict[str, Union[str, int]]]:
    """Usa un LLM vía Groq para extraer chats de texto forense corrupto."""
    prompt_sistema = """
    Eres un perito informático forense. Tu objetivo es extraer transcripciones de chats (WhatsApp/SMS) de un documento policial escaneado con formato corrupto.
    Devuelve ÚNICAMENTE un array JSON válido con la siguiente estructura exacta:
    [
      {"Fecha": "DD/MM/AAAA", "Hora": "HH:MM:SS", "Emisor": "Nombre del Emisor", "Mensaje": "Texto del mensaje"}
    ]
    Si no hay conversaciones de chat en el texto, devuelve un array vacío: [].
    No incluyas explicaciones, markdown extra, ni texto fuera del JSON.
    """
    
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile", 
            messages=[
                {"role": "system", "content": prompt_sistema},
                {"role": "user", "content": f"Extrae los chats de este texto:\n\n{texto}"}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        
        raw_json = response.choices[0].message.content
        datos = json.loads(raw_json)
        
        lista_chats = []
        if isinstance(datos, list):
            lista_chats = datos
        elif isinstance(datos, dict) and len(datos) > 0:
            clave = list(datos.keys())[0]
            lista_chats = datos[clave] if isinstance(datos[clave], list) else []

        resultados = []
        for chat in lista_chats:
            resultados.append({
                "Pagina_PDF": num_pag,
                "Formato": "LLM_Groq",
                "Fecha": chat.get("Fecha", ""),
                "Hora": chat.get("Hora", ""),
                "Emisor": chat.get("Emisor", "DESCONOCIDO"),
                "Mensaje": chat.get("Mensaje", "")
            })
            
        time.sleep(2) # Pausa técnica para evitar límites 429 de la API
        return resultados

    except Exception as e:
        print(f"Error en extracción LLM para página {num_pag}: {str(e)}")
        return []

def procesar_documento_hibrido(pdf_path: str, pag_inicio: int, pag_fin: int) -> pd.DataFrame:
    """Orquesta la extracción híbrida: Regex primero, LLM como fallback."""
    doc = fitz.open(pdf_path)
    
    # Al inicializar Groq() sin parámetros, busca automáticamente GROQ_API_KEY en el entorno
    groq_client = Groq()
    todos_los_chats = []
    
    print("Iniciando extracción Híbrida (RegEx + Groq LLM)...")
    
    for num_pag in range(pag_inicio, pag_fin + 1):
        texto_pagina = doc[num_pag].get_text()
        
        # 1er intento: Extracción rápida 
        chats_pagina = extraer_con_regex(texto_pagina, num_pag)
        
        # Criterio de fallo: Si hay mucho texto pero la regex extrajo muy poco
        if len(chats_pagina) <= 2 and len(texto_pagina.strip()) > 200:
            print(f"  -> Anomalía detectada en página {num_pag}. Activando agente LLM...")
            chats_llm = extraer_con_llm(texto_pagina, num_pag, groq_client)
            
            # Si el LLM extrajo más información, nos quedamos con su versión
            if len(chats_llm) > len(chats_pagina):
                chats_pagina = chats_llm
                
        todos_los_chats.extend(chats_pagina)
        
    df = pd.DataFrame(todos_los_chats)
    
    if not df.empty:
        df = df.dropna(subset=['Mensaje'])
        df = df[df['Mensaje'].str.strip() != ""]
        df = df.sort_values(by=["Pagina_PDF", "Fecha", "Hora"]).reset_index(drop=True)
        
    return df