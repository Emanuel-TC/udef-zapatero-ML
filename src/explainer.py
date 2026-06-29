import os
import pandas as pd
from groq import Groq
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

class ForensicExplainer:
    """
    Clase para interactuar con LLMs vía Groq API.
    Añade una capa de explicabilidad semántica a los hallazgos matemáticos previos.
    """
    def __init__(self, model_name: str = "llama-3.3-70b-versatile"):
        self.client = Groq()
        self.model_name = model_name
        
        self.system_prompt = (
            "Eres un Analista Experto en Inteligencia Financiera. "
            "Tu tarea es interpretar mensajes extraídos de un sumario policial. "
            "Debes ser directo, analítico, objetivo y usar un tono corporativo/forense. "
            "No alucines datos, básate estrictamente en el texto proporcionado. "
            "Responde siempre en español, en un único párrafo de máximo 4 líneas."
        )

    def _call_groq(self, prompt: str) -> str:
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                model=self.model_name,
                temperature=0.2, 
                max_tokens=256
            )
            return chat_completion.choices[0].message.content.strip()
        except Exception as e:
            return f"Error en la generación LLM: {str(e)}"

    def explain_actor_profile(self, nombre: str, rol_matematico: str, pagerank: float, mensajes_contexto: list) -> str:
        prompt = (
            f"Un modelo matemático (K-Means) clasificó a '{nombre}' como '{rol_matematico}' "
            f"con un nivel de poder (PageRank) de {pagerank:.4f}.\n"
            f"Analiza esta muestra de sus comunicaciones (incluyendo la Emoción NLP extraída mediante RoBERTa):\n"
            f"{mensajes_contexto}\n\n"
            f"Redacta una evaluación forense (máximo 4 líneas). Explica SI su forma de hablar y sus emociones "
            f"justifican este rol jerárquico. Si es un 'Ejecutor', ¿recibe órdenes? Si es 'Cúpula', ¿da directrices?"
        )
        return self._call_groq(prompt)

    def explain_anomaly(self, fecha: str, hora: str, longitud: int, score: float, emocion_nlp: str, mensaje: str) -> str:
        prompt = (
            f"Un modelo Isolation Forest marcó este evento como anomalía matemática (Score: {score:.3f}).\n"
            f"Metadatos: Fecha {fecha}, Hora {hora}, Longitud {longitud} caracteres.\n"
            f"Modelo NLP (RoBERTa) detectó Emoción: {emocion_nlp.upper()}\n"
            f"Mensaje: '{mensaje}'\n\n"
            f"Redacta una evaluación forense (máximo 4 líneas). Usa el contexto de la hora y la EMOCIÓN NLP para no alucinar. "
            f"Si el mensaje es corto y neutral a las 3 AM, asume que es ruido por hora intempestiva y NO una crisis. "
            f"Si hay emoción de Miedo/Ira, destaca la urgencia."
        )
        return self._call_groq(prompt)