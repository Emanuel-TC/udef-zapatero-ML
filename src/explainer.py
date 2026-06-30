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
            f"con un nivel de jerarquía (PageRank) de {pagerank:.4f}.\n"
            f"Muestra de comunicaciones y su Emoción NLP (RoBERTa):\n"
            f"{mensajes_contexto}\n\n"
            f"Redacta una evaluación forense (máx 4 líneas). Valida su rol. "
            f"NOTA FORENSE: Si predominan las emociones 'OTHERS' o Neutras, interprétalo como 'Afecto Plano': "
            f"un comportamiento típico de operadores de alto nivel que usan un tono frío, corporativo o encriptado "
            f"para no dejar evidencia emocional, propio de gestiones de lobby o influencia."
        )
        return self._call_groq(prompt)

    def explain_anomaly(self, fecha: str, hora: str, longitud: int, score: float, emocion_nlp: str, mensaje: str) -> str:
        prompt = (
            f"Un algoritmo Isolation Forest marcó este evento como anomalía matemática crítica (Score: {score:.3f}).\n"
            f"Fecha {fecha}, Hora {hora}, Longitud {longitud} caracteres.\n"
            f"Modelo NLP detectó Emoción: {emocion_nlp.upper()}\n"
            f"Mensaje: '{mensaje}'\n\n"
            f"Redacta una evaluación forense (máx 4 líneas). "
            f"REGLA CRÍTICA: En inteligencia financiera, una emoción 'OTHERS' (Neutra) cruzada con una anomalía matemática "
            f"(como escribir 1000 caracteres a horas atípicas o usar intermediarios) es el mayor indicador de "
            f"COORDINACIÓN CLANDESTINA CALCULADA o LENGUAJE CODIFICADO. No lo descartes como ruido. "
            f"Explica qué gestión crítica o presión refleja el texto basándote en esto."
        )
        return self._call_groq(prompt)