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

    def explain_actor_profile(self, nombre: str, rol_matematico: str, mensajes_clave: list) -> str:
        """
        [Método Público] Genera un perfil forense de cualquier actor basado en su etiqueta K-Means.
        """
        mensajes_formateados = "\n".join([f"- {msg}" for msg in mensajes_clave])
        
        prompt = (
            f"Un modelo matemático de Machine Learning ha clasificado a '{nombre}' "
            f"con el rol estructural de '{rol_matematico}' dentro de una red de influencias.\n"
            f"Analiza esta muestra de sus comunicaciones más relevantes:\n{mensajes_formateados}\n\n"
            f"Redacta una breve conclusión forense (máximo 4 líneas) validando si el contenido "
            f"de sus mensajes concuerda con este rol, qué tipo de información maneja y su nivel de jerarquía."
        )
        return self._call_groq(prompt)

    def explain_anomaly(self, fecha: str, hora: str, longitud: int, score: float, mensaje: str) -> str:
        """
        [Método Público] Explica por qué un evento temporal es crítico para la investigación.
        """
        prompt = (
            f"Un modelo de Isolation Forest ha marcado este evento como una anomalía matemática extrema "
            f"(Score: {score:.3f}). Ocurrió el {fecha} a las {hora}, con una longitud inusual de {longitud} caracteres:\n\n"
            f"Mensaje:\n'{mensaje}'\n\n"
            f"Analiza el contenido y el contexto de los metadatos. "
            f"Redacta una evaluación forense (máximo 4 líneas) justificando por qué este evento refleja "
            f"una situación de crisis, coordinación clandestina o revelación de pruebas clave."
        )
        return self._call_groq(prompt)