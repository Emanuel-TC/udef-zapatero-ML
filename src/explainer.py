import os
import pandas as pd
from groq import Groq
from dotenv import load_dotenv, find_dotenv

# Cargamos las variables de entorno desde la raíz
load_dotenv(find_dotenv())

class ForensicExplainer:
    """
    Clase para interactuar con LLMs vía Groq API.
    Añade una capa de explicabilidad semántica a los hallazgos matemáticos previos.
    """
    def __init__(self, model_name: str = "llama-3.3-70b-versatile"):
        self.client = Groq()
        self.model_name = model_name
        
        # System prompt que define la "personalidad" y el rigor del modelo
        self.system_prompt = (
            "Eres un Analista Experto en Inteligencia Financiera. "
            "Tu tarea es interpretar mensajes extraídos de un sumario policial. "
            "Debes ser directo, analítico, objetivo y usar un tono corporativo/forense. "
            "No alucines datos, básate estrictamente en el texto proporcionado. "
            "Responde siempre en español, en un único párrafo de máximo 4 líneas."
        )

    def _call_groq(self, prompt: str) -> str:
        """[Método Privado] Maneja la comunicación con la API de Groq."""
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                model=self.model_name,
                temperature=0.2, # Baja temperatura para que sea analítico, no creativo
                max_tokens=256
            )
            return chat_completion.choices[0].message.content.strip()
        except Exception as e:
            return f"Error en la generación LLM: {str(e)}"

    def explain_broker_profile(self, nombre_broker: str, mensajes_clave: list) -> str:
        """
        [Método Público] Genera un perfil del Broker basándose en sus mensajes.
        """
        mensajes_formateados = "\n".join([f"- {msg}" for msg in mensajes_clave])
        
        prompt = (
            f"Un modelo matemático de grafos ha identificado a '{nombre_broker}' como un "
            f"'Broker/Intermediario Clave' en una red de influencias por su alta centralidad de intermediación.\n"
            f"Analiza esta muestra de sus comunicaciones más relevantes:\n{mensajes_formateados}\n\n"
            f"Redacta una breve conclusión forense (máximo 4 líneas) explicando qué tipo de rol ejerce, "
            f"con quién conecta y qué tipo de información maneja."
        )
        return self._call_groq(prompt)

    def explain_anomaly(self, fecha: str, hora: str, mensaje: str) -> str:
        """
        [Método Público] Explica por qué un evento temporal es crítico para la investigación.
        """
        prompt = (
            f"Un modelo de Machine Learning (Isolation Forest) ha marcado el siguiente mensaje como una "
            f"anomalía crítica de comportamiento dentro de la red temporal de comunicaciones:\n\n"
            f"Fecha: {fecha}\nHora: {hora}\nMensaje: '{mensaje}'\n\n"
            f"Analiza el contenido y el contexto (por ejemplo, si ocurre de madrugada o si denota tensión). "
            f"Redacta una evaluación forense (máximo 4 líneas) justificando por qué este evento refleja "
            f"una situación de urgencia, presión o coordinación clandestina."
        )
        return self._call_groq(prompt)