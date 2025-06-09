import json
import tensorflow as tf
import numpy as np
import joblib
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

class DiagnosticBot:
    def __init__(self, model_path, knowledge_base_path):
        print("Cargando modelo afinado, tokenizer y label encoder...")
        self.model = TFAutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.label_encoder = joblib.load(f'{model_path}/label_encoder.joblib')
        
        print("Cargando base de conocimiento para diálogo...")
        with open(knowledge_base_path, 'r', encoding='utf-8') as f:
            self.kb = json.load(f)
        
        print("Bot listo.")
        self.reset()

    def reset(self):
        self.state = "INIT"
        self.potential_disease = None
        self.question_index = 0
        self.confirmed_symptoms = []

    def process_input(self, user_input):
        user_input_lower = user_input.lower()

        for alerta in self.kb['alertas']:
            if alerta in user_input_lower:
                self.reset()
                return f"¡ALERTA! Ha mencionado '{alerta}', que puede ser un síntoma grave. Por favor, busque atención médica de emergencia inmediatamente. Esta conversación se reiniciará."

        if self.state == "INIT":
            self.state = "GATHERING_INFO"
            return "Hola, soy tu asistente de diagnóstico inicial. Por favor, describe tus síntomas principales. Recuerda, esto no reemplaza a un médico."

        elif self.state == "GATHERING_INFO":
            if len(user_input) < 10:
                return "Por favor, dame un poco más de detalle sobre cómo te sientes."
            
            # --- CAMBIO PRINCIPAL AQUÍ ---
            # Realizar la predicción
            inputs = self.tokenizer(user_input, return_tensors="tf", truncation=True, padding=True)
            outputs = self.model(inputs)
            logits = outputs.logits[0]
            probabilities = tf.nn.softmax(logits, axis=-1).numpy()
            
            # Obtener los índices de las 3 mejores predicciones
            top_3_indices = np.argsort(probabilities)[-3:][::-1] # Índices de las 3 más probables

            # Generar el texto del diagnóstico diferencial
            differential_diagnosis_text = "Basado en tu descripción, estas son las posibilidades más relevantes:\n\n"
            
            for i, index in enumerate(top_3_indices):
                confidence = probabilities[index]
                disease_name = self.label_encoder.inverse_transform([index])[0]
                
                # No mostrar opciones con confianza muy baja
                if confidence < 0.05:
                    continue

                if i == 0: # La más probable
                    differential_diagnosis_text += f"1. **{disease_name}** (Probabilidad más alta: {confidence:.1%})\n"
                    # Guardamos la enfermedad principal para continuar el diálogo
                    self.potential_disease = next((d for d in self.kb['enfermedades'] if d['nombre'] == disease_name), None)
                else: # Las alternativas
                    differential_diagnosis_text += f"{i+1}. {disease_name} (Posibilidad: {confidence:.1%})\n"
            
            differential_diagnosis_text += "\nPara afinar el resultado, te haré un par de preguntas sobre la opción más probable."
            
            print(f"Diagnóstico Diferencial Interno: {[self.label_encoder.inverse_transform([i])[0] for i in top_3_indices]}")
            
            # Si por alguna razón no encontramos la enfermedad principal en la KB
            if self.potential_disease is None:
                self.reset()
                return "El modelo ha detectado una posibilidad, pero no tengo información de diálogo para continuar. Se recomienda consultar a un médico."

            self.state = "ASKING_QUESTIONS"
            # Devolvemos el texto del diagnóstico diferencial y luego la primera pregunta
            return differential_diagnosis_text + "\n\n" + self._ask_next_question()


        elif self.state == "ASKING_QUESTIONS":
            if any(word in user_input_lower for word in ['sí', 'si', 'afirmativo', 'correcto', 'tengo']):
                current_question = self.potential_disease['preguntas'][self.question_index-1]
                self.confirmed_symptoms.append(current_question['sintoma_clave'])
            
            return self._ask_next_question()

    def _ask_next_question(self):
        if self.question_index < len(self.potential_disease['preguntas']):
            question_data = self.potential_disease['preguntas'][self.question_index]
            self.question_index += 1
            return question_data['pregunta']
        else:
            return self._generate_conclusion()

    def _generate_conclusion(self):
        conclusion = "--- Conclusión del Asistente ---\n"
        conclusion += f"Tras confirmar algunos síntomas, la orientación principal sigue apuntando hacia: **{self.potential_disease['nombre']}**.\n\n"
        if self.confirmed_symptoms:
            conclusion += "Síntomas clave confirmados:\n"
            for sym in self.confirmed_symptoms:
                conclusion += f"- {sym}\n"
        else:
            conclusion += "No se confirmaron síntomas específicos en las preguntas, pero la descripción inicial sigue siendo consistente con esta condición.\n"
        conclusion += f"\n**Sugerencia:** {self.potential_disease['sugerencia_final']}\n\n"
        conclusion += "**IMPORTANTE:** Este es un análisis automático y no un diagnóstico médico. Siempre consulte a un profesional de la salud para un diagnóstico definitivo."
        self.reset()
        return conclusion