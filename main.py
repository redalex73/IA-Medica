import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QTextEdit, QLineEdit, 
                             QPushButton, QVBoxLayout, QWidget, QHBoxLayout, QMessageBox)
from diagnostic_bot import DiagnosticBot

MODEL_PATH = './fine_tuned_diagnostic_model'
KB_PATH = 'knowledge_base.json'

class ChatWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Asistente de Diagnóstico IA (Modelo Afinado)")
        self.setGeometry(100, 100, 700, 500)

        # Cargar el bot
        self.bot = DiagnosticBot(model_path=MODEL_PATH, knowledge_base_path=KB_PATH)

        # --- Widgets ---
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setHtml("<b>Bot:</b> " + self.bot.process_input("").replace('\n', '<br>'))

        self.input_line = QLineEdit()
        self.input_line.setPlaceholderText("Escribe tus síntomas aquí...")
        self.input_line.returnPressed.connect(self.send_message)

        self.send_button = QPushButton("Enviar")
        self.send_button.clicked.connect(self.send_message)

        # --- Layout ---
        input_layout = QHBoxLayout()
        input_layout.addWidget(self.input_line)
        input_layout.addWidget(self.send_button)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.chat_display)
        main_layout.addLayout(input_layout)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def send_message(self):
        user_text = self.input_line.text().strip()
        if not user_text:
            return

        self.add_message("Tú", user_text)
        self.input_line.clear()

        bot_response = self.bot.process_input(user_text)
        self.add_message("Bot", bot_response)

    def add_message(self, sender, message):
        temp_message = message.replace('\n', '<br>')
        while "**" in temp_message:
            temp_message = temp_message.replace("**", "<b>", 1)
            temp_message = temp_message.replace("**", "</b>", 1)
        formatted_message = f"<b>{sender}:</b> {temp_message}"
        self.chat_display.append(formatted_message)
        self.chat_display.verticalScrollBar().setValue(self.chat_display.verticalScrollBar().maximum())

def check_model_exists():
    """Verifica si la carpeta del modelo entrenado y sus archivos existen."""
    if not os.path.isdir(MODEL_PATH):
        return False
    # Verificar archivos clave
    files_needed = ['tf_model.h5', 'config.json', 'tokenizer.json', 'label_encoder.joblib']
    for f in files_needed:
        if not os.path.exists(os.path.join(MODEL_PATH, f)):
            return False
    return True

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    if not check_model_exists():
        QMessageBox.critical(None, "Error: Modelo no encontrado",
                             f"No se encontró el modelo entrenado en la carpeta '{MODEL_PATH}'.\n\n"
                             "Por favor, ejecute el script 'train.py' primero para entrenar y guardar el modelo.\n\n"
                             "Comando a ejecutar en la terminal:\n"
                             "python train.py")
        sys.exit(1)

    window = ChatWindow()
    window.show()
    sys.exit(app.exec())