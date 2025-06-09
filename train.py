import pandas as pd
import tensorflow as tf
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

# --- CONFIGURACIÓN ---
MODELO_NOMBRE = "dccuchile/bert-base-spanish-wwm-uncased"
RUTA_GUARDADO = './fine_tuned_diagnostic_model'
DATOS_CSV = 'training_data.csv'
EPOCHS = 3
BATCH_SIZE = 8
LEARNING_RATE = 5e-5

# --- 1. CARGAR Y PREPARAR DATOS ---
print("Cargando datos de entrenamiento...")
df = pd.read_csv(DATOS_CSV)
textos = df['texto_sintomas'].tolist()
etiquetas_texto = df['enfermedad'].tolist()

label_encoder = LabelEncoder()
etiquetas_num = label_encoder.fit_transform(etiquetas_texto)
num_clases = len(label_encoder.classes_)

print(f"Encontradas {len(textos)} muestras y {num_clases} enfermedades.")

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(textos, etiquetas_num, test_size=0.2, random_state=42, stratify=etiquetas_num)

# --- 2. CARGAR TOKENIZER Y TOKENIZAR DATOS ---
print(f"Cargando tokenizer de '{MODELO_NOMBRE}'...")
tokenizer = AutoTokenizer.from_pretrained(MODELO_NOMBRE)

print("Tokenizando datos...")
train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(X_test, truncation=True, padding=True, max_length=128)

# Convertir a formato de TensorFlow Dataset para eficiencia
train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings), y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((dict(test_encodings), y_test))

# --- 3. CARGAR MODELO BASE Y CONFIGURAR PARA FINE-TUNING ---
print(f"Cargando modelo base '{MODELO_NOMBRE}' para fine-tuning...")
model = TFAutoModelForSequenceClassification.from_pretrained(MODELO_NOMBRE, num_labels=num_clases, from_pt=True)

optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

# --- 4. REALIZAR EL FINE-TUNING ---
print("\n--- INICIANDO FINE-TUNING ---")
history = model.fit(train_dataset.shuffle(len(X_train)).batch(BATCH_SIZE),
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    validation_data=test_dataset.batch(BATCH_SIZE))
print("--- FINE-TUNING COMPLETADO ---\n")

# --- 5. GUARDAR EL MODELO AFINADO, TOKENIZER Y LABEL ENCODER ---
print(f"Guardando modelo afinado en '{RUTA_GUARDADO}'...")
model.save_pretrained(RUTA_GUARDADO)
tokenizer.save_pretrained(RUTA_GUARDADO)
joblib.dump(label_encoder, f'{RUTA_GUARDADO}/label_encoder.joblib')

print("\n¡Proceso de entrenamiento finalizado con éxito!")
print(f"El modelo, tokenizer y label encoder están guardados en la carpeta: {RUTA_GUARDADO}")