import pickle
import json
import tensorflow as tf

# ===============================
# 1. Convert pickle files to JSON
# ===============================
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))

with open("words.json", "w", encoding="utf-8") as f:
    json.dump(words, f, ensure_ascii=False, indent=2)

with open("classes.json", "w", encoding="utf-8") as f:
    json.dump(classes, f, ensure_ascii=False, indent=2)

print("✅ Converted words.pkl -> words.json and classes.pkl -> classes.json")

# ======================================
# 2. Convert Keras .h5 model to .tflite
# ======================================
try:
    # Load the trained Keras model
    model = tf.keras.models.load_model("intent_detection_model.h5")

    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save as .tflite file
    with open("intent_detection_model.tflite", "wb") as f:
        f.write(tflite_model)

    print("✅ Converted intent_detection_model.h5 -> intent_detection_model.tflite")
except Exception as e:
    print("⚠️ Error converting model:", e)
