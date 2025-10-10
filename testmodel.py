import numpy as np
import tensorflow as tf
import pickle
import nltk
from nltk.stem import WordNetLemmatizer

# Load words and classes (same as training)
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="intent_detection_model.tflite")
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

lemmatizer = WordNetLemmatizer()

def preprocess_sentence(sentence):
    """Convert input sentence into bag-of-words vector"""
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(w.lower()) for w in sentence_words]
    bag = [1 if w in sentence_words else 0 for w in words]
    return np.array([bag], dtype=np.float32)

while True:
    test_sentence = input("\nType something (or 'quit' to stop): ")
    if test_sentence.lower() == "quit":
        break

    # Preprocess input
    input_data = preprocess_sentence(test_sentence)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Get predicted class
    pred_index = np.argmax(output_data)
    confidence = output_data[0][pred_index]

    print(f"Prediction: {classes[pred_index]}  (confidence: {confidence:.2f})")
