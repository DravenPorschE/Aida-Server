import whisper
import os
import string
import librosa
import soundfile as sf  # pip install soundfile

import sentence_intent_detection as detect

transcriptor_model = whisper.load_model("small")

def transcibe_wav_file(file_path: str):
    # ✅ Check duration
    try:
        duration = librosa.get_duration(path=file_path)
        if duration < 0.5:  # skip too short clips
            print(f"⚠️ Skipping too short audio ({duration:.2f}s)")
            return ""
    except Exception as e:
        print("⚠️ Could not measure duration:", e)
        return ""

    # ✅ Check that audio has samples
    try:
        data, sr = sf.read(file_path)
        if data is None or len(data) == 0:
            print("⚠️ Empty audio file, skipping")
            return ""
    except Exception as e:
        print("⚠️ Could not read audio file:", e)
        return ""

    # ✅ Safe to run Whisper
    result = transcriptor_model.transcribe(file_path, language="en")
    text = result["text"].lower()

    wakewords = ["hey aida", "eye the", "hey ada", "he aida"]
    command = ""

    for ww in wakewords:
        idx = text.find(ww)
        if idx != -1:
            command = text[idx + len(ww):].strip()
            command = command.lstrip(string.punctuation + " ")
            break

    print("Transcribed text:", text)
    print("Detected command:", command)

    if command:
        intent = detect.get_actions(command)
        print(intent)
        return intent

    return text
