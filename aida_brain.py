# ----------------------------
# AIDA Meeting Server (HTML Summary)
# ----------------------------
import os
import re
import uuid
import queue
import threading
import requests
from flask import Flask, request, jsonify
import whisper
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
import time
import whisper.audio

from datetime import datetime
from datetime import date

from flask import send_file

import json
from json.decoder import JSONDecodeError


# ----------------------------
# Load Whisper Model
# ----------------------------
print("Loading Whisper model...")
meeting_transcriptor_model = whisper.load_model("small")
model_lock = threading.Lock()
print("Model loaded successfully.")

LANGUAGE = "english"

# ----------------------------
# Summarization Functions
# ----------------------------
def summarize_meeting_html(transcript_text, max_sentences=7):
    text = re.sub(r'\s+', ' ', transcript_text).strip()
    parser = PlaintextParser.from_string(text, Tokenizer(LANGUAGE))
    summarizer = LexRankSummarizer(Stemmer(LANGUAGE))
    summarizer.stop_words = get_stop_words(LANGUAGE)
    summary_sentences = summarizer(parser.document, max_sentences)

    action_keywords = ["will", "shall", "must", "action", "task", "decide", "assign"]
    highlights, keypoints = [], []

    for sentence in summary_sentences:
        s = str(sentence).strip()
        (highlights if any(k in s.lower() for k in action_keywords) else keypoints).append(s)

    keypoints_html = "<b>Key Points:</b><br>" + "<br>".join(keypoints) if keypoints else "<b>Key Points:</b><br>No key points found."
    highlights_html = "<br><br><b>Highlights:</b><br>" + "<br>".join(f"• {h}" for h in highlights) if highlights else "<br><br><b>Highlights:</b><br>No major highlights."
    return keypoints_html + highlights_html

def getSmartMeetingReader(transcript_text):
    import requests, json, re, os
    from json.decoder import JSONDecodeError
    from datetime import date, datetime

    today = date.today()
    now = datetime.now()
    weekday_number = today.weekday()
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_name = days[weekday_number]

    api_key = "sk-Y7WQU70xrXq6CmGFksnPPg"
    if not api_key:
        print("⚠️ No BLACKBOX_API_KEY found.")
        return None

    url = "https://api.blackbox.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    prompt = (
        f"You are a smart meeting transcript reader. Analyze the meeting transcript below and detect actionable intents.\n\n"
        f"Current date: {today.year}-{today.month}-{today.day} ({day_name})\n"
        f"Current time: {now.strftime('%I:%M %p')}\n\n"
        "You must detect and output **only** these three types of actions:\n"
        "1. create_calendar_event — for statements that mention a date or day (e.g., 'On Monday we will meet with investors'). "
        "Infer the actual date using today's context.\n"
        "2. create_note — for statements that mention taking note of something important (e.g., 'Let's take note that our system needs optimization.'). "
        "Extract the meaningful part to store as a note. Each note must include both a **title** and a **content** field. "
        "Generate the title based on the content (short and descriptive, like a summary of the note).\n"
        "3. set_alarm — for mentions of time or reminders (e.g., 'Set a reminder in one hour' or 'Remind me at 5 PM'). "
        "Calculate the correct time when possible.\n\n"
        "Output ONLY valid JSON with the structure below. Do NOT include explanations, markdown, or text outside JSON.\n"
        "If no action is detected, output an empty JSON object {}.\n\n"
        "Example JSON format:\n"
        "{\n"
        "  \"alarm_action\": [\n"
        "    {\n"
        "      \"action\": \"set_alarm\",\n"
        "      \"data\": {\"hour\": 12, \"minute\": 30, \"ampm\": \"PM\"}\n"
        "    }\n"
        "  ],\n"
        "  \"calendar_action\": [\n"
        "    {\n"
        "      \"action\": \"create_calendar_event\",\n"
        "      \"data\": {\"year\": 2025, \"month\": \"October\", \"day\": 13, \"title\": \"Meeting with investors\"}\n"
        "    }\n"
        "  ],\n"
        "  \"note_action\": [\n"
        "    {\n"
        "      \"action\": \"create_note\",\n"
        "      \"data\": {\n"
        "        \"title\": \"System Optimization\",\n"
        "        \"content\": \"We need to optimize our response time.\"\n"
        "      }\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        f"Transcript:\n{transcript_text}"
    )


    payload = {
        "model": "blackboxai/amazon/nova-micro-v1",
        "messages": [
            {"role": "system", "content": "You are an assistant that extracts structured actions from meeting transcripts."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 2048,
        "temperature": 0.4
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=90)
        response.raise_for_status()

        # Safe extraction of content (handles variations in API response shape)
        data = response.json()
        content = None
        try:
            if isinstance(data, dict) and data.get("choices"):
                choice = data["choices"][0]
                # OpenAI-style: choice may contain 'message' -> 'content'
                if isinstance(choice, dict):
                    if "message" in choice and isinstance(choice["message"], dict) and "content" in choice["message"]:
                        content = choice["message"]["content"]
                    elif "text" in choice:
                        content = choice["text"]
                    else:
                        # fallback to stringifying the choice
                        content = json.dumps(choice)
                else:
                    content = str(choice)
            else:
                # if API returned an error object or different shape, stringify for debugging
                content = json.dumps(data)
        except Exception as e:
            print("⚠️ Unexpected shape in API response:", e)
            content = json.dumps(data)

        if content is None:
            content = ""

        content = str(content).strip()

        # 🧹 Clean common wrappers/code fences
        content = re.sub(r"^```json\s*", "", content, flags=re.IGNORECASE)
        content = re.sub(r"```$", "", content)
        content = content.strip()

        # First attempt: direct parse
        try:
            json_output = json.loads(content)
        except JSONDecodeError as e:
            print(f"⚠️ Initial JSON parse failed: {e}")
            # Attempt progressive fixes

            # 1) Extract substring between first '{' and last '}', to remove surrounding text
            start = content.find("{")
            end = content.rfind("}")
            candidate = content
            if start != -1 and end != -1 and end > start:
                candidate = content[start:end+1]

            # 2) Remove newlines/carriage returns that sometimes break parsing
            candidate = candidate.replace("\r", " ").replace("\n", " ")

            # 3) Remove trailing commas before } or ]
            candidate = re.sub(r",\s*([}\]])", r"\1", candidate)

            # 4) If single quotes used for JSON keys/strings, try swapping to double quotes
            #    (only do this when double quotes are not present much; cautious)
            if candidate.count('"') < candidate.count("'"):
                candidate = candidate.replace("'", '"')

            # 5) Try parsing candidate
            try:
                json_output = json.loads(candidate)
            except Exception as e2:
                print(f"⚠️ Second JSON parse failed: {e2}")
                # As a last-resort fallback, provide empty but correctly structured output
                json_output = {
                    "alarm_action": [],
                    "calendar_action": [],
                    "note_action": []
                }
                # Log the problematic outputs for debugging
                print("---- RAW MODEL OUTPUT ----")
                print(content)
                print("---- CLEANED CANDIDATE ----")
                print(candidate)
                print("--------------------------")

        # Ensure structure keys exist and are lists
        for k in ("alarm_action", "calendar_action", "note_action"):
            if k not in json_output or not isinstance(json_output[k], list):
                json_output[k] = []

        # 📝 Save to file (same filename as before to avoid changing behavior)
        file_path = "meeting_actions.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(json_output, f, indent=2, ensure_ascii=False)

        print(f"✅ JSON actions saved to {os.path.abspath(file_path)}")
        return file_path  # Return the path to the JSON file

    except Exception as e:
        print(f"⚠️ Smart Meeting Reader failed: {e}")
        return None

def summarize_meeting_blackbox(transcript_text):
    api_key = "sk-Y7WQU70xrXq6CmGFksnPPg"
    if not api_key:
        print("⚠️ No BLACKBOX_API_KEY found, using LexRank fallback.")
        return summarize_meeting_html(transcript_text)

    url = "https://api.blackbox.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # ✅ Improved prompt
    prompt = (
        "You are a meeting summarizer. Read the full transcript and generate a clear HTML-formatted summary.\n\n"
        "Your output MUST follow this format:\n"
        "<b>Key Points:</b><br>\n"
        "• Summarize all the main discussion topics and insights (at least 5 bullet points if possible).<br><br>\n"
        "<b>Highlights:</b><br>\n"
        "• List decisions made, tasks assigned, or important follow-ups (include names or deadlines if mentioned).<br><br>\n"
        "<b>Brief Summary:</b><br>\n"
        "Write 3-5 concise sentences summarizing the overall context and outcome of the meeting.<br><br>\n"
        "Avoid using triple quotes or extra text outside HTML tags.\n\n"
        f"Meeting Transcript:\n{transcript_text}"
    )

    payload = {
        "model": "blackboxai/amazon/nova-micro-v1",
        "messages": [
            {"role": "system", "content": "You are an assistant that produces structured, readable meeting summaries in HTML."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 2048,
        "temperature": 0.4
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=90)
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"].strip()

        # 🧹 Remove unwanted text or formatting
        content = re.sub(r"^'+|'+$", "", content).strip()  # remove leading/trailing '''
        content = re.sub(r"```html|```", "", content).strip()  # remove code block markers
        content = content.replace("'''html meeting transcript summary '''", "").replace("'''", "").strip()

        # ✅ Ensure minimal HTML structure
        if not any(tag in content for tag in ["<b>", "<br>", "<ul>", "<li>"]):
            content = content.replace("\n", "<br>")
            content = f"<b>Key Points:</b><br>{content}"

        return content

    except Exception as e:
        print(f"⚠️ Blackbox summarization failed: {e}. Using LexRank fallback.")
        return summarize_meeting_html(transcript_text)

# ----------------------------
# Flask Setup
# ----------------------------
app = Flask(__name__)
task_queue = queue.Queue()
result_queue = queue.Queue()

def worker():
    while True:
        job_id, file_path = task_queue.get()
        if job_id is None:
            break

        # Measure transcription start time
        start_time = time.time()

        # Load audio to get duration
        audio = whisper.audio.load_audio(file_path)
        duration_seconds = len(audio) / whisper.audio.SAMPLE_RATE
        print(f"🎵 Received audio '{file_path}' with duration: {duration_seconds:.2f} seconds")

        with model_lock:
            result = meeting_transcriptor_model.transcribe(file_path, language=None)

        transcript = result["text"].strip()

        # Measure transcription end time
        end_time = time.time()
        processing_time = end_time - start_time
        print(f"⏱ Transcription completed in {processing_time:.2f} seconds")

        result_queue.put((job_id, transcript))
        task_queue.task_done()

for _ in range(2):
    threading.Thread(target=worker, daemon=True).start()

# ----------------------------
# Endpoints
# ----------------------------
@app.route("/retrieve-meeting-recording", methods=["POST"])
def retrieve_meeting_recording():
    if 'file' not in request.files:
        return jsonify({"response": "No file part", "key": "error"}), 400

    wav_file = request.files['file']
    if wav_file.filename == '':
        return jsonify({"response": "No selected file", "key": "error"}), 400

    temp_path = f"recording_{uuid.uuid4().hex}.wav"
    wav_file.save(temp_path)
    job_id = threading.get_ident()
    task_queue.put((job_id, temp_path))

    while True:
        rid, transcript = result_queue.get()
        if rid == job_id:
            try:
                os.remove(temp_path)
            except Exception as e:
                print(f"Cleanup error: {e}")
            return jsonify({"response": transcript, "key": "transcribed-meeting"})

@app.route("/retrieve-summarized-meeting", methods=["POST"])
def retrieve_summarized_meeting():
    try:
        transcript_text = request.files["file"].read().decode("utf-8")

        # Generate HTML summary (as before)
        summary_html = summarize_meeting_blackbox(transcript_text)

        # Generate actions JSON file (getSmartMeetingReader already writes meeting_actions.json)
        file_path = getSmartMeetingReader(transcript_text)
        if not file_path or not os.path.exists(file_path):
            # if SmartMeetingReader failed, still return summary so the client can display it
            return jsonify({"key": "success", "response": summary_html, "summary_html": summary_html, "actions": {
                "alarm_action": [], "calendar_action": [], "note_action": []
            }}), 200

        # Read the actions JSON so we can include it in the response body
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                json_actions = json.load(f)
            except Exception:
                # fallback empty structure
                json_actions = {"alarm_action": [], "calendar_action": [], "note_action": []}

        # Return both summary (HTML) and actions (object) in a JSON response
        return jsonify({
            "key": "success",
            "response": summary_html,      # backward-compatible field (your old client used this)
            "summary_html": summary_html,  # explicit field name
            "actions": json_actions
        }), 200

    except Exception as e:
        print(f"⚠️ retrieve_summarized_meeting failed: {e}")
        return jsonify({
            "key": "error",
            "response": f"Error processing file: {e}"
        }), 500


@app.route("/test-connection", methods=["POST"])
def test_connection():
    return jsonify({"response": "Code 200"})

# ----------------------------
# Run Server
# ----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
