import requests
import re
from bs4 import BeautifulSoup
from transformers import pipeline
from googleapiclient.discovery import build

# Load summarizer ONCE
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

API_KEY = "AIzaSyDw1TohxoR-vmQQnPwP1_LOk_IAoJkVb2U"
CSE_ID = "f2717198f394b45be"


def fetch_page_text(url):
    """Fetch visible text from a webpage."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, timeout=10, headers=headers)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Extract visible text from <p> tags
        paragraphs = [p.get_text() for p in soup.find_all("p")]
        full_text = " ".join(paragraphs)

        # Clean up
        full_text = re.sub(r"\s+", " ", full_text).strip()

        return full_text
    except Exception as e:
        print(f"[ERROR] Could not fetch {url}: {e}")
        return ""


def ai_search(search_text):
    print(f"[DEBUG] Received search_text: {search_text}")

    service = build("customsearch", "v1", developerKey=API_KEY)
    result = service.cse().list(q=search_text, cx=CSE_ID, num=5).execute()

    items = result.get("items", [])
    if not items:
        return "I couldn’t find any information on that."

    # Collect snippets
    snippets = [item.get("snippet", "") for item in items if item.get("snippet")]
    combined_text = " ".join(snippets)

    if not combined_text.strip():
        return "No information available."

    # Summarize snippets into a short, meaningful answer
    try:
        summary = summarizer(
            combined_text,
            max_length=80,   # shorter answer
            min_length=40,   # but still informative
            do_sample=False
        )
        return summary[0]['summary_text']
    except Exception as e:
        print(f"[ERROR] Summarization failed: {e}")
        # fallback: return first snippet
        return snippets[0]
