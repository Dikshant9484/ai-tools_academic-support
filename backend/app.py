from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
from dotenv import load_dotenv
import os, json, re

load_dotenv()

app = Flask(__name__)
CORS(app)

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found. Please set it in the .env file.")

client = Groq(api_key=api_key)
MODEL  = "llama-3.3-70b-versatile"   # latest, fast, free on Groq

def ask_groq(system_prompt, user_message, max_tokens=1500):
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message}
        ]
    )
    return response.choices[0].message.content

def parse_json_response(raw):
    clean = re.sub(r"```(?:json)?|```", "", raw).strip()
    return json.loads(clean)

# ── 1. Answer Subject Doubts ─────────────────────────────────────
@app.route("/api/answer", methods=["POST"])
def answer_doubt():
    data     = request.json
    question = data.get("question", "").strip()
    subject  = data.get("subject", "General").strip()
    if not question:
        return jsonify({"error": "Question is required"}), 400
    system = (
        f"You are an expert tutor in {subject}. "
        "Give clear, concise, well-structured answers suitable for students. "
        "Use examples, analogies, and step-by-step explanations when helpful. "
        "Format using Markdown."
    )
    return jsonify({"answer": ask_groq(system, question)})

# ── 2. Summarize Notes ───────────────────────────────────────────
@app.route("/api/summarize", methods=["POST"])
def summarize_notes():
    data   = request.json
    text   = data.get("text", "").strip()
    detail = data.get("detail", "medium")
    if not text:
        return jsonify({"error": "Text is required"}), 400
    length_map = {
        "short":    "in 3-5 bullet points",
        "medium":   "in 1-2 paragraphs with key bullet points",
        "detailed": "in a structured outline with headings, sub-points, and a key-takeaways section"
    }
    system = "You are an academic summarization assistant. Summarize clearly and accurately. Format using Markdown."
    return jsonify({"summary": ask_groq(system,
        f"Summarize the following text {length_map.get(detail, length_map['medium'])}:\n\n{text}",
        max_tokens=2000)})

# ── 3. Generate Quiz ─────────────────────────────────────────────
@app.route("/api/quiz", methods=["POST"])
def generate_quiz():
    data   = request.json
    topic  = data.get("topic", "").strip()
    num_q  = min(int(data.get("num_questions", 5)), 10)
    q_type = data.get("type", "mcq")
    if not topic:
        return jsonify({"error": "Topic is required"}), 400
    type_map = {
        "mcq":       "multiple-choice questions (4 options each, mark correct answer with ✓)",
        "truefalse": "true/false questions (state the answer after each)",
        "short":     "short-answer questions (include a model answer)"
    }
    system = (
        "You are a quiz generator for academic students. "
        "Return ONLY valid JSON — no markdown fences, no preamble. "
        'Schema: {"quiz": [{"question": str, "options": [str] or null, "answer": str}]}'
    )
    raw = ask_groq(system,
        f"Generate {num_q} {type_map.get(q_type, type_map['mcq'])} on the topic: {topic}",
        max_tokens=2000)
    try:
        return jsonify(parse_json_response(raw))
    except Exception:
        return jsonify({"raw": raw})

# ── 4. Make Flashcards ───────────────────────────────────────────
@app.route("/api/flashcards", methods=["POST"])
def make_flashcards():
    data  = request.json
    text  = data.get("text", "").strip()
    topic = data.get("topic", "").strip()
    count = min(int(data.get("count", 8)), 20)
    source = text or topic
    if not source:
        return jsonify({"error": "Text or topic is required"}), 400
    system = (
        "You are a flashcard creator. "
        "Return ONLY valid JSON — no markdown fences, no preamble. "
        '{"flashcards": [{"front": str, "back": str}]}'
    )
    prompt = (f"Create {count} study flashcards from:\n\n{source}" if text
              else f"Create {count} study flashcards on: {topic}")
    raw = ask_groq(system, prompt, max_tokens=2000)
    try:
        return jsonify(parse_json_response(raw))
    except Exception:
        return jsonify({"raw": raw})

# ── 5. Study Plan ────────────────────────────────────────────────
@app.route("/api/studyplan", methods=["POST"])
def study_plan():
    data    = request.json
    subject = data.get("subject", "").strip()
    days    = min(int(data.get("days", 7)), 30)
    goal    = data.get("goal", "master the subject").strip()
    level   = data.get("level", "intermediate")
    if not subject:
        return jsonify({"error": "Subject is required"}), 400
    system = (
        "You are an academic coach. Format the plan using Markdown with a clear "
        "day-by-day schedule, daily time estimates, topics, resources, and revision tips."
    )
    prompt = (f"Create a {days}-day study plan for a {level} student "
              f"who wants to {goal} in {subject}. Include daily tasks, time allocations, and tips.")
    return jsonify({"plan": ask_groq(system, prompt, max_tokens=2500)})

# ── Health check ─────────────────────────────────────────────────
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": MODEL})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
