from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv

# 1. טעינת משתנים מקובץ .env
load_dotenv()

app = Flask(__name__)
CORS(app)

# ========================================================
# 🔑 הגדרות וקונפיגורציה
# ========================================================

# מפתחות (נטענים מהסביבה)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

# הגדרות מודלים ושרת (לפי ה-PDF והסביבה שלך)
BASE_URL = "https://api.llmod.ai"
INDEX_NAME = "ted-talks"
EMBEDDING_MODEL = "RPRTHPB-text-embedding-3-small"
CHAT_MODEL = "RPRTHPB-gpt-5-mini"

# Hyperparameters (לפי מה שהגדרת בתרגיל) [cite: 91-96]
CHUNK_SIZE = 512
OVERLAP_RATIO = 0.1
TOP_K = 10  # הגדלתי ל-20 כי זה מה שעבד טוב בטסטים שלך

# System Prompt (חובה לפי דרישות המטלה) [cite: 46-52]
SYSTEM_PROMPT_TEXT = """You are a TED Talk assistant that answers questions strictly and
only based on the TED dataset context provided to you (metadata
and transcript passages). You must not use any external
knowledge, the open internet, or information that is not explicitly
contained in the retrieved context. If the answer cannot be
determined from the provided context, respond: "I don't know
based on the provided TED data." Always explain your answer
using the given context, quoting or paraphrasing the relevant
transcript or metadata when helpful. You may add additional clarifications (e.g., response style), but you must
keep the above constraints."""

# בדיקות תקינות
if not OPENAI_API_KEY:
    print("❌ Error: OPENAI_API_KEY is missing!")
if not PINECONE_API_KEY:
    print("❌ Error: PINECONE_API_KEY is missing!")

# חיבור ללקוחות
# שימו לב: משתמשים ב-OPENAI_API_KEY שהוגדר למעלה
client = OpenAI(api_key=OPENAI_API_KEY, base_url=BASE_URL)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)


# ==========================================
# 🛠️ פונקציות עזר
# ==========================================
def get_embedding(text):
    # הסרת ירידות שורה לשיפור הוקטור
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text], model=EMBEDDING_MODEL)
    return response.data[0].embedding


def search_pinecone(query):
    vector = get_embedding(query)
    # שליפת המטא-דאטה כדי שנוכל לבנות את הקונטקסט
    results = index.query(vector=vector, top_k=TOP_K, include_metadata=True)
    return results['matches']


# ==========================================
# 🛣️ Endpoints (API)
# ==========================================

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """
    מחזיר את ההגדרות של המערכת בפורמט JSON ספציפי.
    [cite: 89-101]
    """
    return jsonify({
        "chunk_size": CHUNK_SIZE,
        "overlap_ratio": OVERLAP_RATIO,
        "top_k": TOP_K
    })


@app.route('/api/prompt', methods=['POST'])
def handle_prompt():
    """
    מקבל שאלה, מריץ RAG, ומחזיר תשובה + קונטקסט + פרומפט.
    [cite: 63-88]
    """
    data = request.json
    question = data.get('question', '')

    if not question:
        return jsonify({"error": "No question provided"}), 400

    # 1. חיפוש ב-Pinecone
    try:
        chunks = search_pinecone(question)
    except Exception as e:
        return jsonify({"error": f"Search failed: {str(e)}"}), 500

    # 2. בניית הקונטקסט והפרומפט
    context_text = ""
    context_list_for_json = []

    for chunk in chunks:
        meta = chunk['metadata']
        score = chunk['score']

        # חילוץ מידע בטוח (עם ערכי ברירת מחדל)
        c_text = meta.get('text', '')
        c_title = meta.get('title', 'Unknown')
        c_id = str(meta.get('talk_id', 'N/A'))

        # הוספה לטקסט שנשלח ל-GPT
        context_text += f"---\nTitle: {c_title}\nID: {c_id}\nContent: {c_text}\n"

        # הוספה לרשימה שחוזרת ב-JSON למשתמש
        context_list_for_json.append({
            "talk_id": c_id,
            "title": c_title,
            "chunk": c_text,
            "score": score
        })

    # הודעת המשתמש המלאה (Prompt Engineering)
    user_message_content = f"""
Answer the question based ONLY on the following context:

{context_text}

Question: {question}
"""

    # 3. שליחה למודל GPT
    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_TEXT},
                {"role": "user", "content": user_message_content}
            ],
            temperature=1 # טמפרטורה נמוכה לתשובות עובדתיות יותר
        )
        answer = response.choices[0].message.content
    except Exception as e:
        return jsonify({"error": f"LLM Error: {str(e)}"}), 500

    # 4. בניית התשובה הסופית
    return jsonify({
        "response": answer,
        "context": context_list_for_json,
        "Augmented_prompt": {
            "System": SYSTEM_PROMPT_TEXT,
            "User": user_message_content
        }
    })


if __name__ == '__main__':
    # הרצה מקומית
    app.run(debug=True, port=5000)