import os

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

load_dotenv()

# ==========================================
# קונפיגורציה (מפתחות מ-.env — כמו ב-api.py)
# ==========================================
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

BASE_URL = "https://api.llmod.ai"
INDEX_NAME = "ted-talks"

# שמות המודלים
EMBEDDING_MODEL = "RPRTHPB-text-embedding-3-small"
CHAT_MODEL = "RPRTHPB-gpt-5-mini"

TOP_K = 7

# ==========================================
# אתחול
# ==========================================
if not OPENAI_API_KEY or not PINECONE_API_KEY:
    raise SystemExit(
        "Missing OPENAI_API_KEY or PINECONE_API_KEY. Set them in your .env file."
    )

client = OpenAI(api_key=OPENAI_API_KEY, base_url=BASE_URL)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# System Prompt (מהמסמך)
SYSTEM_PROMPT_TEXT = """
You are a TED Talk assistant that answers questions strictly and 
only based on the TED dataset context provided to you (metadata 
and transcript passages). You must not use any external 
knowledge, the open internet, or information that is not explicitly 
contained in the retrieved context. If the answer cannot be 
determined from the provided context, respond: "I don't know 
based on the provided TED data." Always explain your answer 
using the given context, quoting or paraphrasing the relevant 
transcript or metadata when helpful.
"""


def get_embedding(text):
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text], model=EMBEDDING_MODEL)
    return response.data[0].embedding


def search_pinecone(query):
    vector = get_embedding(query)
    results = index.query(vector=vector, top_k=TOP_K, include_metadata=True)
    return results['matches']


def ask_gpt(question, context_chunks):
    # 1. בניית הקונטקסט
    context_text = ""
    for i, chunk in enumerate(context_chunks):
        meta = chunk['metadata']
        context_text += f"\n--- Source {i + 1} ---\n"
        context_text += f"Talk ID: {meta.get('talk_id', 'N/A')}\n"
        context_text += f"Title: {meta.get('title', 'Unknown')}\n"
        context_text += f"Speaker: {meta.get('speaker', 'Unknown')}\n"
        context_text += f"Text Chunk: {meta.get('text', '')}\n"


    # 2. בניית ההודעה למשתמש
    user_message = f"""
    Context information is below:
    {context_text}

    Question: {question}
    """

    # 3. שליחה למודל (עם התיקון!)
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_TEXT},
            {"role": "user", "content": user_message}
        ],
        temperature=1  # <--- התיקון שפתר את הבעיה
    )

    return response.choices[0].message.content


def main():
    print(f"--- Testing RAG with model: {CHAT_MODEL} ---")

    while True:
        question = input("\nשאל שאלה (באנגלית): ")
        if question.lower() in ['exit', 'quit']:
            break

        print("🔍 Searching Pinecone...")
        chunks = search_pinecone(question)
        print(f"   Found {len(chunks)} relevant segments.")

        # --- הוסף את הבלוק הזה ---
        print("   Debugging - Top results found:")
        for i, c in enumerate(chunks):
            print(f"   {i + 1}. {c['metadata'].get('title', 'No Title')} (Score: {c['score']:.4f})")
        # -------------------------

        print("🤖 Thinking...")
        try:
            answer = ask_gpt(question, chunks)
            print("\n=== ANSWER ===")
            print(answer)
            print("==============")
        except Exception as e:
            print(f"Error form model: {e}")


if __name__ == "__main__":
    main()