import json
import time
import os
from openai import OpenAI

# ==========================================
# קונפיגורציה (מותאם ל-llmod.ai)
# ==========================================
INPUT_FILE = 'ted_talks_chunks.json'
OUTPUT_FILE = 'ted_talks_with_embeddings.json'

API_KEY = os.environ.get("OPENAI_API_KEY")

BASE_URL = "https://api.llmod.ai"

# 3. שם המודל המיוחד של הקורס (חובה!)
MODEL_NAME = "RPRTHPB-text-embedding-3-small"

# בדיקה שיש מפתח
if API_KEY == "sk-..." or "..." in API_KEY:
    print("Error: Please replace 'sk-...' with your real API Key from the website.")
    exit()

# חיבור לשרת של הקורס
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)


def get_embeddings_batch(texts):
    """שולח קבוצת טקסטים ומחזיר וקטורים"""
    try:
        # ניקוי ירידות שורה
        clean_texts = [t.replace("\n", " ") for t in texts]

        # שימוש במודל הספציפי של הקורס
        response = client.embeddings.create(input=clean_texts, model=MODEL_NAME)

        return [data.embedding for data in response.data]
    except Exception as e:
        print(f"Error calling API: {e}")
        return None


def main():
    print(f"--- Starting Step 2: Embedding with {BASE_URL} ---")

    # 1. טעינה
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Make sure you ran Step 1.")
        return

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        records = json.load(f)

    total_records = len(records)
    print(f"Loaded {total_records} chunks.")

    # 2. עיבוד בקבוצות
    BATCH_SIZE = 100
    successful_vectors = 0

    for i in range(0, total_records, BATCH_SIZE):
        batch_end = min(i + BATCH_SIZE, total_records)
        print(f"Processing batch {i} to {batch_end}...")

        batch_slice = records[i: batch_end]
        texts_to_embed = [r['metadata']['text'] for r in batch_slice]

        vectors = get_embeddings_batch(texts_to_embed)

        if vectors:
            for idx, vector in enumerate(vectors):
                records[i + idx]['values'] = vector
            successful_vectors += len(vectors)
        else:
            print("Warning: Failed to process batch. Stopping to save budget.")
            break

        time.sleep(0.2)

    # 3. שמירה
    print(f"\nFinished! Embedded {successful_vectors}/{total_records} chunks.")
    print(f"Saving to {OUTPUT_FILE}...")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False)

    print("Done! You are ready for Pinecone.")


if __name__ == "__main__":
    main()