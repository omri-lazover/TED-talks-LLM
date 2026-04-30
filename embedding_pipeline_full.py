import pandas as pd
import json
import time
import os
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

load_dotenv()

# ==========================================
# ⚙️ קונפיגורציה והגדרות
# ==========================================

# --- שליטה על כמות הדאטה ---
PROCESS_ALL_DATA = True  # <--- שנה ל-True כדי לעבד את כל הקובץ!
ROWS_LIMIT_IF_TESTING = 500  # משפיע רק אם PROCESS_ALL_DATA = False

# קבצים
ORIGINAL_CSV_FILE = "ted_talks_en.csv"
BACKUP_JSON_FILE = "ted_talks_with_embeddings_backup.json"

# פרמטרים לחיתוך טקסט
CHUNK_SIZE_WORDS = 512
OVERLAP_WORDS = 50

# הגדרות OpenAI / llmod.ai (מ-.env בלבד — לא לשמור מפתחות בקוד)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
BASE_URL = "https://api.llmod.ai"
MODEL_NAME = "RPRTHPB-text-embedding-3-small"

# הגדרות Pinecone
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "")
INDEX_NAME = "ted-talks"

# חיבור ללקוחות
client = OpenAI(api_key=OPENAI_API_KEY, base_url=BASE_URL)
pc = Pinecone(api_key=PINECONE_API_KEY)


# ==========================================
# 1️⃣ שלב ראשון: טעינה וצמצום דאטה
# ==========================================
def load_data(filepath, process_all, limit):
    print(f"\n[Step 1/4] Loading data...")
    if not os.path.exists(filepath):
        print(f"❌ Error: File '{filepath}' not found!")
        return None

    try:
        df = pd.read_csv(filepath)
        total_rows = len(df)
        print(f"   Original file contains {total_rows} rows.")

        if process_all:
            print(f"   🚀 MODE: Processing ALL data ({total_rows} talks).")
            return df
        else:
            print(f"   🧪 MODE: Testing with top {limit} rows only.")
            return df.head(limit)

    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return None


# ==========================================
# 2️⃣ שלב שני: חיתוך והעשרת טקסט (Chunking)
# ==========================================
def split_text_into_chunks(text, chunk_size, overlap):
    if pd.isna(text) or text == "":
        return []

    words = text.split()
    chunks = []

    if len(words) <= chunk_size:
        return [text]

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]

        if len(chunk_words) > 20:  # סינון רעשים
            chunk_text = " ".join(chunk_words)
            chunks.append(chunk_text)

        start += (chunk_size - overlap)

    return chunks


def process_chunks_with_enrichment(df):
    print(f"\n[Step 2/4] Chunking text and enriching context...")
    all_records = []

    for index, row in df.iterrows():
        speaker_name = row.get('speaker_1', 'Unknown')
        if pd.isna(speaker_name): speaker_name = 'Unknown'

        title = row.get('title', 'Unknown Title')
        if pd.isna(title): title = 'Unknown Title'

        transcript = row.get('transcript', '')
        chunks = split_text_into_chunks(transcript, CHUNK_SIZE_WORDS, OVERLAP_WORDS)

        for i, chunk_text in enumerate(chunks):
            # Context Enrichment
            enriched_text = f"Title: {title}\nSpeaker: {speaker_name}\nTranscript segment: {chunk_text}"

            record = {
                "id": f"{row['talk_id']}_{i}",
                "values": [],
                "metadata": {
                    "talk_id": str(row['talk_id']),
                    "title": title,
                    "speaker": speaker_name,
                    "text": enriched_text,
                    "url": row.get('url', ''),
                    "chunk_index": i
                }
            }
            all_records.append(record)

    print(f"   Created {len(all_records)} chunks from {len(df)} talks.")
    return all_records


# ==========================================
# 3️⃣ שלב שלישי: יצירת וקטורים (Embedding)
# ==========================================
def get_embeddings_batch(texts):
    try:
        clean_texts = [t.replace("\n", " ") for t in texts]
        response = client.embeddings.create(input=clean_texts, model=MODEL_NAME)
        return [data.embedding for data in response.data]
    except Exception as e:
        print(f"❌ Error calling API: {e}")
        return None


def generate_embeddings(records):
    print(f"\n[Step 3/4] Generating embeddings via {BASE_URL}...")
    total_records = len(records)
    batch_size = 100
    successful_vectors = 0

    # חישוב הערכת זמן גסה
    print(f"   Est. batches: {total_records // batch_size + 1}")

    for i in range(0, total_records, batch_size):
        batch_end = min(i + batch_size, total_records)
        # הדפסה כל 500 רשומות כדי לא להציף את המסך
        if i % 500 == 0:
            print(f"   Processing batch {i} to {batch_end}...")

        batch_slice = records[i: batch_end]
        texts_to_embed = [r['metadata']['text'] for r in batch_slice]

        vectors = get_embeddings_batch(texts_to_embed)

        if vectors:
            for idx, vector in enumerate(vectors):
                records[i + idx]['values'] = vector
            successful_vectors += len(vectors)
        else:
            print("❌ Warning: Failed to process batch.")
            break

        time.sleep(0.1)  # מנוחה קצרה

    print(f"   Finished! Embedded {successful_vectors}/{total_records} chunks.")
    return records


# ==========================================
# 4️⃣ שלב רביעי: העלאה ל-Pinecone
# ==========================================
def upload_to_pinecone(records):
    print(f"\n[Step 4/4] Uploading to Pinecone Index '{INDEX_NAME}'...")

    try:
        index = pc.Index(INDEX_NAME)
        stats = index.describe_index_stats()
        print(f"   Connected. Current vectors in DB: {stats.total_vector_count}")
    except Exception as e:
        print(f"❌ Error connecting to Pinecone: {e}")
        return

    BATCH_SIZE = 100
    total_uploaded = 0

    for i in range(0, len(records), BATCH_SIZE):
        batch = records[i: i + BATCH_SIZE]
        try:
            index.upsert(vectors=batch)
            total_uploaded += len(batch)
            if i % 1000 == 0:  # הדפסה רק כל 1000 רשומות
                print(f"   Uploaded batch {i} to {i + len(batch)}")
        except Exception as e:
            print(f"   ❌ Error uploading batch {i}: {e}")

    # בדיקה סופית
    time.sleep(5)  # לחכות שפיינקון יתעדכן
    final_stats = index.describe_index_stats()
    print(f"\n✅ Upload Summary:")
    print(f"   Sent: {total_uploaded}")
    print(f"   Total in DB now: {final_stats.total_vector_count}")


# ==========================================
# 🏁 Main Orchestrator
# ==========================================
def main():
    print("🚀 STARTING FULL PIPELINE")
    print("=" * 40)

    if not OPENAI_API_KEY or not PINECONE_API_KEY:
        print("❌ Missing OPENAI_API_KEY or PINECONE_API_KEY. Set them in a .env file (not committed).")
        return

    if PROCESS_ALL_DATA:
        print("⚠️  WARNING: You are about to process the ENTIRE dataset.")
        print("    This might take 10-20 minutes depending on the size.")
        print("=" * 40)
        time.sleep(2)  # זמן לקרוא את האזהרה

    # 1. טעינה
    df = load_data(ORIGINAL_CSV_FILE, PROCESS_ALL_DATA, ROWS_LIMIT_IF_TESTING)
    if df is None: return

    # 2. חיתוך
    chunks_data = process_chunks_with_enrichment(df)
    if not chunks_data: return

    # 3. Embedding
    final_data = generate_embeddings(chunks_data)

    # שמירת גיבוי
    print(f"\n💾 Saving backup to '{BACKUP_JSON_FILE}'...")
    with open(BACKUP_JSON_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False, indent=4)

    # 4. העלאה ל-Pinecone
    upload_to_pinecone(final_data)

    print("\n🎉 DONE! Workflow complete.")


if __name__ == "__main__":
    main()