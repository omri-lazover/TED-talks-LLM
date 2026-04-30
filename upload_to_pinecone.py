import json
import time
import os
from pinecone import Pinecone

# ==========================================
# ⚙️ הגדרות
# ==========================================

# טעינה מקובץ הגיבוי
INPUT_FILE = 'ted_talks_with_embeddings_backup.json'

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
INDEX_NAME = "ted-talks"


def upload_embeddings_to_pinecone():
    print(f"--- Starting Process: Upload from BACKUP to Index '{INDEX_NAME}' ---")

    # 1. בדיקת קיום הקובץ
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: The file '{INPUT_FILE}' is missing.")
        return

    # 2. טעינת הנתונים
    print(f"-> Loading data from {INPUT_FILE}...")
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            records = json.load(f)
        print(f"✅ Successfully loaded {len(records)} records from JSON.")
    except Exception as e:
        print(f"❌ Error reading JSON file: {e}")
        return

    # 3. התחברות ל-Pinecone
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(INDEX_NAME)
        stats = index.describe_index_stats()
        print(f"-> Connected to Pinecone. Vectors currently in DB: {stats.total_vector_count}")

    except Exception as e:
        print(f"❌ Error connecting to Pinecone: {e}")
        return

    # 4. העלאה בקבוצות (Batching)
    BATCH_SIZE = 100
    total_uploaded = 0
    batch_counter = 0  # <--- מונה לאיטרציות

    print("-> Starting batch upload...")

    for i in range(0, len(records), BATCH_SIZE):
        batch = records[i: i + BATCH_SIZE]
        batch_counter += 1

        try:
            # הפעולה עצמה: Upsert
            index.upsert(vectors=batch)
            total_uploaded += len(batch)

            # הדפסה כל 10 איטרציות (1000 רשומות)
            if batch_counter % 10 == 0:
                print(f"   Uploaded batch {i} to {min(i + BATCH_SIZE, len(records))}")

        except Exception as e:
            print(f"   ❌ Error uploading batch starting at {i}: {e}")
            # ניסיון חוזר (Retry)
            try:
                time.sleep(2)
                print(f"   🔄 Retrying batch {i}...")
                index.upsert(vectors=batch)
                print(f"   ✅ Retry success.")
            except Exception as e2:
                print(f"   ❌ Retry failed. Skipping batch {i}.")

        # --- מנוחה יזומה ---
        # אם עברו 10 איטרציות, נחכה חצי שנייה
        if batch_counter % 10 == 0:
            # print("   ☕ Taking a short break (0.5s)...") # אופציונלי להדפיס
            time.sleep(0.5)

    # 5. סיכום ותוצאות
    print("\n-> Waiting for Pinecone to index data...")
    time.sleep(2)
    final_stats = index.describe_index_stats()

    print("\n--- Upload Summary ---")
    print(f"Upload status: Done")
    print(f"Total vectors processed in this run: {total_uploaded}")
    print(f"Total vectors currently in DB: {final_stats.total_vector_count}")


if __name__ == "__main__":
    upload_embeddings_to_pinecone()