import pandas as pd
import json
import os

# ==========================================
# קבועים והגדרות (Configuration)
# ==========================================
# כרגע מוגדר לקובץ המוקטן לצורך בדיקה.
# לריצה הסופית - תחליף חזרה ל-'ted_talks_en.csv'
INPUT_CSV_FILE = 'ted_talks_en_reduced.csv'

CHUNK_SIZE_WORDS = 1000
OVERLAP_WORDS = 100
OUTPUT_FILE = 'ted_talks_chunks.json'  # הקובץ שיווצר


# ==========================================
# פונקציה 1: טעינת הנתונים
# ==========================================
def load_dataset(filepath):
    print(f"--- Loading data from {filepath} ---")
    if not os.path.exists(filepath):
        print(f"❌ Error: File '{filepath}' not found!")
        print("Tip: Make sure you ran the script that creates the reduced CSV first.")
        return None

    try:
        df = pd.read_csv(filepath)
        print(f"✅ Successfully loaded {len(df)} talks.")
        return df
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return None


# ==========================================
# פונקציה 2: הלוגיקה של החיתוך (Chunking)
# ==========================================
def split_text_into_chunks(text, chunk_size, overlap):
    # טיפול במקרים של טקסט חסר
    if pd.isna(text) or text == "":
        return []

    words = text.split()
    chunks = []

    # אם הטקסט קצר יותר מהגודל המקסימלי, נחזיר אותו כצ'אנק אחד
    if len(words) <= chunk_size:
        return [text]

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]

        # סינון רעשים: אם נשאר צ'אנק פצפון (פחות מ-20 מילים), נתעלם ממנו
        if len(chunk_words) > 20:
            chunk_text = " ".join(chunk_words)
            chunks.append(chunk_text)

        start += (chunk_size - overlap)

    return chunks


# ==========================================
# פונקציה 3: אריזה לפורמט Pinecone עם Context Enrichment
# ==========================================
def format_talk_records(row, chunks):
    records = []

    # חילוץ וניקוי שם הדובר והכותרת
    speaker_name = row.get('speaker_1', 'Unknown')
    if pd.isna(speaker_name): speaker_name = 'Unknown'

    title = row.get('title', 'Unknown Title')
    if pd.isna(title): title = 'Unknown Title'

    for i, chunk_text in enumerate(chunks):
        # --- Context Enrichment ---
        # אנחנו יוצרים טקסט חדש שמכיל בתוכו את המטא-דאטה.
        # זה מה שיישלח למודל ה-Embedding ויבטיח שהוקטור "יודע" מי הדובר.
        enriched_text = f"Title: {title}\nSpeaker: {speaker_name}\nTranscript segment: {chunk_text}"

        record = {
            "id": f"{row['talk_id']}_{i}",
            "values": [],  # זה יתמלא בשלב הבא (embedding.py)
            "metadata": {
                "talk_id": str(row['talk_id']),
                "title": title,
                "speaker": speaker_name,
                # שים לב: אנחנו שומרים את הטקסט המועשר!
                "text": enriched_text,
                "url": row.get('url', ''),
                "chunk_index": i
            }
        }
        records.append(record)
    return records


# ==========================================
# פונקציה 4: הפונקציה הראשית (Orchestrator)
# ==========================================
def create_json():
    df = load_dataset(INPUT_CSV_FILE)
    if df is None: return

    all_ready_records = []

    print("--- Processing talks and enriching context ---")
    # מעבר על כל שורה ב-DataFrame
    for index, row in df.iterrows():
        chunks = split_text_into_chunks(row.get('transcript', ''), CHUNK_SIZE_WORDS, OVERLAP_WORDS)
        talk_records = format_talk_records(row, chunks)
        all_ready_records.extend(talk_records)

    print(f"\n✅ Success! Prepared {len(all_ready_records)} enriched chunks from {len(df)} talks.")
    return all_ready_records


if __name__ == "__main__":
    res = create_json()

    if res:
        # שמירה לקובץ JSON
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(res, f, ensure_ascii=False, indent=4)
        print(f"💾 Data saved to '{OUTPUT_FILE}'")
        print("\nNext Steps:")
        print("1. Run 'embedding.py' (Wait for it to finish).")
        print("2. Run 'upload_to_pinecone.py'.")
        print("3. Run 'check_server_full.py' to verify.")