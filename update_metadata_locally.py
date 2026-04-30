import json
import pandas as pd

# ==========================================
# קבצים
# ==========================================
CSV_FILE = 'ted_talks_en.csv'  # הקובץ המקורי עם כל המידע
VECTORS_FILE = 'ted_talks_with_embeddings.json'  # הקובץ שיצרת בשלב 2
OUTPUT_FILE = 'ted_talks_with_embeddings.json'  # הקובץ החדש שנשמור


def add_speaker_to_metadata():
    print("--- Adding Speaker Name to Metadata ---")

    # 1. טעינת קובץ ה-CSV כדי ליצור "מילון דוברים"
    # אנחנו צריכים לדעת איזה talk_id שייך לאיזה speaker_1
    try:
        df = pd.read_csv(CSV_FILE)
        # יוצרים מילון: { '1': 'Al Gore', '2': 'Ken Robinson', ... }
        # המרה ל-str כדי לוודא התאמה ל-id ב-json
        speaker_map = dict(zip(df['talk_id'].astype(str), df['speaker_1']))
        print(f"Loaded speaker mapping for {len(speaker_map)} talks.")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # 2. טעינת קובץ הוקטורים הקיים
    try:
        with open(VECTORS_FILE, 'r', encoding='utf-8') as f:
            records = json.load(f)
        print(f"Loaded {len(records)} vector records.")
    except Exception as e:
        print(f"Error reading JSON: {e}")
        return

    # 3. הוספת הדובר למטא-דאטה
    updated_count = 0
    for record in records:
        # אנחנו צריכים את ה-talk_id מתוך המטא-דאטה הקיימת
        talk_id = str(record['metadata'].get('talk_id'))

        if talk_id in speaker_map:
            # הוספת השדה החדש
            record['metadata']['speaker'] = speaker_map[talk_id]
            updated_count += 1
        else:
            print(f"Warning: Could not find speaker for talk_id {talk_id}")

    print(f"Updated {updated_count} records with speaker names.")

    # 4. שמירה לקובץ חדש (או דריסה של הישן)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False)

    print(f"Done! Saved to {OUTPUT_FILE}")
    print("Now run Step 3 again using this new file.")


if __name__ == "__main__":
    add_speaker_to_metadata()