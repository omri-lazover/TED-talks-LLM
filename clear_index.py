import os

from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
INDEX_NAME = "ted-talks"


def clear_pinecone_index():
    if not PINECONE_API_KEY:
        print("❌ Error: PINECONE_API_KEY is missing. Set it in .env")
        return

    print(f"--- Connecting to Pinecone Index '{INDEX_NAME}' ---")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)

    # בדיקת מצב לפני
    stats = index.describe_index_stats()
    print(f"Current vector count: {stats.total_vector_count}")

    if stats.total_vector_count > 0:
        print("🗑️  Deleting all vectors... (This might take a few seconds)")

        # הפקודה שמוחקת הכל
        index.delete(delete_all=True)

        print("✅ Index cleared successfully.")
    else:
        print("✅ Index is already empty.")


if __name__ == "__main__":
    clear_pinecone_index()
