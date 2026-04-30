import requests
import json
import time

BASE_URL = "https://ted-talks-zi9mzlhwc-omri-lazovers-projects.vercel.app"
OUTPUT_FILE = "test_results_full.txt"


def full_dataset():
    # רשימת שאלות מורחבת - 3 מכל סוג
    questions = [
        # ==========================================
        # [cite_start]Type 1: Precise Fact Retrieval [cite: 13-16]
        # ==========================================
        # 1. Ken Robinson (Gillian Lynne story)
        "Find the TED talk where the speaker tells a story about a little girl named Gillian Lynne who couldn't sit still in school but later became a famous dancer. Provide the title and speaker.",

        # # 2. Al Gore (Shoney's story)
        # "Find the talk where the speaker tells a funny story about flying on Air Force Two and eating at a Shoney's restaurant. Provide the title and speaker.",
        #
        # # 3. Joachim de Posada (Marshmallow test)
        # "Which speaker talks about a 'marshmallow test' given to children to measure self-discipline and delayed gratification? Provide the title and speaker.",
        #
        # # ==========================================
        # # [cite_start]Type 2: Multi-Result Topic Listing [cite: 17-21]
        # # ==========================================
        # # 1. Psychology/Happiness
        # "Which TED talks discuss the concept of happiness or psychology? Return a list of exactly 3 talk titles.",
        #
        # # 2. Space/Ocean
        # "Find TED talks related to deep ocean exploration or space travel. Return a list of exactly 3 talk titles.",
        #
        # # 3. Introverts/Personality
        # "I am interested in talks about introverts, personality types, or quiet people. Return a list of exactly 3 talk titles.",
        #
        # # ==========================================
        # # [cite_start]Type 3: Key Idea Summary Extraction [cite: 22-25]
        # # ==========================================
        # # 1. Jill Bolte Taylor (Stroke)
        # "Find a TED talk where a brain researcher describes her own experience having a stroke. Provide the title and a short summary of the key idea.",
        #
        # # 2. Simon Sinek (Golden Circle)
        # "Find the talk that explains the 'Golden Circle' concept (Why, How, What) and why some leaders inspire action. Provide the title and a short summary.",
        #
        # # 3. Brene Brown (Vulnerability)
        # "Find a talk about the power of vulnerability, shame, and human connection. Provide the title and a summary of the key idea.",
        #
        # # ==========================================
        # # [cite_start]Type 4: Recommendation with Evidence [cite: 26-29]
        # # ==========================================
        # # 1. Hans Rosling (Stats)
        # "I am interested in how statistics and data visualization can show us the truth about the developing world and poverty. Which talk would you recommend?",
        #
        # # 2. Amy Cuddy (Body Language)
        # "I'm feeling very nervous about a job interview and want to know how my body language affects my confidence and hormones. Which talk would you recommend?",
        #
        # # 3. Pamela Meyer (Lying)
        # "I suspect someone is lying to me. Is there a TED talk that teaches how to spot a liar using behavioral cues? Which one do you recommend?"
    ]

    # פתיחת הקובץ לכתיבה
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

        def log(message=""):
            print(message)
            f.write(message + "\n")
            f.flush()

        log(f"🚀 --- Starting FULL DATASET Tests ({len(questions)} Questions) ---")
        log(f"📅 Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        log("=" * 60)

        for i, q in enumerate(questions, 1):
            log(f"\n[{i}] Question:")
            log(f"    Query: {q}")
            log("    Sending request...")

            try:
                start_time = time.time()
                response = requests.post(f"{BASE_URL}/api/prompt", json={"question": q})
                elapsed = time.time() - start_time

                if response.status_code == 200:
                    data = response.json()
                    answer = data.get('response', 'No response found')

                    log(f"    ✅ Success ({elapsed:.2f}s)")
                    log("-" * 20 + " ANSWER " + "-" * 20)
                    log(answer)
                    log("-" * 48)
                else:
                    log(f"    ❌ Failed: {response.status_code}")
                    log(f"    Error: {response.text}")

            except Exception as e:
                log(f"    ❌ Error: {e}")

            # המתנה קצרה בין בקשות כדי להקל על השרת המקומי (אופציונלי)
            time.sleep(1)

        log("\n🏁 Tests Completed. Results saved to " + OUTPUT_FILE)


if __name__ == "__main__":
    full_dataset()