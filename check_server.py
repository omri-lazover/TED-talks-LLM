import requests

# הכתובת שלך ב-Vercel
BASE_URL = "https://ted-talks-zi9mzlhwc-omri-lazovers-projects.vercel.app"


def check_get_request():
    url = f"{BASE_URL}/api/stats"
    print(f"📡 Testing GET request to: {url}...")

    try:
        response = requests.get(url)

        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            print("✅ Success! Server is UP and responding.")
            print("📄 Response JSON:")
            print(response.json())
        else:
            print("❌ Failed. Server returned an error.")
            print("Response text:", response.text)

    except Exception as e:
        print(f"❌ Error connecting to server: {e}")


if __name__ == "__main__":
    check_get_request()