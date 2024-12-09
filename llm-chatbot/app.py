from flask import Flask, request, jsonify
import chat, chat_model
import re
from flask_cors import CORS
import langid
import json

app = Flask(__name__)

CORS(app)

# Fungsi untuk memuat kata kunci dari file JSON
def load_keywords():
    with open("keywords.json", "r", encoding="utf-8") as f:
        return json.load(f)

# Memuat daftar kata kunci
keywords = load_keywords()

def detect_language(text):
    text = text.lower()
    
    # Cek kata kunci untuk setiap bahasa yang diperbolehkan
    for lang, lang_keywords in keywords.items():
        if any(keyword in text for keyword in lang_keywords):
            return lang
    
    # Jika tidak ada kata kunci yang terdeteksi, gunakan langid sebagai fallback
    lang, _ = langid.classify(text)
    return lang

@app.route("/chat", methods=["POST"])
def chatBot():
    json_content = request.json
    message = json_content.get("message")
    conversation_history = json_content.get("conversation_history", "")

    # Deteksi bahasa menggunakan fungsi detect_language
    lang = detect_language(message)

    # Jika bahasa yang terdeteksi bukan salah satu bahasa yang diperbolehkan
    allowed_languages = ["id", "en", "zh", "fr", "es", "pt", "de", "it", "ru", "ja", "ko", "vi", "th", "ar"]
    if lang not in allowed_languages:
        return jsonify({"response": "Untuk menggunakan Satria BIPA, gunakan salah satu bahasa berikut: Indonesia, Inggris, Mandarin, Prancis, Spanyol, Portugis, Jerman, Italia, Rusia, Jepang, Korea, Vietnam, Thailand, atau Arab."})

    # Jika bahasa valid, lanjutkan dengan proses model
    response = chat.answer_question_with_context(conversation_history, message)

    # Post-processing of the response
    if isinstance(response, dict) and "response" in response:
        if "Alibaba Cloud" in response["response"]:
            response["response"] = response["response"].replace("Alibaba Cloud", "Badan Pengembangan dan Pembinaan Bahasa, Kementerian Pendidikan, Kebudayaan")
        elif "please" in response["response"]:
            response["response"] = response["response"].replace("please", "ya")
        elif "Berapa harganya " in response["response"]:
            response["response"] = response["response"].replace("Berapa harganya ", "Berapa harga ")
        elif "Indonesian" in response["response"]:
            response["response"] = response["response"].replace("Indonesian", "Bahasa Indonesia")
        response["response"] = re.sub(r'"(.*?)"', r'*\1*', response["response"])

    return jsonify(response)

@app.route("/chat-model", methods=["POST"])
def chatModel():
    json_content = request.json
    message = json_content.get("message")

    # Proses model
    response = chat_model.answer_question_with_context(message)

    # Cek jika response adalah dictionary dan mengandung key "response"
    if isinstance(response, dict) and "response" in response:
        # Ganti kata "Alibaba Cloud" jika ada dalam response
        if "Alibaba Cloud" in response["response"]:
            response["response"] = response["response"].replace("Alibaba Cloud", "Badan Pengembangan dan Pembinaan Bahasa, Kementerian Pendidikan, Kebudayaan")
        elif "please" in response["response"]:
            response["response"] = response["response"].replace("please", "ya")
        elif "Berapa harganya " in response["response"]:
            response["response"] = response["response"].replace("Berapa harganya ", "Berapa harga ")
        elif "Indonesian" in response["response"]:
            response["response"] = response["response"].replace("Indonesian", "Bahasa Indonesia")
        # Format respons dengan menambahkan tanda * untuk setiap kata dalam tanda kutip
        response["response"] = re.sub(r'"(.*?)"', r'*\1*', response["response"])

    return jsonify(response)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6845, debug=True)
