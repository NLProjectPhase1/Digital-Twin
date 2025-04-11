from flask import Flask, render_template, request, jsonify, Response
from faster_whisper import WhisperModel
from gtts import gTTS
import requests
import os
import tempfile
import io
import zipfile
import gdown
import shutil

# Check if ffmpeg is available
import shutil as sh
print("🎵 FFmpeg found at:", sh.which("ffmpeg"))

app = Flask(__name__)

def download_whisper_model():
    base_tmp = tempfile.gettempdir()
    model_dir = os.path.join(base_tmp, "whisper-tiny")
    zip_path = os.path.join(base_tmp, "whisper-tiny.zip")

    if not os.path.exists(os.path.join(model_dir, "model.bin")):
        print("📦 Downloading Whisper Tiny model from Google Drive...")

        file_id = "1KNh6Nrg3LF1-zK5H3Kwd0tDgvDO7jkU8"
        download_url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(download_url, zip_path, quiet=False)

        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)

        os.makedirs(model_dir, exist_ok=True)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(model_dir)

        os.remove(zip_path)

        nested_dir = os.path.join(model_dir, "whisper-tiny")
        if os.path.exists(nested_dir) and os.path.isdir(nested_dir):
            print("📁 Fixing nested model directory...")
            for item in os.listdir(nested_dir):
                shutil.move(os.path.join(nested_dir, item), model_dir)
            os.rmdir(nested_dir)

        print("✅ Whisper model extracted to:", model_dir)

    return model_dir

try:
    model_path = download_whisper_model()
    stt_model = WhisperModel(model_path, device="cpu", local_files_only=True)
    print("🧠 Whisper model loaded successfully.")
except Exception as e:
    raise RuntimeError(f"❌ Failed to load Whisper model: {e}")

# === OpenRouter API Key ===
OPENROUTER_API_KEY = "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"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/stt", methods=["POST"])
def speech_to_text():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file received"}), 400

    audio_file = request.files["audio"]
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
        audio_file.save(temp_audio_file.name)
        temp_audio_path = temp_audio_file.name

    try:
        print(f"🎧 Audio received and saved at: {temp_audio_path}")
        segments, _ = stt_model.transcribe(temp_audio_path)
        transcribed_text = " ".join([segment.text for segment in segments]).strip()
        print(f"📝 Transcribed text: {transcribed_text}")

        if not transcribed_text:
            return jsonify({"error": "Transcription failed or was empty"}), 500

        ai_message = get_ai_response(transcribed_text)
        print(f"🤖 AI Response: {ai_message}")

        tts_audio = convert_text_to_speech(ai_message)

    except Exception as e:
        print(f"❌ Error during transcription: {e}")
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

    return jsonify({
        "transcribed_text": transcribed_text,
        "ai_response": ai_message,
        "tts_audio_url": "/tts_audio",
    })

def get_ai_response(user_input):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "meta-llama/llama-3-8b-instruct",
        "messages": [{"role": "user", "content": f"{user_input} (Respond briefly in 2-3 sentences)"}],
    }

    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        ai_message = response.json()["choices"][0]["message"]["content"].strip()
        return ai_message

    except Exception as e:
        print(f"❌ OpenRouter Error: {e}")
        return "I'm sorry, I couldn't process your request right now."

def convert_text_to_speech(text):
    print("🔊 Converting AI response to speech...")
    tts = gTTS(text=text, lang='en')
    audio_data = io.BytesIO()
    tts.write_to_fp(audio_data)
    audio_data.seek(0)
    return audio_data

@app.route("/tts_audio", methods=["POST"])
def tts_audio():
    data = request.get_json()
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        print(f"🔈 Generating speech for: {text}")
        tts = gTTS(text=text, lang='en')
        audio_data = io.BytesIO()
        tts.write_to_fp(audio_data)
        audio_data.seek(0)
        return Response(audio_data, mimetype="audio/mpeg")

    except Exception as e:
        print(f"❌ Error during TTS: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))