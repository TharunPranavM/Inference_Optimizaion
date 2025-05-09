Here's an enhanced and **attractive version** of your project documentation. It includes better formatting, clearer sections, and visual/semantic emphasis to improve readability and engagement for developers and stakeholders.

---

# 🚀 Multimodal Chatbot with **Gemma**, **BLIP**, and **Whisper**

An interactive AI assistant that understands **text**, **images**, and **audio**, all in one sleek system. Ideal for **financial analysis**, **captioning**, and **voice transcription**.

---

## 📌 Overview

This project is a **multimodal chatbot** combining top-tier models for diverse inputs:

* 📝 **Text** → *Sentiment analysis* using **Gemma 2B + LoRA**
* 🖼️ **Images** → *Captioning* via **Salesforce's BLIP**
* 🔊 **Audio** → *Speech-to-text* with **OpenAI's Whisper**

💡 Features:

* Streamlit frontend for seamless interaction
* FastAPI backend for scalable API access
* Real-time benchmarking: latency, memory, TTFT, and more

---

## 🧠 Architecture

```plaintext
┌─────────────────────────────────────────────────────────────────────────┐
│                      🌐 Streamlit Frontend (`app.py`)                    │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        🚀 FastAPI Backend (`api.py`)                     │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         🔧 Core Engine (`main.py`)                       │
├─────────────────┬──────────────────┬───────────────────┬────────────────┤
│ 📄 Text Input   │ 🖼️ Image Input   │ 🔉 Audio Input    │ 📊 Benchmarking │
├─────────────────┼──────────────────┼───────────────────┼────────────────┤
│ Gemma + LoRA    │ BLIP Captioning  │ Whisper Tiny ASR  │ Metrics:       │
│ Sentiment Model │ Descriptive Tags │ Transcription     │ Latency, TTFT… │
└─────────────────┴──────────────────┴───────────────────┴────────────────┘
```

---

## ⚙️ Workflow

1. **User Input**

   * Text (e.g. financial statements)
   * Image (e.g. charts or scenes)
   * Audio (e.g. voice memos)

2. **Backend Routing**

   * FastAPI receives & routes to appropriate model

3. **Model Processing**

   * Lightweight, quantized models loaded dynamically

4. **Response Generation**

   * Clean, actionable output for users

5. **Benchmarking**

   * Tracks VRAM usage, response time, throughput, etc.

---

## 💻 Installation & Setup

### ✅ Prerequisites

* Python 3.8+
* CUDA GPU (8GB+ VRAM)
* [FFmpeg](https://ffmpeg.org) installed
* Hugging Face token (for model downloads)

### 🧪 Setup Instructions

```bash
# 1. Clone repository
git clone https://github.com/yourusername/multimodal-chatbot.git
cd multimodal-chatbot

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # (Windows: venv\Scripts\activate)

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add Hugging Face token
echo "HF_TOKEN=your_token_here" > .env
```

---

## 📁 Directory Structure

```bash
multimodal-chatbot/
├── main.py                # 🧠 Core model processing
├── api.py                 # ⚙️ FastAPI server
├── app.py                 # 🌐 Streamlit UI
├── requirements.txt       # 📦 Dependencies
├── .env                   # 🔐 Env vars
├── saved_models/          # 📥 Cached models
│   ├── gemma_quantized/
│   ├── blip_quantized/
│   ├── whisper_tiny_pipeline/
├── gemma_lora_finance/    # 💰 Financial LoRA adapter
```

---

## ▶️ Running the App

1. **Start FastAPI server:**

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

2. **Launch Streamlit UI:**

```bash
streamlit run app.py
```

3. Open your browser at: [http://localhost:8501](http://localhost:8501)

---

## 🧬 Model Details

### 🧾 Gemma 2B

* Quantized (4-bit) for memory efficiency
* Fine-tuned with **LoRA** for financial sentiment tasks

### 🖼️ BLIP

* Base image captioning model
* Converts images into human-readable descriptions

### 🎤 Whisper-Tiny

* Efficient speech-to-text
* Converts voice inputs into structured responses

---

## ⚡ Performance Optimization

* **Quantization**: 4-bit precision for all models
* **Dynamic Model Loading**: Reduces VRAM usage
* **LoRA Tuning**: Efficient domain adaptation
* **Benchmarking Tools**: Track latency, TTFT, memory, energy

---

## 📡 API Reference

### 📝 `POST /text`

```json
{
  "text": "The company reported a strong profit increase."
}
```

---

### 🖼️ `POST /image`

* Upload an image under form field: `image_file`

---

### 🔊 `POST /audio`

* Upload an audio file under form field: `audio_file`

---

## 🛠️ Troubleshooting

| Problem                  | Solution                                 |
| ------------------------ | ---------------------------------------- |
| `CUDA out of memory`     | Use smaller models or reduce batch size  |
| `Audio processing error` | Install `ffmpeg` and ensure it's in PATH |
| `Model not loading`      | Check Hugging Face token permissions     |

---

## 🌱 Future Improvements

* 🎥 Video support and structured data ingestion
* 🔄 Streaming responses (low-latency UX)
* 💬 Multi-turn memory and dialogue flow
* 📊 More domain-specific LoRA adapters
* 🗜️ Model compression for edge deployment

---

Would you like me to create a **visual diagram**, an **HTML readme**, or convert this into a **presentation format**?
