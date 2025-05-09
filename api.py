from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import torch
from typing import Optional
import os
import tempfile
import subprocess
import logging
from main import TEXT_MODEL_PATH, BLIP_MODEL_PATH, WHISPER_MODEL_PATH, LORA_ADAPTER_PATH, chatbot, check_vram
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel  # Correct import from peft

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Multimodal Chatbot API",
    description="API for processing text, image, and audio naturally, using Gemma, BLIP, and Whisper models.",
    version="1.0.0"
)

# Pydantic model for text input
class TextInput(BaseModel):
    text: str

# Global variables to store models
text_model = None
text_tokenizer = None

# Startup event to load fine-tuned Gemma model
@app.on_event("startup")
async def startup_event():
    global text_model, text_tokenizer
    print("Loading fine-tuned Gemma model at startup...")
    text_model = AutoModelForCausalLM.from_pretrained(
        TEXT_MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.float16,
        use_cache=True
    )
    text_tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_PATH)
    if os.path.exists(os.path.join(LORA_ADAPTER_PATH, "adapter_config.json")):
        text_model = PeftModel.from_pretrained(text_model, LORA_ADAPTER_PATH)
    print("VRAM status after loading Gemma at startup:")
    check_vram()

# Shutdown event to clean up
@app.on_event("shutdown")
async def shutdown_event():
    global text_model, text_tokenizer
    if text_model is not None:
        del text_model
    if text_tokenizer is not None:
        del text_tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("VRAM status after shutdown:")
    check_vram()

# Endpoint for text input
@app.post("/text", summary="Process text input", response_description="Generated response from the chatbot")
async def process_text(input_data: TextInput):
    try:
        response = chatbot(
            input_type="text",
            input_data=input_data.text,
            text_model=text_model,
            text_tokenizer=text_tokenizer,
            blip_model=None,
            blip_processor=None,
            transcriber=None
        )
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing text: {str(e)}")

# Endpoint for image input
@app.post("/image", summary="Process image input", response_description="Generated response from the chatbot")
async def process_image(image_file: UploadFile = File(...)):
    try:
        # Save uploaded image to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            temp_file.write(await image_file.read())
            temp_file_path = temp_file.name
            logger.info(f"Saved image to temporary file: {temp_file_path}")

        response = chatbot(
            input_type="image",
            input_data=temp_file_path,
            text_model=text_model,
            text_tokenizer=text_tokenizer,
            blip_model=None,
            blip_processor=None,
            transcriber=None
        )

        # Clean up temporary file
        os.unlink(temp_file_path)
        logger.info(f"Deleted temporary image file: {temp_file_path}")
        return {"response": response}
    except Exception as e:
        if 'temp_file_path' in locals():
            os.unlink(temp_file_path)
            logger.info(f"Deleted temporary image file due to error: {temp_file_path}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

# Endpoint for audio input
@app.post("/audio", summary="Process audio input", response_description="Generated response from the chatbot")
async def process_audio(audio_file: UploadFile = File(...)):
    try:
        # Save uploaded audio to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            content = await audio_file.read()
            if not content:
                raise HTTPException(status_code=400, detail="Uploaded audio file is empty")
            temp_file.write(content)
            temp_file_path = temp_file.name
            logger.info(f"Saved audio to temporary file: {temp_file_path}")

        # Check if ffmpeg is available
        try:
            ffmpeg_version = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=True)
            logger.info(f"FFmpeg version: {ffmpeg_version.stdout.splitlines()[0]}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise HTTPException(
                status_code=500,
                detail="FFmpeg is not installed or not found in PATH. Please install FFmpeg to process audio files. "
                       "Download from https://ffmpeg.org/download.html and add to PATH."
            )

        # Verify audio file is readable
        if not os.path.exists(temp_file_path) or os.path.getsize(temp_file_path) == 0:
            raise HTTPException(status_code=400, detail=f"Audio file at {temp_file_path} is invalid or empty")

        response = chatbot(
            input_type="audio",
            input_data=temp_file_path,
            text_model=text_model,
            text_tokenizer=text_tokenizer,
            blip_model=None,
            blip_processor=None,
            transcriber=None
        )

        # Clean up temporary file
        os.unlink(temp_file_path)
        logger.info(f"Deleted temporary audio file: {temp_file_path}")
        return {"response": response}
    except Exception as e:
        if 'temp_file_path' in locals():
            os.unlink(temp_file_path)
            logger.info(f"Deleted temporary audio file due to error: {temp_file_path}")
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")

# Root endpoint for basic health check
@app.get("/", summary="Health check", response_description="API status")
async def root():
    return {"message": "Multimodal Chatbot API is running"}