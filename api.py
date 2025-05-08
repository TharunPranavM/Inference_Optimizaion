from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import torch
from typing import Optional
import os
import tempfile
import subprocess
import logging
from main import setup_models, chatbot, TEXT_MODEL_PATH, BLIP_MODEL_PATH, WHISPER_MODEL_PATH

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Multimodal Chatbot API",
    description="API for processing text, image, and audio inputs using Gemma, BLIP, and Whisper models.",
    version="1.0.0"
)

# Pydantic model for text input
class TextInput(BaseModel):
    text: str

# Global variables to store models
text_model = None
text_tokenizer = None
blip_model = None
blip_processor = None
transcriber = None

# Startup event to load models
@app.on_event("startup")
async def startup_event():
    global text_model, text_tokenizer, blip_model, blip_processor, transcriber
    save_after_setup = not (
        os.path.exists(TEXT_MODEL_PATH) and
        os.path.exists(BLIP_MODEL_PATH) and
        os.path.exists(WHISPER_MODEL_PATH)
    )
    text_model, text_tokenizer, blip_model, blip_processor, transcriber = setup_models(
        save_after_setup=save_after_setup,
        fine_tune=True
    )

# Shutdown event to clean up
@app.on_event("shutdown")
async def shutdown_event():
    global text_model, text_tokenizer, blip_model, blip_processor, transcriber
    if text_model is not None:
        del text_model
    if text_tokenizer is not None:
        del text_tokenizer
    if blip_model is not None:
        del blip_model
    if blip_processor is not None:
        del blip_processor
    if transcriber is not None:
        del transcriber
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Endpoint for text input
@app.post("/text", summary="Process text input", response_description="Generated response from the chatbot")
async def process_text(input_data: TextInput):
    try:
        response = chatbot(
            input_type="text",
            input_data=input_data.text,
            text_model=text_model,
            text_tokenizer=text_tokenizer,
            blip_model=blip_model,
            blip_processor=blip_processor,
            transcriber=transcriber
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
            blip_model=blip_model,
            blip_processor=blip_processor,
            transcriber=transcriber
        )

        # Clean up temporary file
        os.unlink(temp_file_path)
        return {"response": response}
    except Exception as e:
        # Clean up temporary file in case of error
        if 'temp_file_path' in locals():
            os.unlink(temp_file_path)
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
            blip_model=blip_model,
            blip_processor=blip_processor,
            transcriber=transcriber
        )

        # Clean up temporary file
        os.unlink(temp_file_path)
        logger.info(f"Deleted temporary audio file: {temp_file_path}")
        return {"response": response}
    except Exception as e:
        # Clean up temporary file in case of error
        if 'temp_file_path' in locals():
            os.unlink(temp_file_path)
            logger.info(f"Deleted temporary audio file due to error: {temp_file_path}")
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")

# Root endpoint for basic health check
@app.get("/", summary="Health check", response_description="API status")
async def root():
    return {"message": "Multimodal Chatbot API is running"}