import torch
import subprocess
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, BlipProcessor, BlipForConditionalGeneration, pipeline, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, PeftModel
from PIL import Image
from huggingface_hub import login
from dotenv import load_dotenv
import os
from datasets import load_dataset

# Load environment variables
load_dotenv()

# Hugging Face token
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable not set. Please set it or provide a valid token.")

# Log in to Hugging Face Hub
login(token=HF_TOKEN)

# Determine device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Paths for saving models and adapters
TEXT_MODEL_PATH = "./saved_models/gemma_quantized"
BLIP_MODEL_PATH = "./saved_models/blip_quantized"
WHISPER_MODEL_PATH = "./saved_models/whisper_tiny_pipeline"
LORA_ADAPTER_PATH = "./gemma_lora_finance"

# Define quantization configuration for 4-bit
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

# Function to check VRAM usage using nvidia-smi
def check_vram():
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader"], capture_output=True, text=True)
        used, total = result.stdout.strip().split(", ")
        used = int(used.split()[0])
        total = int(total.split()[0])
        print(f"VRAM Usage: {used} MiB / {total} MiB")
        return used, total
    except Exception as e:
        print(f"Error checking VRAM: {e}")
        return None, None

# Function to clear VRAM
def clear_vram():
    gc.collect()
    torch.cuda.empty_cache()
    print("Cleared VRAM cache.")

# Function to fine-tune Gemma with LoRA
def fine_tune_gemma(text_model, text_tokenizer):
    print("Starting fine-tuning Gemma with LoRA...")
    dataset = load_dataset("financial_phrasebank", "sentences_allagree", split="train", trust_remote_code=True)
    dataset = dataset.shuffle(seed=42).select(range(min(1000, len(dataset))))
    
    def preprocess_function(examples):
        inputs = [f"<s>[INST] Analyze the sentiment of this financial statement: {text} [/INST]" for text in examples["sentence"]]
        label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
        targets = [f"The sentiment is {label_map[label]}." for label in examples["label"]]
        combined_texts = [f"{inp} {tgt}" for inp, tgt in zip(inputs, targets)]
        model_inputs = text_tokenizer(
            combined_texts,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        model_inputs["labels"] = model_inputs["input_ids"].clone()
        for i, inp in enumerate(inputs):
            inp_tokens = text_tokenizer(inp, add_special_tokens=False, return_tensors="pt")["input_ids"].shape[1]
            model_inputs["labels"][i, :inp_tokens] = -100
        return model_inputs

    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=["sentence", "label"])
    train_dataset = tokenized_dataset.shuffle(seed=42).select(range(int(0.9 * len(tokenized_dataset))))
    eval_dataset = tokenized_dataset.shuffle(seed=42).select(range(int(0.9 * len(tokenized_dataset)), len(tokenized_dataset)))

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    text_model = get_peft_model(text_model, lora_config)
    training_args = TrainingArguments(
        output_dir=LORA_ADAPTER_PATH,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_steps=50,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=25,
        load_best_model_at_end=True,
        report_to="none",
        optim="adamw_8bit"
    )
    trainer = Trainer(
        model=text_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=text_tokenizer, mlm=False)
    )
    print("Training model...")
    trainer.train()
    text_model.save_pretrained(LORA_ADAPTER_PATH)
    print(f"LoRA adapter saved to {LORA_ADAPTER_PATH}")
    del trainer
    del text_model
    clear_vram()
    print("VRAM status after fine-tuning:")
    check_vram()

# Function to load or setup models
def setup_models(save_after_setup=False, fine_tune=False):
    print("Initial VRAM status:")
    check_vram()
    text_model_name = "google/gemma-2b-it"
    text_model, text_tokenizer = None, None
    if os.path.exists(TEXT_MODEL_PATH):
        print(f"Loading quantized Gemma model from {TEXT_MODEL_PATH}")
        text_model = AutoModelForCausalLM.from_pretrained(
            TEXT_MODEL_PATH,
            device_map="auto",
            torch_dtype=torch.float16,
            use_cache=True
        )
        text_tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_PATH)
    else:
        print("Quantizing and saving Gemma model with 4-bit quantization...")
        text_model = AutoModelForCausalLM.from_pretrained(
            text_model_name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
            use_cache=True
        )
        text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        if save_after_setup:
            text_model.save_pretrained(TEXT_MODEL_PATH)
            text_tokenizer.save_pretrained(TEXT_MODEL_PATH)
    
    print("VRAM status after loading Gemma:")
    check_vram()

    if fine_tune and not os.path.exists(os.path.join(LORA_ADAPTER_PATH, "adapter_config.json")):
        fine_tune_gemma(text_model, text_tokenizer)
        text_model = AutoModelForCausalLM.from_pretrained(
            TEXT_MODEL_PATH,
            device_map="auto",
            torch_dtype=torch.float16,
            use_cache=True
        )
        text_tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_PATH)

    if os.path.exists(os.path.join(LORA_ADAPTER_PATH, "adapter_config.json")):
        print(f"Loading fine-tuned LoRA adapter from {LORA_ADAPTER_PATH}")
        text_model = PeftModel.from_pretrained(text_model, LORA_ADAPTER_PATH)
    else:
        print("No fine-tuned LoRA adapter found. Running fine-tuning if requested.")

    print("VRAM status after loading LoRA adapter:")
    check_vram()

    if not save_after_setup:
        del text_model
        del text_tokenizer
        clear_vram()
        print("VRAM status after clearing Gemma:")
        check_vram()

    blip_model_name = "Salesforce/blip-image-captioning-base"
    blip_processor, blip_model = None, None
    if os.path.exists(BLIP_MODEL_PATH):
        print(f"Loading quantized BLIP model from {BLIP_MODEL_PATH}")
        blip_processor = BlipProcessor.from_pretrained(BLIP_MODEL_PATH)
        blip_model = BlipForConditionalGeneration.from_pretrained(
            BLIP_MODEL_PATH,
            torch_dtype=torch.float16
        )
    else:
        print("Quantizing and saving BLIP model with 4-bit quantization...")
        blip_processor = BlipProcessor.from_pretrained(blip_model_name)
        blip_model = BlipForConditionalGeneration.from_pretrained(
            blip_model_name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16
        )
        if save_after_setup:
            blip_processor.save_pretrained(BLIP_MODEL_PATH)
            blip_model.save_pretrained(BLIP_MODEL_PATH)

    print("VRAM status after loading BLIP:")
    check_vram()

    if not save_after_setup:
        del blip_model
        del blip_processor
        clear_vram()
        print("VRAM status after clearing BLIP:")
        check_vram()

    whisper_model_name = "openai/whisper-tiny"
    transcriber = None
    if os.path.exists(WHISPER_MODEL_PATH):
        print(f"Loading Whisper-tiny pipeline from {WHISPER_MODEL_PATH}")
        transcriber = pipeline(
            "automatic-speech-recognition",
            model=WHISPER_MODEL_PATH,
            device=0 if torch.cuda.is_available() else -1,
            model_kwargs={"use_cache": True}
        )
    else:
        print("Setting up and saving Whisper-tiny pipeline...")
        transcriber = pipeline(
            "automatic-speech-recognition",
            model=whisper_model_name,
            device=0 if torch.cuda.is_available() else -1,
            model_kwargs={"use_cache": True}
        )
        if save_after_setup:
            transcriber.save_pretrained(WHISPER_MODEL_PATH)

    print("VRAM status after loading Whisper-tiny:")
    check_vram()

    if not save_after_setup:
        del transcriber
        clear_vram()
        print("VRAM status after clearing Whisper-tiny:")
        check_vram()

    text_model, text_tokenizer = None, None
    blip_processor, blip_model = None, None
    transcriber = None
    return text_model, text_tokenizer, blip_model, blip_processor, transcriber

# Input Processing Functions
def process_text_input(text):
    return text

def process_image_input(image_path, blip_processor, blip_model):
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        image = Image.open(image_path)
        inputs = blip_processor(image, return_tensors="pt").to(device)
        out = blip_model.generate(**inputs, use_cache=True)
        caption = blip_processor.decode(out[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        raise ValueError(f"Failed to process image: {e}")

def process_audio_input(audio_path, transcriber):
    try:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        transcription = transcriber(audio_path)["text"]
        return transcription
    except Exception as e:
        raise ValueError(f"Failed to process audio: {e}")

# Chatbot Function
def chatbot(input_type, input_data, text_model, text_tokenizer, blip_model, blip_processor, transcriber):
    if input_type == "text":
        if text_model is None:
            print("Loading Gemma for text input...")
            text_model = AutoModelForCausalLM.from_pretrained(
                TEXT_MODEL_PATH,
                device_map="auto",
                torch_dtype=torch.float16,
                use_cache=True
            )
            text_tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_PATH)
            if os.path.exists(os.path.join(LORA_ADAPTER_PATH, "adapter_config.json")):
                text_model = PeftModel.from_pretrained(text_model, LORA_ADAPTER_PATH)
            print("VRAM status after loading Gemma for inference:")
            check_vram()
        processed_input = process_text_input(input_data)
        input_text = f"<s>[INST] Analyze the sentiment of this financial statement: {processed_input} [/INST]"
    elif input_type == "image":
        if blip_model is None:
            print("Loading BLIP for image input...")
            blip_processor = BlipProcessor.from_pretrained(BLIP_MODEL_PATH)
            blip_model = BlipForConditionalGeneration.from_pretrained(
                BLIP_MODEL_PATH,
                torch_dtype=torch.float16
            )
            print("VRAM status after loading BLIP for inference:")
            check_vram()
        processed_input = process_image_input(input_data, blip_processor, blip_model)
        del blip_model
        del blip_processor
        clear_vram()
        print("VRAM status after clearing BLIP:")
        check_vram()
        blip_model, blip_processor = None, None
        input_text = f"<s>[INST] {processed_input} [/INST]"
    elif input_type == "audio":
        if transcriber is None:
            print("Loading Whisper-tiny for audio input...")
            transcriber = pipeline(
                "automatic-speech-recognition",
                model=WHISPER_MODEL_PATH,
                device=0 if torch.cuda.is_available() else -1,
                model_kwargs={"use_cache": True}
            )
            print("VRAM status after loading Whisper-tiny for inference:")
            check_vram()
        processed_input = process_audio_input(input_data, transcriber)
        del transcriber
        clear_vram()
        print("VRAM status after clearing Whisper-tiny:")
        check_vram()
        transcriber = None
        input_text = f"<s>[INST] {processed_input} [/INST]"
    else:
        raise ValueError("Invalid input type. Must be 'text', 'image', or 'audio'.")

    if text_model is None:
        print("Loading Gemma for final response generation...")
        text_model = AutoModelForCausalLM.from_pretrained(
            TEXT_MODEL_PATH,
            device_map="auto",
            torch_dtype=torch.float16,
            use_cache=True
        )
        text_tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_PATH)
        if os.path.exists(os.path.join(LORA_ADAPTER_PATH, "adapter_config.json")):
            text_model = PeftModel.from_pretrained(text_model, LORA_ADAPTER_PATH)
        print("VRAM status after loading Gemma for response:")
        check_vram()

    inputs = text_tokenizer(input_text, return_tensors="pt").to(device)
    
    try:
        outputs = text_model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            temperature=0.5,
            top_p=0.95,
            use_cache=True
        )
        response = text_tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Raw response before cleaning (input_type={input_type}): {response}")
        response = (response
                    .replace("<s>", "")
                    .replace("[INST]", "")
                    .replace("[/INST]", "")
                    .replace("~~", "")
                    .replace("~", "")
                    .replace("*", "")
                    .replace("_", "")
                    .replace("`", "")
                    .replace("#", "")
                    .replace("-", "")
                    .replace(">", "")
                    .strip())
        if input_type == "text":
            prompt_prefix = f"Analyze the sentiment of this financial statement: {processed_input}"
            if prompt_prefix in response:
                response = response.split(prompt_prefix, 1)[1].strip()
        print(f"Sanitized response (input_type={input_type}): {response}")
        del text_model
        del text_tokenizer
        clear_vram()
        print("VRAM status after clearing Gemma:")
        check_vram()
        return response
    except Exception as e:
        raise RuntimeError(f"Failed to generate response: {e}")

# Main Execution
if __name__ == "__main__":
    save_after_setup = not (os.path.exists(TEXT_MODEL_PATH) and os.path.exists(BLIP_MODEL_PATH) and os.path.exists(WHISPER_MODEL_PATH))
    text_model, text_tokenizer, blip_model, blip_processor, transcriber = setup_models(save_after_setup=save_after_setup, fine_tune=True)
    try:
        print("Text Response:", chatbot("text", "Analyze the sentiment of this financial statement: 'The company reported a strong profit increase.'", 
                                     text_model, text_tokenizer, blip_model, blip_processor, transcriber))
        # Image test requires a local file path now
        # print("Image Response:", chatbot("image", "path/to/local/image.png", 
        #                              text_model, text_tokenizer, blip_model, blip_processor, transcriber))
        # audio_path = "path/to/audio/file.mp3"
        # print("Audio Response:", chatbot("audio", audio_path, 
        #                              text_model, text_tokenizer, blip_model, blip_processor, transcriber))
    except Exception as e:
        print(f"Error during execution: {e}")