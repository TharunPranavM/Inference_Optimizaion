import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import bleach
from datetime import datetime

# FastAPI server URL
API_URL = "http://localhost:8000"

# Streamlit app configuration
st.set_page_config(
    page_title="Multimodal Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .chat-message { 
        padding: 10px; 
        border-radius: 10px; 
        margin: 5px 0; 
        max-width: 80%; 
        word-wrap: break-word; 
        color: black;
    }
    .user-message { 
        background-color: #d1e7dd; 
        margin-left: auto; 
        text-align: right; 
        color: black;
    }
    .bot-message { 
        background-color: #e9ecef; 
        margin-right: auto; 
        color: black;
    }
    .stButton>button { 
        background-color: #007bff; 
        color: white; 
        border-radius: 5px; 
    }
    .stButton>button:hover { 
        background-color: #0056b3; 
    }
    .sidebar .sidebar-content { 
        background-color: #ffffff; 
    }
    .badge { 
        padding: 5px 10px; 
        border-radius: 12px; 
        font-size: 12px; 
        font-weight: bold; 
    }
    .text-badge { 
        background-color: #28a745; 
        color: white; 
    }
    .image-badge { 
        background-color: #ffc107; 
        color: black; 
    }
    .audio-badge { 
        background-color: #dc3545; 
        color: white; 
    }
    .xai-logo { 
        font-size: 24px; 
        font-weight: bold; 
        color: #007bff; 
        text-align: center; 
        margin-bottom: 10px; 
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = {
        "text": [],
        "image": [],
        "audio": []
    }

# Sidebar for navigation and settings
with st.sidebar:
    st.title("Multimodal Chatbot")
    st.markdown("Interact with text, images, or audio!")
    
    input_mode = st.selectbox(
        "Select Input Mode",
        ["Text", "Image", "Audio"],
        help="Choose how you want to interact with the chatbot."
    )
    
    st.markdown("---")
    st.markdown("**Settings**")
    clear_history = st.button("Clear Chat History")
    if clear_history:
        st.session_state.chat_history = {"text": [], "image": [], "audio": []}
        st.success("Chat history cleared!")
    
    st.markdown("---")
    st.markdown("**Note**: Ensure the FastAPI server is running at `http://localhost:8000`.")

# Function to display chat history
def display_chat_history(history, input_type):
    for msg in history:
        if msg["role"] == "user":
            # For text input, display the user message as text
            if input_type == "text":
                st.markdown(
                    f'<div class="chat-message user-message"><span class="badge {input_type}-badge">{input_type.capitalize()}</span><br>{msg["content"]}</div>',
                    unsafe_allow_html=True
                )
            # For image input, display the image preview
            elif input_type == "image" and "image_data" in msg:
                st.markdown(
                    f'<div class="chat-message user-message"><span class="badge {input_type}-badge">{input_type.capitalize()}</span><br></div>',
                    unsafe_allow_html=True
                )
                st.image(msg["image_data"], width=150)
            # For audio input, display the audio player
            elif input_type == "audio" and "audio_data" in msg:
                st.markdown(
                    f'<div class="chat-message user-message"><span class="badge {input_type}-badge">{input_type.capitalize()}</span><br></div>',
                    unsafe_allow_html=True
                )
                st.audio(msg["audio_data"], format=msg["audio_format"])
        else:
            # Display bot message with fallback sanitization
            cleaned_response = bleach.clean(msg["content"], tags=["div", "span", "br"], strip=True)
            try:
                st.markdown(
                    f'<div class="chat-message bot-message"><span class="badge {input_type}-badge">Bot</span><br>{cleaned_response}</div>',
                    unsafe_allow_html=True
                )
            except Exception as e:
                st.error(f"Error displaying response: {str(e)}. Please try again.")

# Main content
st.title("🤖 Multimodal Chatbot")
st.markdown("Engage with the chatbot using text, images, or audio. Your conversations are displayed in a chat-like format.")

# Map input mode to section
if input_mode == "Text":
    with st.container():
        st.header("📝 Text Input")
        st.write("Enter a financial statement or query for sentiment analysis.")
        
        # Input form above the conversation
        with st.form(key="text_form"):
            text_input = st.text_area(
                "Your Message",
                placeholder="e.g., The company reported a strong profit increase.",
                height=100
            )
            submit_text = st.form_submit_button("Send")
        
        if submit_text and text_input.strip():
            with st.spinner("Processing..."):
                try:
                    response = requests.post(f"{API_URL}/text", json={"text": text_input}, timeout=30)
                    response.raise_for_status()
                    result = response.json()
                    # Add to chat history
                    st.session_state.chat_history["text"].append({"role": "user", "content": text_input, "timestamp": datetime.now()})
                    st.session_state.chat_history["text"].append({"role": "bot", "content": result["response"], "timestamp": datetime.now()})
                except requests.exceptions.RequestException as e:
                    st.error(f"Error: {str(e)}")
                    st.button("Retry", key="retry_text")
        
        # Display chat history below the input
        st.subheader("Conversation")
        display_chat_history(st.session_state.chat_history["text"], "text")

elif input_mode == "Image":
    with st.container():
        st.header("🖼️ Image Input")
        st.write("Upload an image for the chatbot to describe and respond to.")
        
        # File uploader for image
        image_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
        
        # Image preview
        image_data = None
        if image_file:
            image_data = Image.open(image_file)
            st.image(image_data, caption="Uploaded Image", width=300)
        
        with st.form(key="image_form"):
            submit_image = st.form_submit_button("Send")
        
        if submit_image and image_file is not None:
            with st.spinner("Processing..."):
                try:
                    response = requests.post(
                        f"{API_URL}/image",
                        files={"image_file": (image_file.name, image_file.getvalue(), image_file.type)},
                        timeout=30
                    )
                    response.raise_for_status()
                    result = response.json()
                    # Add to chat history
                    st.session_state.chat_history["image"].append({
                        "role": "user",
                        "image_data": image_data,
                        "timestamp": datetime.now()
                    })
                    st.session_state.chat_history["image"].append({
                        "role": "bot",
                        "content": result["response"],
                        "timestamp": datetime.now()
                    })
                except requests.exceptions.RequestException as e:
                    st.error(f"Error: {str(e)}")
                    st.button("Retry", key="retry_image")
        elif submit_image:
            st.warning("Please upload an image file.")
        
        # Display chat history
        st.subheader("Conversation")
        display_chat_history(st.session_state.chat_history["image"], "image")

elif input_mode == "Audio":
    with st.container():
        st.header("🎙️ Audio Input")
        st.write("Upload an audio file for the chatbot to transcribe and respond to.")
        st.warning("Ensure FFmpeg is installed and added to your PATH for audio processing.")
        
        audio_file = st.file_uploader("Upload Audio", type=["mp3", "wav"])
        
        # Audio playback
        audio_data = None
        audio_format = None
        if audio_file:
            audio_data = audio_file.getvalue()
            audio_format = audio_file.type
            st.audio(audio_data, format=audio_format)
        
        with st.form(key="audio_form"):
            submit_audio = st.form_submit_button("Send")
        
        if submit_audio and audio_file is not None:
            with st.spinner("Processing..."):
                try:
                    response = requests.post(
                        f"{API_URL}/audio",
                        files={"audio_file": (audio_file.name, audio_file.getvalue(), audio_file.type)},
                        timeout=30
                    )
                    response.raise_for_status()
                    result = response.json()
                    # Add to chat history
                    st.session_state.chat_history["audio"].append({
                        "role": "user",
                        "audio_data": audio_data,
                        "audio_format": audio_format,
                        "timestamp": datetime.now()
                    })
                    st.session_state.chat_history["audio"].append({
                        "role": "bot",
                        "content": result["response"],
                        "timestamp": datetime.now()
                    })
                except requests.exceptions.RequestException as e:
                    st.error(f"Error: {str(e)}")
                    st.button("Retry", key="retry_audio")
        elif submit_audio:
            st.warning("Please upload an audio file.")
        
        # Display chat history
        st.subheader("Conversation")
        display_chat_history(st.session_state.chat_history["audio"], "audio")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center;'>
        Built with <b>Streamlit</b> and <b>FastAPI</b>. Inference optimized multimodal AI models.
    </div>
""", unsafe_allow_html=True)