import streamlit as st
import torch
import re
import scipy.io.wavfile
import gc
import numpy as np
import os
from typing import Optional, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, BarkModel
import tempfile
from io import BytesIO
import base64

# Set page config
st.set_page_config(
    page_title="🎬 EchoVerse",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for semi-transparent UI with custom wallpaper
def get_custom_css():
    return """
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;500;600;700&display=swap');
    
    /* Main app background with custom wallpaper */
    .stApp {
        background: linear-gradient(rgba(0, 0, 0, 0.7), rgba(0, 0, 0, 0.8)), 
                    url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1920 1080"><defs><radialGradient id="grad1" cx="30%" cy="40%"><stop offset="0%" style="stop-color:%23ff6b6b;stop-opacity:0.3"/><stop offset="50%" style="stop-color:%234ecdc4;stop-opacity:0.2"/><stop offset="100%" style="stop-color:%23000;stop-opacity:1"/></radialGradient></defs><rect width="1920" height="1080" fill="url(%23grad1)"/><g fill="none" stroke="%23333" stroke-width="1" opacity="0.3"><line x1="0" y1="200" x2="1920" y2="200"/><line x1="0" y1="400" x2="1920" y2="400"/><line x1="0" y1="600" x2="1920" y2="600"/><line x1="0" y1="800" x2="1920" y2="800"/><line x1="200" y1="0" x2="200" y2="1080"/><line x1="400" y1="0" x2="400" y2="1080"/><line x1="600" y1="0" x2="600" y2="1080"/><line x1="800" y1="0" x2="800" y2="1080"/><line x1="1000" y1="0" x2="1000" y2="1080"/><line x1="1200" y1="0" x2="1200" y2="1080"/><line x1="1400" y1="0" x2="1400" y2="1080"/><line x1="1600" y1="0" x2="1600" y2="1080"/></g></svg>') fixed;
        background-size: cover;
        background-position: -200px center;
        background-repeat: no-repeat;
    }
    
    /* Semi-transparent containers */
    .main .block-container {
        background: rgba(15, 15, 30, 0.85);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 2px solid rgba(76, 175, 80, 0.4);
        box-shadow: 0 12px 40px 0 rgba(31, 38, 135, 0.5);
        padding: 2.5rem;
        margin: 1.5rem 0;
        max-width: 1400px;
        margin-left: auto;
        margin-right: auto;
        padding-bottom: 140px !important;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(20, 20, 40, 0.95) !important;
        backdrop-filter: blur(20px);
        border-right: 3px solid rgba(76, 175, 80, 0.5);
        min-width: 300px !important;
        padding: 1.5rem !important;
    }
    
    /* Title styling */
    h1 {
        color: #4CAF50 !important;
        font-family: 'Orbitron', monospace !important;
        font-weight: 900 !important;
        text-align: center !important;
        text-shadow: 0 0 30px rgba(76, 175, 80, 0.7) !important;
        margin-bottom: 2.5rem !important;
        font-size: 3.2rem !important;
        letter-spacing: 2px !important;
    }
    
    /* Headers styling */
    h2, h3 {
        color: #81C784 !important;
        font-family: 'Rajdhani', sans-serif !important;
        font-weight: 700 !important;
        text-shadow: 0 0 15px rgba(129, 199, 132, 0.4) !important;
        margin-bottom: 1.5rem !important;
        font-size: 1.8rem !important;
    }
    
    /* Sidebar headers */
    .css-1d391kg h2, .css-1d391kg h3 {
        font-size: 1.4rem !important;
        margin-bottom: 1rem !important;
        padding-bottom: 0.5rem !important;
        border-bottom: 1px solid rgba(76, 175, 80, 0.3) !important;
    }
    
    /* Text styling */
    p, .stMarkdown, .stText {
        color: #E8F5E8 !important;
        font-family: 'Rajdhani', sans-serif !important;
        font-weight: 400 !important;
        font-size: 1.1rem !important;
        line-height: 1.6 !important;
    }
    
    /* Input fields */
    .stTextArea textarea, .stSelectbox select, .stTextInput input {
        background: rgba(30, 30, 60, 0.9) !important;
        color: #E8F5E8 !important;
        border: 2px solid rgba(76, 175, 80, 0.6) !important;
        border-radius: 12px !important;
        backdrop-filter: blur(8px) !important;
        padding: 1rem !important;
        font-size: 1rem !important;
        font-family: 'Rajdhani', sans-serif !important;
        min-height: 45px !important;
    }
    
    /* Text areas specific sizing */
    .stTextArea textarea {
        min-height: 200px !important;
        resize: vertical !important;
    }
    
    /* Select boxes */
    .stSelectbox select {
        padding: 0.8rem 1rem !important;
        font-weight: 500 !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(45deg, #4CAF50, #45a049) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-family: 'Rajdhani', sans-serif !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        padding: 0.8rem 1.5rem !important;
        min-height: 50px !important;
        box-shadow: 0 6px 20px 0 rgba(76, 175, 80, 0.4) !important;
        transition: all 0.3s ease !important;
        letter-spacing: 0.5px !important;
        width: 100% !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(45deg, #45a049, #4CAF50) !important;
        box-shadow: 0 8px 25px 0 rgba(76, 175, 80, 0.6) !important;
        transform: translateY(-3px) !important;
    }
    
    /* Primary button */
    .stButton > button[kind="primary"] {
        background: linear-gradient(45deg, #FF5722, #FF7043) !important;
        box-shadow: 0 6px 20px 0 rgba(255, 87, 34, 0.4) !important;
        min-height: 60px !important;
        font-size: 1.3rem !important;
        padding: 1rem 2rem !important;
        font-weight: 700 !important;
    }
    
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(45deg, #FF7043, #FF5722) !important;
        box-shadow: 0 8px 25px 0 rgba(255, 87, 34, 0.6) !important;
    }
    
    /* Progress bars */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #4CAF50, #81C784) !important;
    }
    
    /* Success/Error messages */
    .stSuccess, .stInfo, .stWarning, .stError {
        background: rgba(30, 30, 60, 0.8) !important;
        backdrop-filter: blur(10px) !important;
        border-radius: 10px !important;
        border-left: 4px solid #4CAF50 !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(30, 30, 60, 0.9) !important;
        border-radius: 12px !important;
        color: #E8F5E8 !important;
        font-family: 'Rajdhani', sans-serif !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        padding: 1rem 1.5rem !important;
        border: 1px solid rgba(76, 175, 80, 0.4) !important;
    }
    
    .streamlit-expanderContent {
        background: rgba(20, 20, 40, 0.95) !important;
        backdrop-filter: blur(15px) !important;
        border-radius: 0 0 12px 12px !important;
        padding: 1.5rem !important;
        border: 1px solid rgba(76, 175, 80, 0.3) !important;
        border-top: none !important;
    }
    
    /* Checkbox */
    .stCheckbox label {
        color: #E8F5E8 !important;
        font-family: 'Rajdhani', sans-serif !important;
        font-size: 1.1rem !important;
        font-weight: 500 !important;
    }
    
    /* MAIN COLUMN ALIGNMENT FIXES */
    [data-testid="column"] {
        padding: 0 1rem !important;
        height: 100% !important;
        display: flex !important;
        flex-direction: column !important;
    }
    
    [data-testid="column"]:first-child {
        padding-left: 0 !important;
        padding-right: 1.5rem !important;
    }
    
    [data-testid="column"]:last-child {
        padding-right: 0 !important;
        padding-left: 1.5rem !important;
    }
    
    /* Remove column content containers - no boxes */
    .column-section {
        padding: 1.5rem 0 !important;
        flex-grow: 1 !important;
        display: flex !important;
        flex-direction: column !important;
    }
    
    /* Button container alignment */
    .button-row {
        display: flex !important;
        gap: 1rem !important;
        margin: 1rem 0 !important;
    }
    
    .button-row .stButton {
        flex: 1 !important;
    }
    
    /* Info boxes */
    .stInfo, .stSuccess, .stWarning, .stError {
        padding: 1rem 1.5rem !important;
        border-radius: 12px !important;
        font-size: 1rem !important;
        margin: 1rem 0 !important;
    }
    
    /* Team info footer */
    .team-footer {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: linear-gradient(135deg, rgba(76, 175, 80, 0.98), rgba(67, 160, 71, 0.98));
        backdrop-filter: blur(25px);
        border-top: 3px solid rgba(129, 199, 132, 0.6);
        padding: 20px 0;
        text-align: center;
        z-index: 1000;
        box-shadow: 0 -6px 30px rgba(0, 0, 0, 0.4);
        min-height: 80px;
    }
    
    .team-name {
        font-family: 'Orbitron', monospace;
        font-size: 28px;
        font-weight: 900;
        color: #FFFFFF;
        text-shadow: 0 0 20px rgba(255, 255, 255, 0.6);
        margin-bottom: 10px;
        letter-spacing: 3px;
    }
    
    .team-members {
        font-family: 'Rajdhani', sans-serif;
        font-size: 18px;
        font-weight: 600;
        color: #F1F8E9;
        text-shadow: 0 0 12px rgba(241, 248, 233, 0.4);
        letter-spacing: 1px;
    }
    
    .team-lead {
        color: #FFD54F;
        font-weight: 700;
        text-shadow: 0 0 15px rgba(255, 213, 79, 0.5);
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(30, 30, 60, 0.5);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(45deg, #4CAF50, #81C784);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(45deg, #81C784, #4CAF50);
    }
    
    /* Audio player styling */
    .stAudio {
        background: rgba(30, 30, 60, 0.9) !important;
        border-radius: 15px !important;
        padding: 1.5rem !important;
        border: 2px solid rgba(76, 175, 80, 0.4) !important;
        margin: 1.5rem 0 !important;
        box-shadow: 0 8px 25px rgba(76, 175, 80, 0.2) !important;
    }
    
    /* Download button special styling */
    .stDownloadButton > button {
        background: linear-gradient(45deg, #2196F3, #21CBF3) !important;
        box-shadow: 0 6px 20px 0 rgba(33, 150, 243, 0.4) !important;
        min-height: 55px !important;
        font-size: 1.2rem !important;
        padding: 1rem 2rem !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
    }
    
    .stDownloadButton > button:hover {
        background: linear-gradient(45deg, #21CBF3, #2196F3) !important;
        box-shadow: 0 8px 25px 0 rgba(33, 150, 243, 0.6) !important;
        transform: translateY(-3px) !important;
    }
    
    /* Spinner customization */
    .stSpinner > div {
        border-top-color: #4CAF50 !important;
    }
    
    /* Labels and small text */
    .stSelectbox label, .stTextArea label, .stTextInput label {
        color: #81C784 !important;
        font-family: 'Rajdhani', sans-serif !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Ensure equal height containers */
    .element-container {
        height: 100% !important;
    }
    
    .element-container > div {
        height: 100% !important;
    }
    </style>
    """

# Add team footer
def add_team_footer():
    st.markdown("""
    <div class="team-footer">
        <div class="team-name">SHADOW PROTOCOL</div>
        <div class="team-members">
            Team Lead: <span class="team-lead">Yashwanth Reddy</span> | 
            Co-Developers: <strong>Venu Nakirthi</strong> & <strong>Akhilesh Yadav</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ---------------- Text Enhancer ----------------
class TextEnhancer:
    def __init__(self):
        self.granite_model = None
        self.granite_tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def enhance_text_for_tone(self, text: str, tone: str) -> str:
        tone_enhancements = {
            "Neutral": {"replacements": {}},
            "Suspenseful": {
                "replacements": {
                    r'\bsaid\b': 'whispered', 
                    r'\bwalked\b': 'crept',
                    r'\bmoved\b': 'stalked',
                    r'\bquiet\b': 'eerily silent'
                }
            },
            "Inspiring": {
                "replacements": {
                    r'\bcan\b': 'CAN', 
                    r'\bwill\b': 'WILL',
                    r'\bgood\b': 'AMAZING',
                    r'\bpossible\b': 'absolutely possible'
                }
            },
            "Conversational": {
                "replacements": {
                    r'\. ([A-Z])': r'. Well, \1',
                    r'\bI think\b': 'You know what I think',
                    r'\bHowever\b': 'But here\'s the thing'
                }
            },
            "Educational": {
                "replacements": {
                    r'\bshow\b': 'demonstrate', 
                    r'\btell\b': 'explain',
                    r'\bimportant\b': 'crucial to understand'
                }
            }
        }
        enhancement = tone_enhancements.get(tone, tone_enhancements["Neutral"])
        enhanced_text = text
        for pattern, replacement in enhancement["replacements"].items():
            enhanced_text = re.sub(pattern, replacement, enhanced_text, flags=re.IGNORECASE)
        return enhanced_text

    def rewrite_text_with_advanced_tone(self, text: str, tone: str) -> Optional[str]:
        if not self.granite_model or not self.granite_tokenizer:
            return text

        # Simplified prompts to reduce processing load
        prompts = {
            "Neutral": f"Rewrite clearly: {text[:200]}",  # Limit input length
            "Suspenseful": f"Make mysterious: {text[:200]}",
            "Inspiring": f"Make motivational: {text[:200]}",
            "Conversational": f"Make conversational: {text[:200]}",
            "Educational": f"Explain clearly: {text[:200]}"
        }
        
        prompt = prompts.get(tone, f"Rewrite: {text[:200]}")

        try:
            inputs = self.granite_tokenizer.encode(prompt, return_tensors="pt", max_length=256, truncation=True).to(self.device)
            
            with torch.no_grad():
                outputs = self.granite_model.generate(
                    inputs,
                    max_new_tokens=150,  # Reduced from 500
                    temperature=0.7,    # Reduced for stability
                    do_sample=True,
                    top_p=0.8,
                    repetition_penalty=1.05,
                    pad_token_id=self.granite_tokenizer.eos_token_id,
                    early_stopping=True
                )
            
            generated_text = self.granite_tokenizer.decode(outputs[0], skip_special_tokens=True)
            rewritten = generated_text.replace(prompt, "").strip()
            
            # Clean up memory
            del inputs, outputs
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            return rewritten if len(rewritten) > 10 else text
        except Exception as e:
            st.error(f"Text rewriting failed: {e}")
            # Fallback to basic enhancement
            return self.enhance_text_for_tone(text, tone)

    def load_granite_model(self):
        try:
            # Use smaller, more stable model for Streamlit Cloud
            model_name = "microsoft/DialoGPT-medium"
            self.granite_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.granite_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,  # Use float32 for stability
                device_map=None,  # Avoid auto device mapping on cloud
                low_cpu_mem_usage=True
            ).to(self.device)
            if self.granite_tokenizer.pad_token is None:
                self.granite_tokenizer.pad_token = self.granite_tokenizer.eos_token
            return True
        except Exception as e:
            st.warning(f"Advanced model failed to load: {e}")
            return self._load_fallback_model()

    def _load_fallback_model(self):
        try:
            # Even smaller fallback model
            model_name = "gpt2"
            self.granite_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.granite_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            ).to(self.device)
            self.granite_tokenizer.pad_token = self.granite_tokenizer.eos_token
            return True
        except Exception as e:
            st.error(f"Failed to load any text model: {e}")
            return False

    def initialize_models(self):
        return self.load_granite_model()

# ---------------- Bark TTS ----------------
class BarkTTS:
    def __init__(self):
        self.device = "cpu"  # Force CPU for Streamlit Cloud stability
        self.processor = None
        self.model = None

    def load_model(self):
        """Load Bark model with better error handling"""
        if self.model is not None:
            return True
        
        try:
            # Load processor first
            if self.processor is None:
                self.processor = AutoProcessor.from_pretrained("suno/bark")
            
            # Load model with conservative settings for cloud deployment
            self.model = BarkModel.from_pretrained(
                "suno/bark", 
                torch_dtype=torch.float32,  # Use float32 for stability
                low_cpu_mem_usage=True
            )
            self.model.to(self.device)
            
            # Enable eval mode to save memory
            self.model.eval()
            return True
            
        except Exception as e:
            st.error(f"Failed to load Bark model: {e}")
            # Clean up on failure
            self.model = None
            self.processor = None
            gc.collect()
            return False

    def convert_audio_format(self, audio_array: np.ndarray) -> np.ndarray:
        """Convert audio to a format compatible with scipy.io.wavfile.write"""
        if hasattr(audio_array, 'numpy'):
            audio_array = audio_array.numpy()
        
        if not isinstance(audio_array, np.ndarray):
            audio_array = np.array(audio_array)
        
        audio_array = audio_array.squeeze()
        
        # Convert to float32 if it's float16 or other unsupported types
        if audio_array.dtype in [np.float16, np.float64]:
            audio_array = audio_array.astype(np.float32)
        
        # Normalize audio to prevent clipping
        if audio_array.dtype == np.float32:
            max_val = np.max(np.abs(audio_array))
            if max_val > 0:
                audio_array = audio_array / max_val
            
            # Convert to 16-bit PCM for better compatibility
            audio_array = (audio_array * 32767).astype(np.int16)
        
        return audio_array

    def generate_audio(self, text: str, preset: str, mood: str) -> tuple:
        """Generate audio with improved error handling and memory management"""
        if not self.load_model():
            return None, None
        
        # Limit text length for stability
        if len(text) > 200:
            text = text[:200] + "..."
            st.warning("Text truncated to 200 characters for stability")
        
        # Enhanced mood mapping (simplified)
        mood_mapping = {
            "neutral": "",
            "suspenseful": "[whispers]",
            "inspiring": "[excited]",
            "conversational": "[friendly]",
            "educational": "[clear]",
            "happy": "[happy]",
            "sad": "[sad]",
            "angry": "[angry]",
            "surprised": "[surprised]",
            "scared": "[scared]",
            "excited": "[excited]",
            "calm": "[calm]"
        }
        
        mood_tag = mood_mapping.get(mood.lower(), "")
        styled_text = f"{mood_tag} {text}" if mood_tag else text
        
        try:
            inputs = self.processor(styled_text, voice_preset=preset, return_tensors="pt")
            
            # Move inputs to device
            for k, v in inputs.items():
                if hasattr(v, 'to'):
                    inputs[k] = v.to(self.device)

            with torch.no_grad():
                audio_array = self.model.generate(**inputs)

            # Convert to proper format
            audio_array = audio_array.cpu().numpy()
            audio_array = self.convert_audio_format(audio_array)
            
            # Clean up
            del inputs
            gc.collect()
            
            sample_rate = self.model.generation_config.sample_rate
            return audio_array, sample_rate
            
        except Exception as e:
            st.error(f"Audio generation failed: {e}")
            # Clean up on error
            gc.collect()
            return None, None

# ---------------- Streamlit App ----------------
def initialize_session_state():
    """Initialize session state variables"""
    if 'text_enhancer' not in st.session_state:
        st.session_state.text_enhancer = None
    if 'bark_tts' not in st.session_state:
        st.session_state.bark_tts = None
    if 'enhanced_text' not in st.session_state:
        st.session_state.enhanced_text = ""
    if 'original_text' not in st.session_state:
        st.session_state.original_text = ""
    if 'models_loaded' not in st.session_state:
        st.session_state.models_loaded = False

def load_models():
    """Load models with progress tracking"""
    if not st.session_state.models_loaded:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Load Text Enhancer
        status_text.text("Loading Text Enhancement Model...")
        progress_bar.progress(25)
        st.session_state.text_enhancer = TextEnhancer()
        
        status_text.text("Initializing Text Enhancement...")
        progress_bar.progress(50)
        if st.session_state.text_enhancer.initialize_models():
            st.success("✅ Text Enhancement Model loaded!")
        else:
            st.warning("⚠️ Text Enhancement Model failed to load, using basic enhancement")
        
        # Load Bark TTS
        status_text.text("Loading Bark TTS Model...")
        progress_bar.progress(75)
        st.session_state.bark_tts = BarkTTS()
        
        status_text.text("Model loading complete!")
        progress_bar.progress(100)
        st.session_state.models_loaded = True
        
        status_text.empty()
        progress_bar.empty()

def main():
    # Apply custom CSS
    st.markdown(get_custom_css(), unsafe_allow_html=True)
    
    st.title("🎬 EchoVerse")
    st.markdown("Transform your text into engaging audiobooks with emotion and tone!")
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Voice selection
        st.subheader("🎙️ Voice Settings")
        voice_options = [f"v2/en_speaker_{i}" for i in range(10)]
        voice_descriptions = {
            "v2/en_speaker_0": "Speaker 0 - Male, Clear",
            "v2/en_speaker_1": "Speaker 1 - Female, Warm", 
            "v2/en_speaker_2": "Speaker 2 - Male, Deep",
            "v2/en_speaker_3": "Speaker 3 - Female, Energetic",
            "v2/en_speaker_4": "Speaker 4 - Male, Calm",
            "v2/en_speaker_5": "Speaker 5 - Female, Professional",
            "v2/en_speaker_6": "Speaker 6 - Male, Narrative",
            "v2/en_speaker_7": "Speaker 7 - Female, Expressive",
            "v2/en_speaker_8": "Speaker 8 - Male, Authoritative",
            "v2/en_speaker_9": "Speaker 9 - Female, Gentle"
        }
        
        selected_voice = st.selectbox(
            "Choose Voice",
            voice_options,
            index=6,
            format_func=lambda x: voice_descriptions.get(x, x)
        )
        
        # Tone selection
        st.subheader("🎨 Tone Settings")
        tone_options = ["Neutral", "Suspenseful", "Inspiring", "Conversational", "Educational"]
        tone_descriptions = {
            "Neutral": "📝 Professional and clear",
            "Suspenseful": "🕵️ Mysterious and intriguing",
            "Inspiring": "⭐ Motivational and uplifting",
            "Conversational": "💬 Friendly and casual",
            "Educational": "🎓 Instructive and informative"
        }
        
        selected_tone = st.selectbox(
            "Choose Tone",
            tone_options,
            format_func=lambda x: tone_descriptions.get(x, x)
        )
        
        # Additional mood options
        st.subheader("🎭 Emotion Settings")
        emotion_options = ["neutral", "happy", "sad", "excited", "calm", "surprised", "angry", "scared"]
        selected_emotion = st.selectbox("Choose Emotion", emotion_options)
        
        # Load models button
        if st.button("🔄 Load Models", type="primary"):
            load_models()
    
    # Main content with proper alignment
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Clean section without unnecessary containers
        st.markdown('<div class="column-section">', unsafe_allow_html=True)
        
        st.header("📝 Text Input & Enhancement")
        
        # Text input
        user_text = st.text_area(
            "Enter your text:",
            height=200,
            placeholder="Enter the text you want to convert to audiobook...",
            key="text_input"
        )
        
        # Text processing buttons
        col1a, col1b = st.columns(2)
        
        with col1a:
            if st.button("✨ Enhance Text Only", disabled=not user_text):
                if st.session_state.text_enhancer is None:
                    st.error("Please load models first!")
                else:
                    with st.spinner("Enhancing text..."):
                        enhanced = st.session_state.text_enhancer.enhance_text_for_tone(user_text, selected_tone)
                        st.session_state.enhanced_text = enhanced
                        st.session_state.original_text = user_text
                        st.success("Text enhanced!")
        
        with col1b:
            if st.button("🚀 AI Rewrite Text", disabled=not user_text):
                if st.session_state.text_enhancer is None:
                    st.error("Please load models first!")
                else:
                    with st.spinner("Rewriting text with AI..."):
                        rewritten = st.session_state.text_enhancer.rewrite_text_with_advanced_tone(user_text, selected_tone)
                        enhanced = st.session_state.text_enhancer.enhance_text_for_tone(rewritten, selected_tone)
                        st.session_state.enhanced_text = enhanced
                        st.session_state.original_text = user_text
                        st.success("Text rewritten and enhanced!")
        
        # Display enhanced text
        if st.session_state.enhanced_text:
            st.subheader("📖 Enhanced Text")
            st.text_area(
                "Enhanced version:",
                value=st.session_state.enhanced_text,
                height=150,
                key="enhanced_display"
            )
            
            # Option to edit enhanced text
            if st.checkbox("✏️ Edit enhanced text"):
                edited_text = st.text_area(
                    "Edit the enhanced text:",
                    value=st.session_state.enhanced_text,
                    height=150,
                    key="edit_enhanced"
                )
                if st.button("💾 Save Edits"):
                    st.session_state.enhanced_text = edited_text
                    st.success("Edits saved!")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # Clean section without unnecessary containers
        st.markdown('<div class="column-section">', unsafe_allow_html=True)
        
        st.header("🎧 Audio Generation")
        
        # Audio generation section
        text_to_convert = st.session_state.enhanced_text if st.session_state.enhanced_text else user_text
        
        if text_to_convert:
            st.subheader("🎵 Generate Audio")
            
            # Display selected settings
            with st.expander("📋 Current Settings", expanded=False):
                st.write(f"**Voice:** {voice_descriptions.get(selected_voice, selected_voice)}")
                st.write(f"**Tone:** {tone_descriptions.get(selected_tone, selected_tone)}")
                st.write(f"**Emotion:** {selected_emotion.title()}")
                st.write(f"**Text Length:** {len(text_to_convert)} characters")
            
            # Warning for long texts
            if len(text_to_convert) > 200:
                st.warning("⚠️ Text will be truncated to 200 characters for Streamlit Cloud stability")
            
            # Simplified chunking (removed for cloud stability)
            st.info("💡 For best results on Streamlit Cloud, keep text under 200 characters")
            
            # Generate audio button
            if st.button("🎬 Generate Audiobook", type="primary"):
                if st.session_state.bark_tts is None:
                    st.error("Please load models first!")
                else:
                    with st.spinner("Generating audio... Please wait..."):
                        try:
                            # Single processing only for cloud stability
                            final_audio, sample_rate = st.session_state.bark_tts.generate_audio(
                                text_to_convert, selected_voice, selected_emotion
                            )
                            
                            if final_audio is not None:
                                # Save to temporary file
                                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                                    scipy.io.wavfile.write(tmp_file.name, sample_rate, final_audio)
                                    
                                    # Read the file for download
                                    with open(tmp_file.name, "rb") as audio_file:
                                        audio_bytes = audio_file.read()
                                
                                # Display audio player
                                st.success("🎉 Audio generated successfully!")
                                st.audio(audio_bytes, format="audio/wav")
                                
                                # Download button
                                st.download_button(
                                    label="📥 Download Audiobook",
                                    data=audio_bytes,
                                    file_name=f"echoverse_{selected_tone}_{selected_emotion}.wav",
                                    mime="audio/wav"
                                )
                                
                                # Clean up
                                try:
                                    os.unlink(tmp_file.name)
                                except:
                                    pass
                                
                            else:
                                st.error("Failed to generate audio. Please try with shorter text or different settings.")
                        
                        except Exception as e:
                            st.error(f"Error generating audio: {e}")
                            st.info("Try reducing text length or reloading the page")
        else:
            st.info("👆 Enter text in the left panel to generate audio")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    with st.expander("ℹ️ About", expanded=False):
        st.markdown("""
        **EchoVerse** combines AI text enhancement with high-quality speech synthesis:
        
        🔧 **Features:**
        - **Text Enhancement**: Basic pattern-based improvements for different tones
        - **AI Rewriting**: Advanced text rewriting using language models
        - **Multiple Voices**: 10 different speaker voices to choose from
        - **Emotion Control**: Add emotions like happy, sad, excited, etc.
        - **Chunking**: Process long texts in manageable chunks
        - **Download**: Save your audiobook as WAV file
        
        🎯 **How to use:**
        1. Load the models using the sidebar button
        2. Enter your text in the left panel
        3. Choose enhancement options (basic or AI rewrite)
        4. Configure voice, tone, and emotion settings
        5. Generate your audiobook!
        
        ⚡ **Tips:**
        - Use chunking for texts longer than 1000 characters
        - Different voices work better with different content types
        - AI rewriting provides more natural flow for audiobooks
        """)
    
    # Add team footer
    add_team_footer()

if __name__ == "__main__":
    main()
