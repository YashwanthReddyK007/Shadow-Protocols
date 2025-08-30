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

# Custom CSS for black semi-transparent UI
def get_custom_css():
    return """
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;500;600;700&display=swap');
    
    /* Main app background - Pure black semi-transparent */
    .stApp {
        background: rgba(0, 0, 0, 0.85) !important;
        backdrop-filter: blur(10px);
    }
    
    /* Semi-transparent containers with black background */
    .main .block-container {
        background: rgba(0, 0, 0, 0.75) !important;
        backdrop-filter: blur(15px);
        border-radius: 20px;
        border: 2px solid rgba(76, 175, 80, 0.4);
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.8);
        padding: 2.5rem;
        margin: 1.5rem 0;
        max-width: 1400px;
        margin-left: auto;
        margin-right: auto;
        padding-bottom: 140px !important;
    }
    
    /* Sidebar styling - Black semi-transparent */
    .css-1d391kg {
        background: rgba(0, 0, 0, 0.9) !important;
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
    
    /* Input fields - Black semi-transparent */
    .stTextArea textarea, .stSelectbox select, .stTextInput input {
        background: rgba(0, 0, 0, 0.8) !important;
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
    
    /* Success/Error messages - Black semi-transparent */
    .stSuccess, .stInfo, .stWarning, .stError {
        background: rgba(0, 0, 0, 0.8) !important;
        backdrop-filter: blur(10px) !important;
        border-radius: 10px !important;
        border-left: 4px solid #4CAF50 !important;
    }
    
    /* Expander - Black semi-transparent */
    .streamlit-expanderHeader {
        background: rgba(0, 0, 0, 0.85) !important;
        border-radius: 12px !important;
        color: #E8F5E8 !important;
        font-family: 'Rajdhani', sans-serif !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        padding: 1rem 1.5rem !important;
        border: 1px solid rgba(76, 175, 80, 0.4) !important;
    }
    
    .streamlit-expanderContent {
        background: rgba(0, 0, 0, 0.9) !important;
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
    
    /* Info boxes - Black semi-transparent */
    .stInfo, .stSuccess, .stWarning, .stError {
        padding: 1rem 1.5rem !important;
        border-radius: 12px !important;
        font-size: 1rem !important;
        margin: 1rem 0 !important;
        background: rgba(0, 0, 0, 0.8) !important;
        backdrop-filter: blur(10px) !important;
    }
    
    /* Team info footer - Black semi-transparent */
    .team-footer {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: rgba(0, 0, 0, 0.95) !important;
        backdrop-filter: blur(25px);
        border-top: 3px solid rgba(129, 199, 132, 0.6);
        padding: 20px 0;
        text-align: center;
        z-index: 1000;
        box-shadow: 0 -6px 30px rgba(76, 175, 80, 0.3);
        min-height: 80px;
    }
    
    .team-name {
        font-family: 'Orbitron', monospace;
        font-size: 28px;
        font-weight: 900;
        color: #4CAF50;
        text-shadow: 0 0 20px rgba(76, 175, 80, 0.8);
        margin-bottom: 10px;
        letter-spacing: 3px;
    }
    
    .team-members {
        font-family: 'Rajdhani', sans-serif;
        font-size: 18px;
        font-weight: 600;
        color: #81C784;
        text-shadow: 0 0 12px rgba(129, 199, 132, 0.6);
        letter-spacing: 1px;
    }
    
    .team-lead {
        color: #FFD54F;
        font-weight: 700;
        text-shadow: 0 0 15px rgba(255, 213, 79, 0.5);
    }
    
    /* Scrollbar styling - Black theme */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(0, 0, 0, 0.5);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(45deg, #4CAF50, #81C784);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(45deg, #81C784, #4CAF50);
    }
    
    /* Audio player styling - Black semi-transparent */
    .stAudio {
        background: rgba(0, 0, 0, 0.85) !important;
        border-radius: 15px !important;
        padding: 1.5rem !important;
        border: 2px solid rgba(76, 175, 80, 0.4) !important;
        margin: 1.5rem 0 !important;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.6) !important;
        backdrop-filter: blur(10px) !important;
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
    
    /* Additional black theme elements */
    .stSlider > div > div > div {
        background: rgba(0, 0, 0, 0.8) !important;
    }
    
    .stNumberInput input {
        background: rgba(0, 0, 0, 0.8) !important;
        color: #E8F5E8 !important;
        border: 2px solid rgba(76, 175, 80, 0.6) !important;
    }
    
    /* Metric containers */
    .metric-container {
        background: rgba(0, 0, 0, 0.8) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        border: 1px solid rgba(76, 175, 80, 0.3) !important;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(0, 0, 0, 0.8) !important;
        border-radius: 12px !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(0, 0, 0, 0.6) !important;
        color: #81C784 !important;
        border-radius: 8px !important;
        margin: 0 4px !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(45deg, #4CAF50, #45a049) !important;
        color: white !important;
    }
    
    /* File uploader */
    .stFileUploader {
        background: rgba(0, 0, 0, 0.8) !important;
        border: 2px dashed rgba(76, 175, 80, 0.6) !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
    }
    
    /* Code blocks */
    .stCode {
        background: rgba(0, 0, 0, 0.9) !important;
        border: 1px solid rgba(76, 175, 80, 0.3) !important;
        border-radius: 8px !important;
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
            "v2/en_speaker_7": "Speaker 7 -
