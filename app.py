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

# Set page config
st.set_page_config(
    page_title="🎬 Bark TTS Audiobook Generator",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

        advanced_prompts = {
            "Neutral": "Rewrite this text for a professional audiobook with clear narration:\n",
            "Suspenseful": "Transform this text into a suspenseful narrative with mysterious undertones:\n",
            "Inspiring": "Rewrite this text as inspiring and motivational with uplifting language:\n",
            "Conversational": "Rewrite this text as a natural podcast-style conversation:\n",
            "Educational": "Rewrite this text for educational narration with clear explanations:\n"
        }
        prompt = f"{advanced_prompts.get(tone, advanced_prompts['Neutral'])}\n{text}\n\nRewritten version:"

        try:
            inputs = self.granite_tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.granite_model.generate(
                    inputs,
                    max_new_tokens=500,
                    temperature=0.8,
                    do_sample=True,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=self.granite_tokenizer.eos_token_id
                )
            generated_text = self.granite_tokenizer.decode(outputs[0], skip_special_tokens=True)
            rewritten = generated_text.split("Rewritten version:")[-1].strip()
            return rewritten if len(rewritten) > 10 else text
        except Exception as e:
            st.error(f"Error in text rewriting: {e}")
            return text

    def load_granite_model(self):
        try:
            model_name = "ibm-granite/granite-3b-code-instruct"
            self.granite_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.granite_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            if self.granite_tokenizer.pad_token is None:
                self.granite_tokenizer.pad_token = self.granite_tokenizer.eos_token
            return True
        except:
            return self._load_fallback_model()

    def _load_fallback_model(self):
        try:
            model_name = "gpt2-medium"
            self.granite_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.granite_model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
            self.granite_tokenizer.pad_token = self.granite_tokenizer.eos_token
            return True
        except:
            return False

    def initialize_models(self):
        return self.load_granite_model()

# ---------------- Bark TTS ----------------
class BarkTTS:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained("suno/bark")
        self.model = None

    def load_model(self):
        """Load Bark model"""
        if self.model is not None:
            return True
        
        try:
            # Try GPU with FP16
            self.model = BarkModel.from_pretrained("suno/bark", torch_dtype=torch.float16)
            self.model.to(self.device)
            return True
        except RuntimeError:
            gc.collect()
            torch.cuda.empty_cache()
            self.device = "cpu"
            self.model = BarkModel.from_pretrained("suno/bark", torch_dtype=torch.float32)
            self.model.to(self.device)
            return True
        except Exception as e:
            st.error(f"Failed to load Bark model: {e}")
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
        """Generate audio and return audio data and sample rate"""
        if not self.load_model():
            return None, None
        
        # Enhanced mood mapping
        mood_mapping = {
            "neutral": "",
            "suspenseful": "[Whispers mysteriously]",
            "inspiring": "[Excited and motivated]",
            "conversational": "[Casual and friendly]",
            "educational": "[Clear and instructive]",
            "happy": "[Happy]",
            "sad": "[Sad]",
            "angry": "[Angry]",
            "surprised": "[Surprised]",
            "scared": "[Fearful]",
            "excited": "[Excited]",
            "calm": "[Calm]"
        }
        
        mood_tag = mood_mapping.get(mood.lower(), f"[{mood}]")
        styled_text = f"{mood_tag} {text}" if mood_tag else text
        
        inputs = self.processor(styled_text, voice_preset=preset, return_tensors="pt")
        for k, v in inputs.items():
            inputs[k] = v.to(self.device)

        with torch.no_grad():
            audio_array = self.model.generate(**inputs)

        # Convert to proper format
        audio_array = audio_array.cpu().numpy()
        audio_array = self.convert_audio_format(audio_array)
        
        sample_rate = self.model.generation_config.sample_rate
        return audio_array, sample_rate

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
    st.title("🎬 Bark TTS Audiobook Generator")
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
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
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
    
    with col2:
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
            
            # Chunking option for long texts
            chunk_audio = False
            if len(text_to_convert) > 1000:
                chunk_audio = st.checkbox(
                    f"📏 Process in chunks (Text is {len(text_to_convert)} characters)",
                    help="Recommended for texts longer than 1000 characters"
                )
            
            # Generate audio button
            if st.button("🎬 Generate Audiobook", type="primary"):
                if st.session_state.bark_tts is None:
                    st.error("Please load models first!")
                else:
                    with st.spinner("Generating audio... This may take a while."):
                        try:
                            if chunk_audio:
                                # Process in chunks
                                sentences = text_to_convert.split('. ')
                                chunks = []
                                current_chunk = ""
                                
                                for sentence in sentences:
                                    if len(current_chunk) + len(sentence) < 500:
                                        current_chunk += sentence + ". "
                                    else:
                                        if current_chunk:
                                            chunks.append(current_chunk.strip())
                                        current_chunk = sentence + ". "
                                
                                if current_chunk:
                                    chunks.append(current_chunk.strip())
                                
                                st.info(f"Processing {len(chunks)} chunks...")
                                
                                # Generate audio for each chunk
                                combined_audio = []
                                chunk_progress = st.progress(0)
                                
                                for i, chunk in enumerate(chunks):
                                    audio_data, sample_rate = st.session_state.bark_tts.generate_audio(
                                        chunk, selected_voice, selected_emotion
                                    )
                                    if audio_data is not None:
                                        combined_audio.append(audio_data)
                                        # Add small pause between chunks
                                        silence = np.zeros(int(sample_rate * 0.5))
                                        combined_audio.append(silence.astype(audio_data.dtype))
                                    
                                    chunk_progress.progress((i + 1) / len(chunks))
                                
                                # Combine all chunks
                                final_audio = np.concatenate(combined_audio)
                                chunk_progress.empty()
                                
                            else:
                                # Process as single piece
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
                                    file_name=f"audiobook_{selected_tone}_{selected_emotion}.wav",
                                    mime="audio/wav"
                                )
                                
                                # Clean up
                                os.unlink(tmp_file.name)
                                
                            else:
                                st.error("Failed to generate audio. Please try again.")
                        
                        except Exception as e:
                            st.error(f"Error generating audio: {e}")
        else:
            st.info("👆 Enter text in the left panel to generate audio")
    
    # Footer
    st.markdown("---")
    with st.expander("ℹ️ About", expanded=False):
        st.markdown("""
        **Bark TTS Audiobook Generator** combines AI text enhancement with high-quality speech synthesis:
        
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

if __name__ == "__main__":
    main()
