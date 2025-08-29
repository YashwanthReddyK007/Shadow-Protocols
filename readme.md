# 🎬 Bark TTS Audiobook Generator

A Streamlit web application for creating audiobooks with AI-powered text enhancement and emotional speech synthesis using Suno's Bark TTS model.

## 🚀 Features

- **AI Text Enhancement**: Transform text with different tones and styles
- **Multiple Voices**: Choose from 10 different speaker voices
- **Emotion Control**: Add emotions like happy, sad, excited, etc.
- **Interactive UI**: Separate text processing and audio generation
- **Chunking Support**: Handle long texts efficiently
- **Download Audio**: Save audiobooks as WAV files

## 📦 Installation

### Option 1: Standard Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd bark-tts-audiobook-generator

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run streamlit_app.py
```

### Option 2: Using Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run streamlit_app.py
```

### Option 3: Using Conda

```bash
# Create conda environment
conda create -n bark-tts python=3.9
conda activate bark-tts

# Install PyTorch (choose based on your system)
conda install pytorch torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt

# Run the application
streamlit run streamlit_app.py
```

## 🔧 System Requirements

- **Python**: 3.8 or higher
- **RAM**: Minimum 8GB, 16GB recommended
- **GPU**: CUDA-compatible GPU recommended (optional, will use CPU if not available)
- **Storage**: ~5GB for models and cache

## 🖥️ GPU Setup (Optional but Recommended)

For faster audio generation, install CUDA:

```bash
# Check if CUDA is available
python -c "import torch; print(torch.cuda.is_available())"

# Install CUDA version of PyTorch if needed
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 📝 Usage

1. **Start the application**:
   ```bash
   streamlit run streamlit_app.py
   ```

2. **Open in browser**: The app will automatically open at `http://localhost:8501`

3. **Load Models**: Click "Load Models" in the sidebar (first time only)

4. **Process Text**:
   - Enter your text in the left panel
   - Choose "Enhance Text Only" for basic improvements
   - Choose "AI Rewrite Text" for advanced transformation

5. **Generate Audio**:
   - Configure voice, tone, and emotion in the sidebar
   - Click "Generate Audiobook" in the right panel
   - Preview and download your audiobook

## 🎛️ Configuration Options

### Voices
- `v2/en_speaker_0` - Male, Clear
- `v2/en_speaker_1` - Female, Warm
- `v2/en_speaker_2` - Male, Deep
- `v2/en_speaker_3` - Female, Energetic
- `v2/en_speaker_4` - Male, Calm
- `v2/en_speaker_5` - Female, Professional
- `v2/en_speaker_6` - Male, Narrative
- `v2/en_speaker_7` - Female, Expressive
- `v2/en_speaker_8` - Male, Authoritative
- `v2/en_speaker_9` - Female, Gentle

### Tones
- **Neutral**: Professional and clear
- **Suspenseful**: Mysterious and intriguing
- **Inspiring**: Motivational and uplifting
- **Conversational**: Friendly and casual
- **Educational**: Instructive and informative

### Emotions
- neutral, happy, sad, excited, calm, surprised, angry, scared

## 🚨 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# The app will automatically fall back to CPU
# Or try reducing batch size in the code
```

**2. Model Download Issues**
```bash
# Clear cache and retry
rm -rf ~/.cache/huggingface/
pip install -r requirements.txt --force-reinstall
```

**3. Audio Generation Fails**
```bash
# Check dependencies
pip install scipy --upgrade
pip install numpy --upgrade
```

**4. Streamlit Issues**
```bash
# Update Streamlit
pip install streamlit --upgrade

# Clear cache
streamlit cache clear
```

### Performance Tips

- **Use GPU**: Install CUDA for 5-10x faster generation
- **Chunk Long Texts**: Enable chunking for texts >1000 characters
- **Close Other Applications**: Free up RAM for model loading
- **Use SSD Storage**: Faster model loading from SSD drives

## 📁 Project Structure

```
bark-tts-audiobook-generator/
├── streamlit_app.py          # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                # This file
└── .gitignore               # Git ignore file
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Suno AI](https://github.com/suno-ai/bark) for the Bark TTS model
- [Hugging Face](https://huggingface.co/) for transformer models
- [Streamlit](https://streamlit.io/) for the web framework

## ⚠️ Disclaimers

- **Resource Usage**: This application requires significant computational resources
- **Model Loading**: First-time setup downloads several GB of model data
- **Audio Quality**: Quality depends on input text and chosen settings
- **Commercial Use**: Check Bark TTS licensing for commercial applications
