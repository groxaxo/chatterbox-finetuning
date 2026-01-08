# Chatterbox: Fine-Tuning Inference Kit for TTS & Turbo 🎙️

> ## 🚀 **NEW: Chatterbox Turbo Support ADDED!** 🚀
>
> This repository now fully supports the fine-tuning of the **Chatterbox Turbo** model!
>
> *   **What is it?** A faster, GPT-2 based architecture with a strong English foundation.
> *   **Smart Multi-Language Support:** The setup script **automatically merges** Turbo's large English vocabulary with our custom 23-language grapheme set.
> *   **The Result:** Get the speed and quality of Turbo while seamlessly fine-tuning on new languages like Turkish, French, Spanish, and more.
>
> Read the **Standart vs. Turbo Modes** section below for details!


---

A modular infrastructure for **fine-tuning** both **Chatterbox TTS (Standart)** and **Chatterbox Turbo** models with your own dataset and generating high-quality speech synthesis.

This kit is specially designed to support **new languages** by intelligently extending the model's vocabulary for maximum performance and faster adaptation.

---


## ⚠️ Understanding the Two Modes: Standart vs. Turbo

This repository operates in two distinct modes, controlled by the `is_turbo` setting in `src/config.py`. Please decide which mode you need before you begin.

### 1. Standart Mode (`is_turbo = False`)
*   **Architecture:** Llama-based.
*   **Tokenizer:** **Grapheme (character) based.** The `tokenizer.json` downloaded by `setup.py` contains a small, efficient vocabulary (~2,454 tokens) covering 23 languages.
*   **Best for:** Training a model with full control over a specific language from a more fundamental level.

### 2. Turbo Mode (`is_turbo = True`)
*   **Architecture:** GPT-2 based.
*   **Tokenizer:** **BPE-based.** It starts with a large, powerful English vocabulary (~50,000+ tokens).
*   **Smart Merging:** When you run `setup.py`, this large vocabulary is **automatically extended** with our multi-language grapheme set.
*   **Best for:** Leveraging a strong English base for faster, high-quality fine-tuning on other languages.


---

## ⚠️ **CRITICAL: Switching Between Training Modes**

If you plan to switch between **Standard Mode** (`is_turbo = False`) and **Turbo Mode** (`is_turbo = True`), you **MUST** completely delete the `pretrained_models` directory and the preprocessed_dir directory created with the `preprocess = True` operation before running the `setup.py` file again.

The setup script replaces the token files in place. If you run the setup for Standard mode after setting up for Turbo (or vice versa), the token files will become corrupted and cause errors that are difficult to debug during training.

**Important:** The two modes use different tokenization schemes and end-of-text (EOF) tokens:
- **Standard Mode**: Uses stop_text_token = 0
- **Turbo Mode**: Uses stop_text_token = 50256 (GPT-2's EOS token)

This means preprocessed data from one mode is **NOT compatible** with the other mode.

**Correct Workflow for Changing Modes:**

1. **DELETE the entire `pretrained_models` folder AND the preprocessed data folder.**

```bash

# On Linux or macOS
rm -rf pretrained_models
rm -rf MyTTSDataset/preprocess  # or your custom preprocessed_dir path

# On Windows (in Command Prompt)
rmdir /s /q pretrained_models
rmdir /s /q MyTTSDataset\preprocess
```
2. **Update** the `src/config.py` file, setting the `is_turbo` flag to your desired new mode and setting preprocess = True if it is False.

3. **RUN setup.py again** to download and prepare the correct files for the new mode.

```bash
python setup.py

```
4. **Update** the `new_vocab_size` value in the `src/config.py` file with the new value provided by the setup script. Also ensure preprocess = True.

---


## ⚠️ CRITICAL INFORMATION (Please Read)

### 0. Preprocessing is Mandatory
This repository uses an **offline preprocessing** strategy to maximize training speed. This script processes all audio files, extracts speaker embeddings and acoustic tokens, and saves them as `.pt` files.

### 1. Tokenizer and Vocab Size (Most Important)
Chatterbox uses a grapheme-based (character-level) tokenizer. The `tokenizer.json` file downloaded by `setup.py` includes support for **23 languages** from the original Chatterbox repository, covering most common characters across multiple languages.

*   **Default Support:** The provided tokenizer already includes characters for English, Turkish, French, German, Spanish, and 18+ other languages
*   **When to customize:** If your target language has special characters not covered in the default tokenizer, you can create a custom `tokenizer.json`
*   **Examples of special characters by language:**
    *   Turkish: `ç, ğ, ş, ö, ü, ı`
    *   French: `é, è, ê, à, ù, ç`
    *   German: `ä, ö, ü, ß`
    *   Spanish: `ñ, á, é, í, ó, ú`
*   **Critical:** The `NEW_VOCAB_SIZE` variable in both `src/config.py` AND `inference.py` **must exactly match** the total number of tokens in your `tokenizer.json` file
*   **Default vocab size:** Check the downloaded `tokenizer.json` to see the exact token count, then set `NEW_VOCAB_SIZE` accordingly

### 2. Audio Sample Rates
*   **Training (Input):** Chatterbox's encoder and T3 module work with **16,000 Hz (16kHz)** audio. Even if your dataset uses different rates, `dataset.py` automatically resamples to 16kHz.
*   **Output (Inference):** The model's vocoder generates audio at **24,000 Hz (24kHz)**.

---

## 📂 Folder Structure

```text
chatterbox-finetune/
├── pretrained_models/                             # setup.py downloads required models here
│   ├── ve.safetensors
│   ├── s3gen.safetensors
│   ├── t3.safetensors
│   └── tokenizer.json
├── MyTTSDataset/                                  # Your custom dataset in LJSpeech format
│   ├── metadata.csv                               # Dataset metadata (file|text|normalized_text)
│   └── wavs/                                      # Directory containing WAV files
├── FileBasedDataset/                              # Your custom dataset in LJSpeech format
│   ├── 0a0bc5d3-f195-464a-8716-d6e01fd4784f.txt   # Dataset metadata (text)
│   └── 0a0bc5d3-f195-464a-8716-d6e01fd4784f.wav   # WAV files
├── speaker_reference/                             # Speaker reference audio files
│   └── reference.wav                              # Reference audio for voice cloning
├── src/
│   ├── config.py                                  # All settings and hyperparameters
│   ├── dataset.py                                 # Data loading and processing
│   ├── model.py                                   # Model weight transfer and training wrapper
|   ├── preprocess_ljspeech.py                     # Preprocessing script
|   ├── preprocess_file_based.py                   # Preprocessing script
│   └── utils.py                                   # Logger and VAD utilities
├── train.py                                       # Main training script
├── inference.py                                   # Speech synthesis script (with VAD support)
├── setup.py                                       # Setup script for downloading models
├── requirements.txt                               # Required dependencies
└── README.md                                      # This file
```

---

## 🚀 Installation

### 1. Install Dependencies
Requires Python 3.8+ and GPU (recommended):

**Install FFmpeg (Required):**
```bash
# on Ubuntu or Debian
sudo apt update && sudo apt install ffmpeg

# on Arch Linux
sudo pacman -S ffmpeg

# on MacOS using Homebrew (https://brew.sh/)
brew install ffmpeg

# on Windows using Chocolatey (https://chocolatey.org/)
choco install ffmpeg

# on Windows using Scoop (https://scoop.sh/)
scoop install ffmpeg
```

**Install Python Dependencies:**
```bash

git clone https://github.com/gokhaneraslan/chatterbox-finetuning.git
cd chatterbox-finetuning

pip install -r requirements.txt
```


### 2. Download & Prepare Models (CRITICAL)
This multi-step process prepares all necessary files based on your chosen mode. This script downloads the necessary base models (`ve`, `s3gen`, `t3`) and default tokenizer. **Must be run before training.**

**Step 2.1: Choose Your Mode**
Open `src/config.py` and set the `is_turbo` variable to `True` or `False`.

```python
# In src/config.py
is_turbo: bool = True  # Set to True for Turbo, False for Standart```
```

**Step 2.2: Run the Setup Script**
This command will download the correct model files. If Turbo mode is enabled, it will also **automatically merge the tokenizers for you.**

```bash
python setup.py
```

**Step 2.3: Update Config (Turbo Mode ONLY)**
If you ran the setup in Turbo mode, the script will output a final message like this:

`Please update the 'new_vocab_size' in 'src/config.py' to the following value: 52260`

Copy this exact number and paste it into the `new_vocab_size` variable in `src/config.py`. **Do not skip this step!**

---


### 3. Configure Environment
Create a `.env` file or edit `src/config.py` to specify your dataset location and training parameters.

---

## 🏋️ Training (Fine-Tuning)

During training, the script loads the original model weights, **intelligently resizes them** for the new vocabulary size, and initializes new tokens using mean initialization from existing tokens for faster adaptation.

### 1. Dataset Preparation

#### Option A: Using TTS Dataset Generator (Recommended)
We recommend using the [TTS Dataset Generator](https://github.com/gokhaneraslan/tts-dataset-generator) tool to automatically create high-quality datasets from audio or video files.

**Quick Start:**
```bash
# Install the dataset generator
git clone https://github.com/gokhaneraslan/tts-dataset-generator.git
cd tts-dataset-generator
pip install -r requirements.txt

# Generate dataset from your audio/video file
python main.py --file your_audio.mp4 --model large --language en --ljspeech True
```

This will automatically:
- Segment audio into optimal chunks (3-10 seconds)
- Transcribe using Whisper AI
- Generate properly formatted `metadata.csv` and audio files
- Output directly to `MyTTSDataset/` folder in LJSpeech format

**Benefits:**
- Saves hours of manual segmentation and transcription
- Optimizes chunk duration for TTS training
- Handles multiple languages (en, tr, fr, de, es, etc.)
- Works with both audio and video files

#### Option B: Manual Dataset Creation
Your dataset should follow the LJSpeech format with a CSV file:
`filename|raw_text|normalized_text`

Example `metadata.csv`:
```text
recording_001|Hello world.|hello world
recording_002|This is a test recording.|this is a test recording
```

Place your dataset in the `MyTTSDataset/` folder:
```text
MyTTSDataset/
├── metadata.csv
└── wavs/
    ├── recording_001.wav
    ├── recording_002.wav
    └── ...
```

**Dataset Quality Requirements:**
- Sample rate: 16kHz, 22.05kHz, or 44.1kHz (will be resampled to 16kHz automatically)
- Format: WAV (mono or stereo - will be converted to mono automatically)
- Duration: 3-10 seconds per segment (optimal for TTS)
- Minimum total duration: 30+ minutes for basic training
- **Recommended:** 1 hour of clean audio for optimal results
- Audio quality: Clean, minimal background noise

### 2. Configuration
**Important:** Ensure the `NEW_VOCAB_SIZE` in **both** `src/config.py` **AND** `inference.py` matches the number of tokens in your custom `tokenizer.json`.

**For non-English languages:**
1. Create your custom `tokenizer.json` with all characters in your target language
2. Count the total tokens in your JSON file
3. Update `NEW_VOCAB_SIZE` in both files to match this count

**Most Important Settings:**
```python
# In src/config.py
is_turbo: bool = True # Set True if you're training Turbo, False if you're training Normal.

# --- Vocabulary ---
# The size of the NEW vocabulary (from tokenizer.json)
# Ensure this matches the JSON file generated by your tokenizer script.
# For Turbo mode: Use the exact number provided by setup.py (e.g., 52260)
new_vocab_size: int = 52260 if is_turbo else 2454 

# In inference.py
NEW_VOCAB_SIZE = 2454  # Must be identical to config.py
```

Other key parameters to adjust:
```python
# Dataset
DATASET_PATH = "MyTTSDataset"
METADATA_FILE = "metadata.csv"

# Training
BATCH_SIZE = 4         # Adjust based on your GPU VRAM
LEARNING_RATE = 5e-5
NUM_EPOCHS = 50
```

If your dataset is file-based dataset, set ljspeech = False in the configuration file.
```python
# In src/config.py
ljspeech = False  

```

If you have already done the preprocessing process once, set preprocess=False in the config file to avoid doing it again.
```python
# In src/config.py
preprocess = False  

```


### 3. Start Training
```bash
python train.py
```

The trained model will be saved as `chatterbox_output/t3_finetuned.safetensors`. The filename will be 
`t3_turbo_finetuned.safetensors` for Turbo mode.

**Training Tips:**
*   **VRAM:** T3 is a Transformer model with high VRAM usage. For 12GB VRAM, use `batch_size=4`. For lower VRAM, use `batch_size=2` with `grad_accum=32`.
*   **Mixed Precision:** Code uses `fp16=True` by default for faster training and memory efficiency.
*   **Checkpointing:** Models are saved every epoch in `chatterbox_output/`.
*   **Recommended Training Duration:** For optimal results with 1 hour of target speaker audio, train for **150 epochs** or **1000 steps**. This configuration typically produces high-quality voice cloning results.

---

## 🗣️ Inference (Speech Synthesis)

The inference script loads your fine-tuned `.safetensors` file and uses **Silero VAD** to automatically trim unwanted silence/noise at the end of generated audio.

### 1. Prepare Reference Audio (Prompt)
Chatterbox is a voice cloning/style transfer model. You **must provide a reference `.wav` file** (audio prompt) for inference.

Place your reference audio in `speaker_reference/`:
```text
speaker_reference/
└── reference.wav
```

**Reference Audio Requirements:**
*   Format: WAV, mono or stereo
*   Sample rate: Any (will be resampled automatically)
*   Duration: 3-10 seconds recommended
*   Quality: Clean audio with minimal background noise

### 2. Running Inference
Edit `inference.py` to set your text and audio prompt paths:

```python
TEXT_TO_SAY = "This is a test of the fine-tuned model."
AUDIO_PROMPT = "speaker_reference/reference.wav"
```

Run inference:
```bash
python inference.py
```

The output will be saved as `output_stitched.wav` (24kHz).

### 3. Advanced Usage

**Multiple Sentences:**
The script automatically splits long text into sentences for better quality:
```python
TEXT_TO_SAY = "Hello! How are you today? This is amazing."
```

**Audio Processing:**
All audio is automatically processed to mono and resampled to the correct sample rate using FFmpeg. The output format is:
- **Channels:** Mono (1 channel)
- **Sample Rate:** 24kHz
- **Codec:** 16-bit PCM WAV

---

## 🛠️ Technical Details

### Why Preprocessing?
Original Chatterbox training pipelines often process audio "on-the-fly" (resampling, feature extraction) during training. This causes the GPU to wait for the CPU, slowing down training significantly.
By running `preprocess.py`, we:
1.  Extract Speaker Embeddings (Voice Encoder)
2.  Extract Acoustic Tokens (S3Gen)
3.  Tokenize Text
4.  Save everything as optimized PyTorch tensors (`.pt`)
This allows the `dataset.py` to simply load tensors, maximizing GPU utilization.


### Tokenizer Structure

**Standart Model Tokenizer:**
The `pretrained_models/tokenizer.json` file downloaded by `setup.py` includes support for **23 languages** with extensive grapheme coverage. This file is used by `src/chatterbox/tokenizer.py` during both training and inference.

**Turbo Model Tokenizer (Smart Vocab Extension):**
Turbo mode uses GPT-2's powerful BPE tokenizer as a base. The `setup.py` script performs a **"Vocab Extension"**: it intelligently adds all unique characters from our 23-language grapheme set to the GPT-2 vocabulary. This process ensures that:
1.  The model retains its powerful knowledge of English words and structures.
2.  Special characters from other languages (e.g., `ğ, ş, ı` for Turkish; `é, à, ç` for French) are recognized as single, whole tokens, dramatically improving learning efficiency.
3.  **You do not need to create a custom tokenizer manually.** The setup is fully automated.

---

**Default Multi-Language Support:**
The provided tokenizer already covers common characters from 23 languages, including but not limited to:
- Latin-based languages (English, French, Spanish, German, Italian, Portuguese)
- Turkish with special characters (ç, ğ, ı, ö, ş, ü)
- Eastern European languages
- And more

**When to Create a Custom Tokenizer:**
You only need to create a custom tokenizer if:
1. Your target language has special characters not in the default set
2. You want to optimize the vocab size for a specific language
3. You need to add domain-specific symbols or characters

**Creating a Custom Tokenizer (Optional):**

1. **Identify all characters** in your target language:
   - All letters (including accented/special characters)
   - Numbers (0-9)
   - Punctuation marks
   - Special symbols used in your language

2. **Create the JSON mapping** - Example structure:
```json
{
  "a": 0,
  "b": 1,
  "c": 2,
  "ç": 3,
  "d": 4,
  ...
  " ": 100,
  ".": 101,
  ",": 102,
  ...
}
```

3. **Count total tokens** in your JSON file

4. **Update NEW_VOCAB_SIZE** in both `src/config.py` AND `inference.py` to match the token count

5. **Replace** `pretrained_models/tokenizer.json` with your custom file before training

**Vocab Size Examples:**
- **Default (23 languages):** Check your downloaded `tokenizer.json` for exact count
- **Custom French:** ~200 tokens (if you want French-only optimization)
- **Custom German:** ~180 tokens (if you want German-only optimization)

**Important:** The default tokenizer should work for most languages. Only customize if you have specific requirements or encounter missing characters.

### VAD Integration
During inference, `inference.py` uses Silero VAD to prevent hallucinations and sentence-ending elongations. This automatically trims unwanted silence and noise from generated audio.

### Audio Processing Pipeline
All audio processing uses **FFmpeg** for professional-quality results:
- **Input:** Automatic conversion to mono (1 channel)
- **Resampling:** Automatic resampling to required sample rates
- **Training:** 16kHz processing
- **Output:** 24kHz, 16-bit PCM WAV format
- **Codec:** `pcm_s16le` (16-bit signed little-endian PCM)

### Model Architecture
*   **VE (Voice Encoder):** Extracts speaker embeddings from reference audio
*   **T3 (Text-to-Speech):** Main transformer-based TTS model (this is what you fine-tune)
*   **S3Gen (Vocoder):** Converts mel-spectrograms to waveforms

---

## 📝 Troubleshooting

**Error:** `RuntimeError: Error(s) in loading state_dict for T3... size mismatch`
*   **Solution:** `NEW_VOCAB_SIZE` doesn't match the token count in `tokenizer.json`. 
*   **Check:** 
    1. Count tokens in your `tokenizer.json` file
    2. Verify `NEW_VOCAB_SIZE` in `src/config.py` matches this count
    3. Verify `NEW_VOCAB_SIZE` in `inference.py` also matches (must be identical)
*   **Common mistake:** Updating only one file but not the other

**Error:** `FileNotFoundError: ... ve.safetensors`
*   **Solution:** You haven't downloaded base models. Run `python setup.py`.

**Error:** `CUDA out of memory`
*   **Solution:** Reduce `BATCH_SIZE` in `src/config.py` or enable gradient accumulation.

**Poor Quality Output:**
*   Check reference audio quality (should be clean, at least 5 seconds)
*   Ensure adequate training data (minimum 30 minutes recommended)

---

## 🙏 Acknowledgments

Based on the Chatterbox TTS model architecture. Special thanks to the original authors and contributors.

---

## 📧 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review `src/config.py` for configuration options
3. Open an issue on GitHub with detailed error messages and your setup information
