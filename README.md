# Speaker Identification & Transcription System

An automated speaker identification and transcription system using SpeechBrain ECAPA-TDNN embeddings for speaker recognition and OpenAI Whisper for speech-to-text transcription.

## Features

- **Automatic Speaker Identification**: Uses SpeechBrain ECAPA-TDNN model to extract speaker embeddings
- **Speaker Profile Management**: Creates and maintains individual speaker profiles in organized folder structures
- **Adaptive Learning**: Updates existing speaker profiles with new audio data
- **Whisper Transcription**: High-quality speech-to-text with speaker labels
- **Multiple Output Formats**: JSON, TXT, SRT subtitles, and CSV timelines
- **Quality Filters**: Configurable quality thresholds for speaker detection

## System Requirements

- Python 3.8 or higher
- 4GB+ RAM (8GB+ recommended)
- GPU optional but recommended for faster processing
- ~2GB disk space for models and dependencies

## Installation

### 1. Clone or Download the Project

```bash
# If using git
git clone <repository-url>
cd speaker-identification-system

# Or download and extract the ZIP file
```

### 2. Create Virtual Environment

**On Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

You should see `(venv)` in your terminal prompt indicating the virtual environment is active.

### 3. Install Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip

# Install PyTorch (CPU version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# For GPU support (NVIDIA CUDA), use instead:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install audio processing libraries
pip install torchaudio
pip install librosa
pip install soundfile

# Install speaker diarization
pip install pyannote.audio

# Install SpeechBrain
pip install speechbrain

# Install Whisper
pip install openai-whisper

# Install other dependencies
pip install numpy
pip install scikit-learn
pip install python-dotenv
```

### 4. Get Hugging Face Access Token

The system uses pyannote.audio models which require a Hugging Face access token:

1. Create a free account at [huggingface.co](https://huggingface.co)
2. Go to Settings → Access Tokens
3. Create a new token with "read" permissions
4. Accept the terms for the speaker-diarization model at [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)

### 5. Configure Environment Variables

Create a `.env` file in the project root:

```bash
# Create .env file
touch .env

# Edit with your token
echo "HUGGING_FACE_ACCESS_TOKEN=your_token_here" > .env
```

Replace `your_token_here` with your actual Hugging Face token.

## Project Structure

```
speaker-identification-system/
├── .env                                    # Your Hugging Face token
├── README.md                              # This file
├── auto_speaker_tagging_system.py        # Main system
├── speaker_embedding_extractor.py         # Speaker embedding extraction
├── speaker_matcher.py                     # Speaker matching logic
├── speaker_saver.py                       # Speaker profile management
├── embedding_learner.py                   # Profile learning/updating
├── speaker_transcription_system.py        # Combined transcription system
├── simple_auto_tagging_example.py        # Usage examples
├── speakers/                              # Speaker profiles (created automatically)
│   ├── Speaker_001/
│   │   ├── Speaker_001.pkl
│   │   └── Speaker_001_info.json
│   └── Speaker_002/
│       ├── Speaker_002.pkl
│       └── Speaker_002_info.json
├── transcription_output/                  # Transcription results
└── pretrained_models/                     # Downloaded models (created automatically)
```

## Quick Start

### Basic Speaker Identification

```python
from auto_speaker_tagging_system import AutoSpeakerTaggingSystem

# Initialize system
system = AutoSpeakerTaggingSystem()

# Process an audio file
results = system.process_audio_file("your_audio.wav")

# Save results
system.save_results(results)
```

### Speaker Identification + Transcription

```python
from speaker_transcription_system import transcribe_with_speakers

# Process audio with speaker identification and transcription
results = transcribe_with_speakers(
    audio_file="your_audio.wav",
    whisper_model="base",
    language="en",  # or None for auto-detect
    simple_output=True
)
```

### Command Line Usage

**Speaker Identification Only:**
```bash
python auto_speaker_tagging_system.py your_audio.wav
```

**Speaker Identification + Transcription:**
```bash
python speaker_transcription_system.py your_audio.wav

# With options
python speaker_transcription_system.py your_audio.wav --language en --model large --output results
```

**Interactive Mode:**
```bash
python simple_auto_tagging_example.py
```

## Usage Examples

### Example 1: Simple Auto-Tagging

```python
from simple_auto_tagging_example import simple_auto_tag_example

results = simple_auto_tag_example("meeting_audio.wav")
```

This will:
1. Extract speakers from the audio
2. Match them against existing profiles
3. Update existing profiles if matches found
4. Create new profiles for unknown speakers

### Example 2: Custom Settings

```python
from auto_speaker_tagging_system import AutoSpeakerTaggingSystem

system = AutoSpeakerTaggingSystem(
    similarity_threshold=0.8,      # Stricter matching
    min_quality=0.5,               # Higher quality requirement
    min_speech_time=3.0,           # Longer minimum speech
    auto_save_unknown=True         # Save unknown speakers
)

results = system.process_audio_file(
    "interview.wav",
    custom_unknown_prefix="Guest",
    update_existing=True
)
```

### Example 3: Batch Processing

```python
audio_files = ["call1.wav", "call2.wav", "call3.wav"]

system = AutoSpeakerTaggingSystem()
all_results = system.process_multiple_files(audio_files)
```

### Example 4: Full Transcription with Speakers

```python
from speaker_transcription_system import quick_speaker_transcript

# Quick transcript with speaker labels
output_file = quick_speaker_transcript("podcast.wav", output_format="txt")
print(f"Transcript saved to: {output_file}")
```

## Configuration Options

### AutoSpeakerTaggingSystem Parameters

- **similarity_threshold** (0.0-1.0): Minimum similarity to match speakers (default: 0.75)
- **min_quality** (0.0-1.0): Minimum quality threshold (default: 0.3)
- **min_speech_time** (seconds): Minimum speech duration (default: 2.0)
- **learning_rate** (0.0-1.0): Rate for updating profiles (default: 0.7)
- **auto_save_unknown** (bool): Automatically save unknown speakers (default: True)

### Whisper Model Sizes

- **tiny**: Fastest, least accurate (~1GB RAM)
- **base**: Good balance (default, ~1GB RAM)
- **small**: Better accuracy (~2GB RAM)
- **medium**: High accuracy (~5GB RAM)
- **large**: Best accuracy (~10GB RAM)

## Output Formats

### Speaker Identification Results

- **Complete JSON**: Full results with all metadata
- **Timeline CSV**: Speaker segments with timestamps
- **Speaker Assignments**: Mapping of detected speakers to known profiles

### Transcription Results

- **Complete JSON**: Full transcription with speaker labels and metadata
- **Simple JSON**: Just speaker, start, end, and text
- **TXT**: Human-readable speaker-labeled transcript
- **SRT**: Subtitle format with speaker labels
- **CSV**: Timeline with speakers and text

## Managing Speaker Profiles

### List All Speakers

```python
system = AutoSpeakerTaggingSystem()
db_info = system.get_speaker_database_info()
print(f"Total speakers: {db_info['total_speakers']}")
print(f"Speakers: {db_info['speakers']}")
```

### Rename a Speaker

```python
from speaker_saver import SpeakerSaver

saver = SpeakerSaver()
saver.rename_speaker("Speaker_001", "John_Doe")
```

### Delete a Speaker

```python
saver.delete_speaker("Speaker_001")
```

### Verify Folder Structure

```bash
python auto_speaker_tagging_system.py --verify-folders
```

### Migrate Old Structure

If upgrading from an older version:

```bash
# Dry run (shows what would be migrated)
python auto_speaker_tagging_system.py --migrate-dry-run

# Actually migrate
python auto_speaker_tagging_system.py --migrate
```

## Troubleshooting

### "HUGGING_FACE_ACCESS_TOKEN not found" Error

Make sure your `.env` file exists and contains:
```
HUGGING_FACE_ACCESS_TOKEN=your_actual_token_here
```

### "Model not found" or Download Errors

The system downloads models automatically on first run. Ensure you have:
- Internet connection
- ~2GB free disk space
- Accepted model terms on Hugging Face

### Low Speaker Detection Quality

Try adjusting parameters:
```python
system = AutoSpeakerTaggingSystem(
    min_quality=0.0,        # Accept all quality levels
    min_speech_time=0.5,    # Accept shorter speech segments
    similarity_threshold=0.65  # More lenient matching
)
```

### Out of Memory Errors

- Use a smaller Whisper model (`tiny` or `base`)
- Process shorter audio files
- Close other applications
- Consider using GPU if available

### Virtual Environment Issues

**Deactivate and recreate:**
```bash
deactivate
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt  # if you created one
```

## Performance Tips

1. **GPU Acceleration**: Install CUDA-enabled PyTorch for 5-10x faster processing
2. **Batch Processing**: Process multiple files together for efficiency
3. **Model Selection**: Use smaller Whisper models for faster transcription
4. **Audio Quality**: Higher quality audio = better speaker identification
5. **Speaker Profiles**: More audio data per speaker = better matching accuracy

## Best Practices

1. **Audio Quality**: Use audio with clear speech and minimal background noise
2. **Profile Building**: Add 10-30 seconds of speech per speaker for best results
3. **Regular Backups**: Backup the `speakers/` folder periodically
4. **Naming Convention**: Use descriptive names when renaming speakers
5. **Quality Thresholds**: Start with default settings, adjust as needed

## Advanced Features

### Custom Speaker Naming

```python
system.save_speaker_with_name(
    embeddings=embeddings,
    speaker_id="SPEAKER_00",
    speaker_name="Dr_Smith",
    overwrite=False
)
```

### Learning Analytics

```python
from embedding_learner import EmbeddingLearner

learner = EmbeddingLearner()
stats = learner.get_speaker_stats("John_Doe")
trends = learner.analyze_learning_trends("John_Doe")
```

### Backup System

```python
from speaker_saver import SpeakerSaver

saver = SpeakerSaver()
backup_path = saver.backup_all_speakers("backups")
```

## Contributing

Contributions are welcome! Please ensure:
- Code follows existing style
- Functions have docstrings
- Changes are tested with sample audio

## License

[MIT License](https://github.com/JustinTheGreat/Speaker-Profiles/blob/main/LICENSE)

## Support

For issues, questions, or contributions:
- Check the troubleshooting section
- Review example scripts in `simple_auto_tagging_example.py`
- Examine the docstrings in each module

## Acknowledgments

- **SpeechBrain**: Speaker embedding extraction
- **pyannote.audio**: Speaker diarization
- **OpenAI Whisper**: Speech transcription
- **Hugging Face**: Model hosting and distribution