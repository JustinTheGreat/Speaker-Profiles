# speechbrain_speaker_embedding_extractor.py
import torch
import numpy as np
import torchaudio
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from pyannote.audio import Pipeline
from pyannote.core import Annotation, Segment
import os
from dotenv import load_dotenv
import warnings

class SpeechBrainSpeakerEmbeddingExtractor:
    """
    Extracts individual speaker embeddings from audio files using SpeechBrain ECAPA-TDNN
    with pyannote diarization for speaker segmentation.
    """
    
    def __init__(self):
        """Initialize the embedding extractor with SpeechBrain ECAPA-TDNN and pyannote diarization."""
        load_dotenv()
        self.hf_token = os.getenv("HUGGING_FACE_ACCESS_TOKEN")
        
        if not self.hf_token or self.hf_token == "YOUR_HUGGING_FACE_ACCESS_TOKEN":
            raise ValueError("Please set a valid HUGGING_FACE_ACCESS_TOKEN in your .env file")
        
        print("Loading models...")
        
        # Load diarization pipeline (using pyannote for speaker segmentation)
        self.diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=self.hf_token
        )
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.diarization_pipeline.to(self.device)
        
        # Load SpeechBrain ECAPA-TDNN model for embeddings
        self._load_speechbrain_model()
        
        print(f"Models loaded successfully on {self.device}")
    
    def _load_speechbrain_model(self):
        """Load SpeechBrain ECAPA-TDNN model."""
        try:
            from speechbrain.pretrained import EncoderClassifier
            
            print("Loading SpeechBrain ECAPA-TDNN model...")
            
            # Load ECAPA-TDNN model
            self.ecapa_model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="pretrained_models/spkrec-ecapa-voxceleb",
                run_opts={"device": str(self.device)}
            )
            print("✅ Loaded SpeechBrain ECAPA-TDNN model")
            
        except ImportError:
            raise ImportError("SpeechBrain is required. Install with: pip install speechbrain")
        except Exception as e:
            raise RuntimeError(f"Failed to load SpeechBrain ECAPA-TDNN model: {e}")
    
    def _preprocess_audio(self, waveform: torch.Tensor, sample_rate: int) -> Tuple[torch.Tensor, int]:
        """
        Preprocess audio to ensure compatibility with models.
        """
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            print("Converted stereo audio to mono")
        
        # Resample to 16kHz if needed (SpeechBrain ECAPA-TDNN expects 16kHz)
        target_sample_rate = 16000
        if sample_rate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, 
                new_freq=target_sample_rate
            )
            waveform = resampler(waveform)
            print(f"Resampled audio from {sample_rate}Hz to {target_sample_rate}Hz")
            sample_rate = target_sample_rate
        
        # Normalize audio
        if waveform.max() > 1.0 or waveform.min() < -1.0:
            waveform = waveform / torch.max(torch.abs(waveform))
            print("Normalized audio amplitude")
        
        return waveform, sample_rate
    
    def _prepare_for_speechbrain(self, segment_audio: torch.Tensor, sample_rate: int) -> Optional[torch.Tensor]:
        """
        Prepare audio segment for SpeechBrain ECAPA-TDNN model.
        
        Args:
            segment_audio: Audio segment tensor
            sample_rate: Sample rate
            
        Returns:
            Properly formatted tensor for SpeechBrain or None if failed
        """
        try:
            # Ensure we have audio data
            if segment_audio.numel() == 0:
                return None
            
            # Convert to mono if needed
            if segment_audio.shape[0] > 1:
                segment_audio = torch.mean(segment_audio, dim=0, keepdim=True)
            
            # Ensure we have the right shape: (1, samples)
            if segment_audio.dim() == 1:
                segment_audio = segment_audio.unsqueeze(0)
            elif segment_audio.dim() > 2:
                segment_audio = segment_audio.squeeze()
                if segment_audio.dim() == 1:
                    segment_audio = segment_audio.unsqueeze(0)
            
            # Ensure 16kHz sample rate
            target_sr = 16000
            if sample_rate != target_sr:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, 
                    new_freq=target_sr
                ).to(segment_audio.device)
                segment_audio = resampler(segment_audio)
            
            # Normalize amplitude
            max_val = torch.max(torch.abs(segment_audio))
            if max_val > 0:
                segment_audio = segment_audio / max_val
            
            # Ensure minimum length (ECAPA-TDNN needs reasonable duration)
            min_samples = target_sr // 2  # 0.5 seconds minimum
            if segment_audio.shape[1] < min_samples:
                # Pad with zeros if too short
                padding = min_samples - segment_audio.shape[1]
                segment_audio = torch.nn.functional.pad(segment_audio, (0, padding))
            
            # Final shape check: should be (1, samples)
            if segment_audio.shape[0] != 1:
                segment_audio = segment_audio.unsqueeze(0)
            
            # Move to CPU for SpeechBrain (it handles device internally)
            segment_audio = segment_audio.cpu()
            
            return segment_audio
            
        except Exception as e:
            print(f"❌ Error preparing audio for SpeechBrain: {e}")
            return None
    
    def extract_speaker_embeddings(self, audio_file: str, min_segment_duration: float = 1.0) -> Dict:
        """
        Extract individual speaker embeddings from an audio file using SpeechBrain ECAPA-TDNN
        with quality-weighted aggregation.
        """
        audio_path = Path(audio_file)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
        
        print(f"Processing audio file: {audio_file}")
        
        try:
            # Load audio
            waveform, sample_rate = torchaudio.load(audio_file)
            print(f"Loaded audio: {waveform.shape} at {sample_rate}Hz")
            
            # Preprocess audio
            waveform, sample_rate = self._preprocess_audio(waveform, sample_rate)
            print(f"Preprocessed audio: {waveform.shape} at {sample_rate}Hz")
            
            # Perform diarization to get speaker segments
            print("Performing speaker diarization...")
            diarization = self.diarization_pipeline(audio_file)
            
            # Extract embeddings for each speaker using ECAPA-TDNN with quality weighting
            speaker_embeddings = {}
            speaker_segments = {}
            speaker_qualities = {}
            
            print(f"Found {len(set([speaker for _, _, speaker in diarization.itertracks(yield_label=True)]))} speakers in diarization")
            
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                # Skip segments that are too short
                if turn.duration < min_segment_duration:
                    print(f"Skipping short segment for {speaker}: {turn.duration:.2f}s")
                    continue
                    
                if speaker not in speaker_embeddings:
                    speaker_embeddings[speaker] = []
                    speaker_segments[speaker] = []
                    speaker_qualities[speaker] = []
                    print(f"Processing speaker: {speaker}")
                
                # Extract audio segment
                start_sample = int(turn.start * sample_rate)
                end_sample = int(turn.end * sample_rate)
                
                # Ensure we don't go beyond audio length
                start_sample = max(0, start_sample)
                end_sample = min(waveform.shape[1], end_sample)
                
                if end_sample <= start_sample:
                    print(f"Invalid segment for {speaker}: start={start_sample}, end={end_sample}")
                    continue
                    
                segment_audio = waveform[:, start_sample:end_sample]
                print(f"Segment for {speaker}: {segment_audio.shape}, duration={turn.duration:.2f}s")
                
                # Get embedding for this segment using SpeechBrain ECAPA-TDNN
                try:
                    # Prepare audio for SpeechBrain
                    processed_audio = self._prepare_for_speechbrain(segment_audio, sample_rate)
                    
                    if processed_audio is None:
                        print(f"Failed to prepare audio for {speaker}")
                        continue
                    
                    with torch.no_grad():
                        # Extract embedding using ECAPA-TDNN
                        embedding = self.ecapa_model.encode_batch(processed_audio)
                        
                        # Handle different output formats
                        if isinstance(embedding, tuple):
                            embedding = embedding[0]  # Take first element if tuple
                        
                        # Convert to numpy
                        if hasattr(embedding, 'cpu'):
                            embedding_np = embedding.cpu().numpy()
                        else:
                            embedding_np = np.array(embedding)
                        
                        # Ensure 1D
                        embedding_np = embedding_np.flatten()
                        
                        # Calculate quality metrics for weighting
                        audio_energy = float(torch.mean(segment_audio ** 2))
                        duration_weight = min(turn.duration / 10.0, 1.0)  # Normalize to max 10s
                        quality_score = duration_weight * (1.0 + audio_energy * 10)
                        
                        speaker_embeddings[speaker].append(embedding_np)
                        speaker_segments[speaker].append({
                            'start': turn.start,
                            'end': turn.end,
                            'duration': turn.end - turn.start
                        })
                        speaker_qualities[speaker].append(quality_score)
                        
                        print(f"Successfully extracted ECAPA-TDNN embedding for {speaker}, shape: {embedding_np.shape}")
                        print(f"Quality score: {quality_score:.4f}")
                        
                except Exception as e:
                    print(f"Error processing segment for speaker {speaker}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # Quality-weighted averaging of embeddings for each speaker
            averaged_embeddings = {}
            for speaker, embeddings_list in speaker_embeddings.items():
                if embeddings_list:
                    print(f"\nProcessing {len(embeddings_list)} embeddings for {speaker}")
                    
                    # Ensure all embeddings have the same shape
                    shapes = [emb.shape for emb in embeddings_list]
                    print(f"Embedding shapes for {speaker}: {shapes}")
                    
                    if len(set(shapes)) > 1:
                        print(f"Warning: Inconsistent embedding shapes for {speaker}: {shapes}")
                        from collections import Counter
                        most_common_shape = Counter(shapes).most_common(1)[0][0]
                        # Filter to keep only embeddings with the most common shape
                        filtered_embeddings = []
                        filtered_qualities = []
                        for i, emb in enumerate(embeddings_list):
                            if emb.shape == most_common_shape:
                                filtered_embeddings.append(emb)
                                filtered_qualities.append(speaker_qualities[speaker][i])
                        embeddings_list = filtered_embeddings
                        speaker_qualities[speaker] = filtered_qualities
                        print(f"Using {len(embeddings_list)} embeddings with shape {most_common_shape}")
                    
                    if embeddings_list:
                        # Quality-weighted averaging
                        weights = np.array(speaker_qualities[speaker])
                        weights = weights / np.sum(weights)  # Normalize weights
                        
                        # Convert to numpy array and apply weighted averaging
                        embeddings_array = np.array(embeddings_list)
                        averaged_embedding = np.average(embeddings_array, weights=weights, axis=0)
                        
                        print(f"Final weighted-averaged embedding for {speaker}: shape={averaged_embedding.shape}")
                        print(f"Weights used: mean={np.mean(weights):.4f}, std={np.std(weights):.4f}")
                        print(f"Final embedding stats for {speaker}: mean={np.mean(averaged_embedding):.6f}, std={np.std(averaged_embedding):.6f}")
                        
                        # Verify the embedding is not problematic
                        if np.allclose(averaged_embedding, 0):
                            print(f"WARNING: Embedding for {speaker} is all zeros!")
                        elif np.std(averaged_embedding) < 1e-6:
                            print(f"WARNING: Embedding for {speaker} has very low variance!")
                        
                        averaged_embeddings[speaker] = averaged_embedding
            
            # Prepare output data
            embedding_data = {
                'file_info': {
                    'filename': audio_path.name,
                    'filepath': str(audio_path.absolute()),
                    'sample_rate': sample_rate,
                    'duration': float(waveform.shape[1] / sample_rate),
                    'channels': waveform.shape[0]
                },
                'speakers': {},
                'diarization': self._annotation_to_dict(diarization),
                'extraction_method': {
                    'diarization_model': 'pyannote/speaker-diarization-3.1',
                    'embedding_model': 'speechbrain/spkrec-ecapa-voxceleb',
                    'aggregation_method': 'quality_weighted_averaging',
                    'min_segment_duration': min_segment_duration
                }
            }
            
            # Add speaker data
            for speaker in averaged_embeddings:
                embedding_data['speakers'][speaker] = {
                    'embedding': averaged_embeddings[speaker],
                    'segments': speaker_segments[speaker],
                    'total_speech_time': sum(seg['duration'] for seg in speaker_segments[speaker]),
                    'segment_count': len(speaker_segments[speaker]),
                    'quality_scores': speaker_qualities[speaker],
                    'average_quality': np.mean(speaker_qualities[speaker]) if speaker_qualities[speaker] else 0.0
                }
            
            print(f"\nExtracted ECAPA-TDNN embeddings for {len(averaged_embeddings)} speakers")
            
            # Cross-similarity analysis for quality check
            speakers_list = list(averaged_embeddings.keys())
            for i in range(len(speakers_list)):
                for j in range(i + 1, len(speakers_list)):
                    speaker1, speaker2 = speakers_list[i], speakers_list[j]
                    emb1, emb2 = averaged_embeddings[speaker1], averaged_embeddings[speaker2]
                    
                    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                    print(f"Cross-similarity between {speaker1} and {speaker2}: {similarity:.6f}")
                    
                    if similarity > 0.95:
                        print(f"WARNING: Very high similarity between {speaker1} and {speaker2}!")
            
            return embedding_data
            
        except Exception as e:
            print(f"Error in extract_speaker_embeddings: {e}")
            import traceback
            traceback.print_exc()
            raise e
    
    def get_raw_segments(self, audio_file: str) -> List[Dict]:
        """
        Get raw diarization segments without embeddings.
        """
        print(f"Getting diarization segments for: {audio_file}")
        diarization = self.diarization_pipeline(audio_file)
        return self._annotation_to_dict(diarization)
    
    def _annotation_to_dict(self, annotation: Annotation) -> List[Dict]:
        """Convert pyannote Annotation to dictionary format."""
        segments = []
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            segments.append({
                'start': turn.start,
                'end': turn.end,
                'speaker': speaker,
                'duration': turn.end - turn.start
            })
        return segments