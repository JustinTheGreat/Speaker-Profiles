# speaker_transcription_system.py
"""
Combined Speaker Identification + Whisper Transcription System

This system combines:
1. Speaker identification using SpeechBrain ECAPA-TDNN (from quick_quality_fix.py)
2. Whisper transcription for accurate speech-to-text
3. Timeline alignment to produce speaker-labeled transcripts

Features:
- Bypassed quality filters for maximum speaker detection
- Word-level timestamp alignment
- Multiple output formats (JSON, SRT, TXT)
- Speaker confidence scores
"""

import json
import whisper
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import numpy as np

# Import the speaker identification components
from auto_speaker_tagging_system import AutoSpeakerTaggingSystem
from simple_auto_tagging_example import simple_auto_tag_example

class SpeakerAwareTranscriptionSystem:
    """
    System that combines speaker identification with Whisper transcription
    to produce speaker-labeled transcripts.
    """
    
    def __init__(self, 
                 whisper_model_size: str = "base",
                 speakers_folder: str = "speakers",
                 bypass_quality_filters: bool = True):
        """
        Initialize the speaker-aware transcription system.
        
        Args:
            whisper_model_size: Whisper model size ("tiny", "base", "small", "medium", "large")
            speakers_folder: Folder containing speaker profiles
            bypass_quality_filters: Use bypassed quality filters for maximum speaker detection
        """
        self.speakers_folder = Path(speakers_folder)
        self.bypass_quality_filters = bypass_quality_filters
        
        print("🎙️ Initializing Speaker-Aware Transcription System...")
        
        # Initialize Whisper
        print(f"📝 Loading Whisper model: {whisper_model_size}")
        self.whisper_model = whisper.load_model(whisper_model_size)
        
        # Initialize speaker identification system with bypassed quality filters
        if bypass_quality_filters:
            print("🔓 Using bypassed quality filters for maximum speaker detection")
            self.speaker_system = AutoSpeakerTaggingSystem(
                similarity_threshold=0.4,     # Lowered similarity threshold
                min_quality=0.0,             # Accept ANY quality (bypass)
                min_speech_time=0.3,         # Accept very short speech (bypass)
                learning_rate=0.7,
                auto_save_unknown=True
            )
        else:
            print("⚙️ Using standard quality filters")
            self.speaker_system = AutoSpeakerTaggingSystem()
        
        print("✅ System initialization complete!")
    
    def transcribe_with_speakers(self, audio_file: str, 
                                language: Optional[str] = None,
                                word_timestamps: bool = True) -> Dict:
        """
        Transcribe audio file with speaker identification.
        
        Args:
            audio_file: Path to audio file
            language: Language for Whisper (None for auto-detection)
            word_timestamps: Include word-level timestamps
            
        Returns:
            Combined transcription and speaker identification results
        """
        audio_path = Path(audio_file)
        print(f"\n🎙️ Processing: {audio_path.name}")
        print("=" * 80)
        
        # Step 1: Speaker Identification
        print("1️⃣ SPEAKER IDENTIFICATION")
        print("-" * 40)
        
        if self.bypass_quality_filters:
            speaker_results = self._process_with_quality_bypass(audio_file)
        else:
            speaker_results = self.speaker_system.process_audio_file(audio_file)
        
        # Step 2: Whisper Transcription
        print("\n2️⃣ WHISPER TRANSCRIPTION")
        print("-" * 40)
        
        transcription_options = {
            "word_timestamps": word_timestamps,
            "verbose": False
        }
        if language:
            transcription_options["language"] = language
        
        print(f"🎵 Transcribing with Whisper ({self.whisper_model.dims.n_mels} mel bins)...")
        whisper_result = self.whisper_model.transcribe(audio_file, **transcription_options)
        
        print(f"📝 Transcription complete!")
        print(f"   Language detected: {whisper_result.get('language', 'unknown')}")
        print(f"   Segments: {len(whisper_result.get('segments', []))}")
        if word_timestamps and 'segments' in whisper_result:
            total_words = sum(len(seg.get('words', [])) for seg in whisper_result['segments'])
            print(f"   Words: {total_words}")
        
        # Step 3: Align Speakers with Transcription
        print("\n3️⃣ SPEAKER-TRANSCRIPT ALIGNMENT")
        print("-" * 40)
        
        aligned_result = self._align_speakers_with_transcript(
            speaker_results, whisper_result, word_timestamps
        )
        
        # Step 4: Generate Final Results
        print("\n4️⃣ GENERATING RESULTS")
        print("-" * 40)
        
        final_result = self._compile_final_results(
            audio_file, speaker_results, whisper_result, aligned_result
        )
        
        # Summary
        print(f"\n✅ PROCESSING COMPLETE!")
        print(f"   🎤 Speakers identified: {len(speaker_results.get('speaker_assignments', {}))}")
        print(f"   📝 Segments transcribed: {len(whisper_result.get('segments', []))}")
        print(f"   🔗 Aligned segments: {len(aligned_result)}")
        
        return final_result
    
    def _process_with_quality_bypass(self, audio_file: str) -> Dict:
        """Process audio with bypassed quality filters (from quick_quality_fix.py)."""
        print(f"🔓 Processing with quality bypass...")
        
        # Get file prefix for naming
        file_prefix = Path(audio_file).stem
        
        # Process with bypassed quality filters
        results = self.speaker_system.process_audio_file(
            audio_file,
            custom_unknown_prefix=f"{file_prefix}_Speaker",
            update_existing=True
        )
        
        # Show detailed results
        print("📊 Speaker identification results:")
        summary = results['summary']
        print(f"   Total speakers detected: {summary['total_detected']}")
        print(f"   Speakers processed: {summary['qualified_speakers']}")
        print(f"   Matched to existing: {summary['matched_existing']}")
        print(f"   New profiles created: {summary['new_profiles_created']}")
        
        return results
    
    def _align_speakers_with_transcript(self, speaker_results: Dict, 
                                      whisper_result: Dict, 
                                      word_timestamps: bool) -> List[Dict]:
        """
        Align speaker segments with transcription segments.
        
        Args:
            speaker_results: Results from speaker identification
            whisper_result: Results from Whisper transcription
            word_timestamps: Whether word timestamps are available
            
        Returns:
            List of aligned segments with speaker and text
        """
        aligned_segments = []
        
        # Get speaker timeline
        speaker_timeline = speaker_results.get('labeled_timeline', [])
        whisper_segments = whisper_result.get('segments', [])
        
        print(f"Aligning {len(speaker_timeline)} speaker segments with {len(whisper_segments)} transcript segments...")
        
        for whisper_seg in whisper_segments:
            start_time = whisper_seg['start']
            end_time = whisper_seg['end']
            text = whisper_seg['text'].strip()
            
            # Find overlapping speaker segments
            overlapping_speakers = self._find_overlapping_speakers(
                start_time, end_time, speaker_timeline
            )
            
            if overlapping_speakers:
                # Use the speaker with the most overlap
                primary_speaker = max(overlapping_speakers, 
                                    key=lambda x: x['overlap_duration'])
                
                segment = {
                    'start': start_time,
                    'end': end_time,
                    'duration': end_time - start_time,
                    'text': text,
                    'speaker': primary_speaker['speaker'],
                    'speaker_confidence': primary_speaker['confidence'],
                    'speaker_overlap': primary_speaker['overlap_duration'],
                    'multiple_speakers': len(overlapping_speakers) > 1,
                    'all_speakers': [s['speaker'] for s in overlapping_speakers] if len(overlapping_speakers) > 1 else None
                }
                
                # Add word-level information if available
                if word_timestamps and 'words' in whisper_seg:
                    segment['words'] = whisper_seg['words']
                    segment['word_count'] = len(whisper_seg['words'])
                
            else:
                # No speaker identified for this segment
                segment = {
                    'start': start_time,
                    'end': end_time,
                    'duration': end_time - start_time,
                    'text': text,
                    'speaker': 'Unknown',
                    'speaker_confidence': 0.0,
                    'speaker_overlap': 0.0,
                    'multiple_speakers': False,
                    'all_speakers': None
                }
                
                if word_timestamps and 'words' in whisper_seg:
                    segment['words'] = whisper_seg['words']
                    segment['word_count'] = len(whisper_seg['words'])
            
            aligned_segments.append(segment)
        
        return aligned_segments
    
    def _find_overlapping_speakers(self, start_time: float, end_time: float, 
                                 speaker_timeline: List[Dict]) -> List[Dict]:
        """
        Find speaker segments that overlap with the given time range.
        
        Args:
            start_time: Start time of transcript segment
            end_time: End time of transcript segment
            speaker_timeline: Speaker timeline from identification
            
        Returns:
            List of overlapping speakers with overlap information
        """
        overlapping_speakers = []
        
        for speaker_seg in speaker_timeline:
            speaker_start = speaker_seg['start']
            speaker_end = speaker_seg['end']
            
            # Calculate overlap
            overlap_start = max(start_time, speaker_start)
            overlap_end = min(end_time, speaker_end)
            
            if overlap_start < overlap_end:
                overlap_duration = overlap_end - overlap_start
                segment_duration = end_time - start_time
                overlap_ratio = overlap_duration / segment_duration
                
                # Determine confidence based on overlap ratio
                if overlap_ratio >= 0.8:
                    confidence = 'high'
                elif overlap_ratio >= 0.5:
                    confidence = 'medium'
                else:
                    confidence = 'low'
                
                overlapping_speakers.append({
                    'speaker': speaker_seg['assigned_speaker'],
                    'overlap_duration': overlap_duration,
                    'overlap_ratio': overlap_ratio,
                    'confidence': confidence,
                    'speaker_start': speaker_start,
                    'speaker_end': speaker_end
                })
        
        # Sort by overlap duration (descending)
        overlapping_speakers.sort(key=lambda x: x['overlap_duration'], reverse=True)
        
        return overlapping_speakers
    
    def _compile_final_results(self, audio_file: str, speaker_results: Dict, 
                             whisper_result: Dict, aligned_segments: List[Dict]) -> Dict:
        """Compile all results into final format."""
        
        # Extract speaker information
        speakers_identified = []
        for detected_id, assignment in speaker_results.get('speaker_assignments', {}).items():
            speakers_identified.append({
                'detected_id': detected_id,
                'assigned_name': assignment['assigned_name'],
                'assignment_type': assignment['type'],
                'confidence': assignment.get('confidence', 'unknown'),
                'similarity': assignment.get('similarity', 0.0)
            })
        
        # Calculate statistics
        total_words = sum(seg.get('word_count', len(seg['text'].split())) for seg in aligned_segments)
        speaker_word_counts = {}
        speaker_speaking_time = {}
        
        for segment in aligned_segments:
            speaker = segment['speaker']
            word_count = segment.get('word_count', len(segment['text'].split()))
            duration = segment['duration']
            
            if speaker not in speaker_word_counts:
                speaker_word_counts[speaker] = 0
                speaker_speaking_time[speaker] = 0.0
            
            speaker_word_counts[speaker] += word_count
            speaker_speaking_time[speaker] += duration
        
        return {
            'file_info': {
                'audio_file': audio_file,
                'processed_at': datetime.now().isoformat(),
                'duration': whisper_result.get('duration', 0),
                'language': whisper_result.get('language', 'unknown')
            },
            'processing_info': {
                'whisper_model': str(self.whisper_model.dims.n_mels) + "_mel_bins",
                'speaker_system': 'SpeechBrain_ECAPA_TDNN',
                'quality_bypass_used': self.bypass_quality_filters,
                'word_timestamps': 'words' in aligned_segments[0] if aligned_segments else False
            },
            'speakers_identified': speakers_identified,
            'transcription': {
                'full_text': whisper_result.get('text', ''),
                'language': whisper_result.get('language', 'unknown'),
                'segments': aligned_segments
            },
            'statistics': {
                'total_segments': len(aligned_segments),
                'total_words': total_words,
                'unique_speakers': len(speaker_word_counts),
                'speaker_word_counts': speaker_word_counts,
                'speaker_speaking_time': speaker_speaking_time,
                'average_words_per_segment': total_words / len(aligned_segments) if aligned_segments else 0
            },
            'raw_results': {
                'speaker_identification': speaker_results,
                'whisper_transcription': whisper_result
            }
        }
    
    def save_simple_transcript(self, results: Dict, output_folder: str = "transcription_output") -> Path:
        """
        Save a simplified JSON transcript with just speaker, start, end, and text.
        
        Args:
            results: Results from transcribe_with_speakers
            output_folder: Folder to save results
            
        Returns:
            Path to the simple JSON file
        """
        output_path = Path(output_folder)
        output_path.mkdir(exist_ok=True)
        
        base_name = Path(results['file_info']['audio_file']).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create simple transcript format
        simple_transcript = {
            "file": results['file_info']['audio_file'],
            "duration": results['file_info']['duration'],
            "language": results['file_info']['language'],
            "segments": []
        }
        
        # Extract just the essential information
        for segment in results['transcription']['segments']:
            simple_segment = {
                "start": round(segment['start'], 2),
                "end": round(segment['end'], 2),
                "speaker": segment['speaker'],
                "text": segment['text'].strip()
            }
            simple_transcript["segments"].append(simple_segment)
        
        # Save simple JSON
        simple_file = output_path / f"{base_name}_simple_transcript_{timestamp}.json"
        with open(simple_file, 'w', encoding='utf-8') as f:
            json.dump(simple_transcript, f, indent=2, ensure_ascii=False)
        
        return simple_file
    
    def save_results(self, results: Dict, output_folder: str = "transcription_output"):
        """
        Save transcription results in multiple formats.
        
        Args:
            results: Results from transcribe_with_speakers
            output_folder: Folder to save results
        """
        output_path = Path(output_folder)
        output_path.mkdir(exist_ok=True)
        
        base_name = Path(results['file_info']['audio_file']).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Save complete results as JSON
        json_file = output_path / f"{base_name}_complete_results_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self._make_json_serializable(results), f, indent=2, ensure_ascii=False)
        
        # 2. Save speaker-labeled transcript (human-readable)
        txt_file = output_path / f"{base_name}_speaker_transcript_{timestamp}.txt"
        self._save_speaker_transcript_txt(results, txt_file)
        
        # 3. Save as SRT subtitle format
        srt_file = output_path / f"{base_name}_speaker_subtitles_{timestamp}.srt"
        self._save_speaker_srt(results, srt_file)
        
        # 4. Save timeline CSV
        csv_file = output_path / f"{base_name}_timeline_{timestamp}.csv"
        self._save_timeline_csv(results, csv_file)
        
        print(f"\n💾 Results saved to {output_folder}:")
        print(f"   📄 Complete results: {json_file.name}")
        print(f"   📝 Speaker transcript: {txt_file.name}")
        print(f"   🎬 SRT subtitles: {srt_file.name}")
        print(f"   📊 Timeline CSV: {csv_file.name}")
        
        return {
            'json_file': json_file,
            'txt_file': txt_file,
            'srt_file': srt_file,
            'csv_file': csv_file
        }
    
    def _save_speaker_transcript_txt(self, results: Dict, output_file: Path):
        """Save human-readable speaker transcript."""
        with open(output_file, 'w', encoding='utf-8') as f:
            # Header
            f.write("SPEAKER-IDENTIFIED TRANSCRIPT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"File: {results['file_info']['audio_file']}\n")
            f.write(f"Duration: {results['file_info']['duration']:.2f} seconds\n")
            f.write(f"Language: {results['file_info']['language']}\n")
            f.write(f"Processed: {results['file_info']['processed_at']}\n\n")
            
            # Speaker summary
            f.write("SPEAKERS IDENTIFIED:\n")
            f.write("-" * 20 + "\n")
            for speaker_info in results['speakers_identified']:
                f.write(f"• {speaker_info['assigned_name']} ({speaker_info['assignment_type']})\n")
            f.write("\n")
            
            # Statistics
            stats = results['statistics']
            f.write("STATISTICS:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total segments: {stats['total_segments']}\n")
            f.write(f"Total words: {stats['total_words']}\n")
            f.write(f"Unique speakers: {stats['unique_speakers']}\n\n")
            
            f.write("SPEAKING TIME BY SPEAKER:\n")
            for speaker, time_val in stats['speaker_speaking_time'].items():
                f.write(f"• {speaker}: {time_val:.1f}s\n")
            f.write("\n")
            
            # Transcript
            f.write("TRANSCRIPT:\n")
            f.write("=" * 50 + "\n\n")
            
            for segment in results['transcription']['segments']:
                start_time = self._format_timestamp(segment['start'])
                end_time = self._format_timestamp(segment['end'])
                speaker = segment['speaker']
                text = segment['text']
                confidence = segment.get('speaker_confidence', 'unknown')
                
                f.write(f"[{start_time} - {end_time}] {speaker}")
                if confidence != 'unknown':
                    f.write(f" ({confidence} confidence)")
                f.write(f":\n{text}\n\n")
    
    def _save_speaker_srt(self, results: Dict, output_file: Path):
        """Save SRT subtitle format with speaker labels."""
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(results['transcription']['segments'], 1):
                start_time = self._format_srt_timestamp(segment['start'])
                end_time = self._format_srt_timestamp(segment['end'])
                speaker = segment['speaker']
                text = segment['text']
                
                f.write(f"{i}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"<font color=\"#00FF00\">{speaker}:</font> {text}\n\n")
    
    def _save_timeline_csv(self, results: Dict, output_file: Path):
        """Save timeline as CSV."""
        import csv
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Start', 'End', 'Duration', 'Speaker', 'Text', 
                'Speaker_Confidence', 'Word_Count', 'Multiple_Speakers'
            ])
            
            for segment in results['transcription']['segments']:
                writer.writerow([
                    f"{segment['start']:.2f}",
                    f"{segment['end']:.2f}",
                    f"{segment['duration']:.2f}",
                    segment['speaker'],
                    segment['text'],
                    segment.get('speaker_confidence', ''),
                    segment.get('word_count', len(segment['text'].split())),
                    segment.get('multiple_speakers', False)
                ])
    
    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds as MM:SS or HH:MM:SS."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"
    
    def _format_srt_timestamp(self, seconds: float) -> str:
        """Format seconds as SRT timestamp (HH:MM:SS,mmm)."""
        td = timedelta(seconds=seconds)
        hours = int(td.total_seconds() // 3600)
        minutes = int((td.total_seconds() % 3600) // 60)
        secs = int(td.total_seconds() % 60)
        milliseconds = int((td.total_seconds() % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"
    
    def _make_json_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects to JSON-compatible format."""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif hasattr(obj, 'tolist'):  # numpy arrays
            return obj.tolist()
        elif hasattr(obj, 'item'):  # numpy scalars
            return obj.item()
        else:
            return obj


# Convenience functions for easy usage
def transcribe_with_speakers(audio_file: str, 
                           whisper_model: str = "base",
                           language: Optional[str] = None,
                           output_folder: str = "transcription_output",
                           simple_output: bool = True) -> Dict:
    """
    Convenience function to transcribe audio with speaker identification.
    
    Args:
        audio_file: Path to audio file
        whisper_model: Whisper model size
        language: Language for transcription (None for auto-detect)
        output_folder: Where to save results
        simple_output: If True, also creates a simple JSON with just speaker, start, end, text
        
    Returns:
        Complete transcription results with speaker identification
    """
    # Validate file path
    audio_path = Path(audio_file)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_file}")
    
    print(f"🎵 Processing: {audio_path.name}")
    print(f"📁 Full path: {audio_path.absolute()}")
    
    # Initialize system
    system = SpeakerAwareTranscriptionSystem(
        whisper_model_size=whisper_model,
        bypass_quality_filters=True  # Use the bypassed quality filters approach
    )
    
    # Process audio
    results = system.transcribe_with_speakers(
        str(audio_path),
        language=language,
        word_timestamps=True
    )
    
    # Save results
    saved_files = system.save_results(results, output_folder)
    
    # Create simple output if requested
    if simple_output:
        simple_file = system.save_simple_transcript(results, output_folder)
        saved_files['simple_json'] = simple_file
        print(f"   📋 Simple transcript: {simple_file.name}")
    
    return results


def quick_speaker_transcript(audio_file: str, output_format: str = "txt") -> str:
    """
    Quick function to get a speaker-labeled transcript.
    
    Args:
        audio_file: Path to audio file
        output_format: "txt", "srt", "json", or "simple"
        
    Returns:
        Path to output file
    """
    # Validate file path
    audio_path = Path(audio_file)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_file}")
    
    print(f"🎵 Processing: {audio_path.name}")
    
    system = SpeakerAwareTranscriptionSystem(bypass_quality_filters=True)
    results = system.transcribe_with_speakers(str(audio_path), word_timestamps=True)
    
    output_files = system.save_results(results)
    
    # Also create simple transcript for quick access
    simple_file = system.save_simple_transcript(results)
    
    if output_format == "txt":
        return str(output_files['txt_file'])
    elif output_format == "srt":
        return str(output_files['srt_file'])
    elif output_format == "json":
        return str(output_files['json_file'])
    elif output_format == "simple":
        return str(simple_file)
    else:
        return str(output_files['txt_file'])


# Example usage
if __name__ == "__main__":
    import sys
    
    print("🎙️ SPEAKER-AWARE TRANSCRIPTION SYSTEM")
    print("=" * 60)
    
    if len(sys.argv) < 2:
        print("Usage examples:")
        print("  python speaker_transcription_system.py /path/to/audio.wav")
        print("  python speaker_transcription_system.py audio.wav --language en")
        print("  python speaker_transcription_system.py /full/path/to/audio.wav --model large --output my_results")
        print("\nInteractive mode:")
        
        # Interactive file selection
        audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac']
        audio_files = []
        
        for ext in audio_extensions:
            audio_files.extend(Path('.').glob(f'*{ext}'))
            audio_files.extend(Path('.').glob(f'*{ext.upper()}'))
        
        if audio_files:
            print(f"\n📁 Found {len(audio_files)} audio files in current directory:")
            for i, file in enumerate(sorted(audio_files), 1):
                print(f"   {i:2d}. {file.name}")
            
            try:
                choice = input(f"\nSelect file (1-{len(audio_files)}) or enter full path: ").strip()
                
                if choice.isdigit():
                    file_index = int(choice) - 1
                    if 0 <= file_index < len(audio_files):
                        audio_file = str(sorted(audio_files)[file_index])
                    else:
                        print("❌ Invalid selection")
                        sys.exit(1)
                else:
                    # User entered a path
                    audio_file = choice
                    if not Path(audio_file).exists():
                        print(f"❌ File not found: {audio_file}")
                        sys.exit(1)
                        
            except (ValueError, KeyboardInterrupt):
                print("❌ Invalid input or cancelled")
                sys.exit(1)
        else:
            audio_file = input("Enter full path to audio file: ").strip()
            if not Path(audio_file).exists():
                print(f"❌ File not found: {audio_file}")
                sys.exit(1)
    else:
        audio_file = sys.argv[1]
    
    # Validate the file path
    audio_path = Path(audio_file)
    if not audio_path.exists():
        print(f"❌ Audio file not found: {audio_file}")
        print(f"   Checked path: {audio_path.absolute()}")
        sys.exit(1)
    
    # Parse arguments
    language = None
    model = "base"
    output_folder = "transcription_output"
    
    for i, arg in enumerate(sys.argv):
        if arg == "--language" and i + 1 < len(sys.argv):
            language = sys.argv[i + 1]
        elif arg == "--model" and i + 1 < len(sys.argv):
            model = sys.argv[i + 1]
        elif arg == "--output" and i + 1 < len(sys.argv):
            output_folder = sys.argv[i + 1]
    
    # Process the file
    try:
        print(f"\n🎵 Processing: {audio_path.name}")
        print(f"📁 Full path: {audio_path.absolute()}")
        print(f"🤖 Whisper model: {model}")
        if language:
            print(f"🌍 Language: {language}")
        else:
            print(f"🌍 Language: auto-detect")
        print(f"📂 Output folder: {output_folder}")
        
        results = transcribe_with_speakers(
            audio_file=str(audio_path),
            whisper_model=model,
            language=language,
            output_folder=output_folder,
            simple_output=True  # Creates both detailed and simple outputs
        )
        
        print(f"\n🎉 SUCCESS!")
        print(f"   📁 Results saved to: {output_folder}/")
        print(f"   🎤 Speakers identified: {results['statistics']['unique_speakers']}")
        print(f"   📝 Total words: {results['statistics']['total_words']}")
        print(f"   ⏱️  Duration: {results['file_info']['duration']:.2f} seconds")
        print(f"   🌍 Language detected: {results['file_info']['language']}")
        
        # Show speaker breakdown
        print(f"\n🎤 Speaker breakdown:")
        for speaker, word_count in results['statistics']['speaker_word_counts'].items():
            speaking_time = results['statistics']['speaker_speaking_time'][speaker]
            percentage = (speaking_time / results['file_info']['duration']) * 100
            print(f"   • {speaker}: {word_count} words ({speaking_time:.1f}s, {percentage:.1f}%)")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()