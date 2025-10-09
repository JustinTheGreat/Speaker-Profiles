# auto_speaker_tagging_system.py
"""
Automatic speaker tagging and profile creation system using SpeechBrain ECAPA-TDNN.

This system:
1. Extracts speakers from audio files
2. Automatically tags them if they match existing speaker profiles
3. Creates new profiles for unknown speakers in individual folders
4. Updates existing profiles with new audio data

NEW FOLDER STRUCTURE:
speakers/
├── Speaker_001/
│   ├── Speaker_001.pkl
│   └── Speaker_001_info.json
├── Speaker_002/
│   ├── Speaker_002.pkl
│   └── Speaker_002_info.json
└── John_Doe/
    ├── John_Doe.pkl
    └── John_Doe_info.json
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import numpy as np

from speaker_embedding_extractor import SpeechBrainSpeakerEmbeddingExtractor
from speaker_matcher import SpeakerMatcher
from embedding_learner import EmbeddingLearner
from speaker_saver import SpeakerSaver

class AutoSpeakerTaggingSystem:
    """
    Automatic speaker tagging and profile creation system with individual folder structure.
    Each speaker gets their own folder containing their .pkl and .json files.
    """
    
    def __init__(self, 
                 speakers_folder: str = "speakers",
                 similarity_threshold: float = 0.75,
                 min_quality: float = 0.3,
                 min_speech_time: float = 2.0,
                 learning_rate: float = 0.7,
                 auto_save_unknown: bool = True):
        """
        Initialize the automatic speaker tagging system.
        
        Args:
            speakers_folder: Folder containing individual speaker folders
            similarity_threshold: Minimum similarity to match existing speakers (0.0-1.0)
            min_quality: Minimum quality threshold for processing speakers
            min_speech_time: Minimum speech time required (seconds)
            learning_rate: Rate for updating existing speaker profiles
            auto_save_unknown: Automatically save unknown speakers
        """
        self.speakers_folder = Path(speakers_folder)
        self.speakers_folder.mkdir(exist_ok=True)
        
        self.similarity_threshold = similarity_threshold
        self.min_quality = min_quality
        self.min_speech_time = min_speech_time
        self.learning_rate = learning_rate
        self.auto_save_unknown = auto_save_unknown
        
        # Initialize components
        print("🎙️ Initializing automatic speaker tagging system...")
        print(f"📁 Using folder structure: {speakers_folder}/[speaker_name]/")
        
        self.extractor = SpeechBrainSpeakerEmbeddingExtractor()
        self.matcher = SpeakerMatcher(speakers_folder)
        self.learner = EmbeddingLearner(speakers_folder)
        self.saver = SpeakerSaver(speakers_folder)
        
        # Check for old file structure and offer migration
        self._check_for_migration_needed()
        
        print(f"✅ System ready! Using similarity threshold: {similarity_threshold}")
        print(f"   Known speakers: {len(self.matcher.get_known_speaker_list())}")
    
    def _check_for_migration_needed(self):
        """Check if migration from old file structure is needed."""
        # Look for old-style files in the speakers folder root
        old_pkl_files = list(self.speakers_folder.glob("*.pkl"))
        if old_pkl_files:
            print(f"\n⚠️  MIGRATION NOTICE:")
            print(f"Found {len(old_pkl_files)} speaker files using old structure")
            print(f"Consider running migrate_to_folder_structure() to update to new folder structure")
            print(f"Old files: {[f.name for f in old_pkl_files[:3]]}")
            if len(old_pkl_files) > 3:
                print(f"... and {len(old_pkl_files) - 3} more")
    
    def migrate_to_folder_structure(self, dry_run: bool = True) -> Dict:
        """
        Migrate speakers from old flat file structure to new folder structure.
        
        Args:
            dry_run: If True, only show what would be migrated without actually doing it
            
        Returns:
            Migration report
        """
        print("🔄 MIGRATION TO FOLDER STRUCTURE")
        print("=" * 50)
        
        migration_report = self.learner.migrate_old_structure(dry_run=dry_run)
        
        if not dry_run and migration_report['migrated_speakers']:
            # Reload the matcher after migration
            print("\n🔄 Reloading speaker matcher after migration...")
            self.matcher.reload_known_speakers()
        
        return migration_report
    
    def process_audio_file(self, audio_file: str, 
                          custom_unknown_prefix: str = "Speaker",
                          update_existing: bool = True) -> Dict:
        """
        Process an audio file: tag known speakers and create profiles for unknown ones.
        
        Args:
            audio_file: Path to audio file
            custom_unknown_prefix: Prefix for naming unknown speakers
            update_existing: Whether to update existing speaker profiles
            
        Returns:
            Dictionary with processing results and speaker assignments
        """
        audio_path = Path(audio_file)
        print(f"\n🎙️ Processing: {audio_path.name}")
        print("="*60)
        
        # Step 1: Extract speakers
        print("1️⃣ Extracting speakers with SpeechBrain ECAPA-TDNN...")
        embeddings = self.extractor.extract_speaker_embeddings(audio_file)
        
        detected_speakers = len(embeddings['speakers'])
        print(f"   🔍 Detected {detected_speakers} speaker(s)")
        
        # Step 2: Filter by quality and speech time
        print("2️⃣ Filtering speakers by quality...")
        qualified_speakers = self._filter_speakers_by_quality(embeddings)
        
        if not qualified_speakers:
            print("   ❌ No speakers meet quality criteria!")
            return self._create_empty_result(audio_file, embeddings)
        
        print(f"   ✅ {len(qualified_speakers)} speaker(s) meet quality criteria")
        
        # Step 3: Match with existing speakers
        print("3️⃣ Matching with existing speaker profiles...")
        match_results = self.matcher.match_speakers(embeddings, self.similarity_threshold)
        
        matched_count = len(match_results['matches'])
        unmatched_count = len(match_results['unmatched_speakers'])
        print(f"   🎯 Matched: {matched_count}, Unknown: {unmatched_count}")
        
        # Step 4: Update existing speaker profiles
        learning_results = {}
        if update_existing and match_results['matches']:
            print("4️⃣ Updating existing speaker profiles...")
            learning_results = self.learner.learn_from_embeddings(
                embeddings, match_results, self.learning_rate
            )
            updated_count = len(learning_results.get('updated_speakers', []))
            print(f"   📚 Updated {updated_count} existing speaker profile(s)")
            
            # Show which folders were updated
            for speaker in learning_results.get('updated_speakers', []):
                folder_path = self.speakers_folder / speaker
                print(f"     📁 {speaker}: {folder_path}")
        else:
            print("4️⃣ Skipping profile updates")
        
        # Step 5: Create new profiles for unknown speakers
        save_results = {}
        if self.auto_save_unknown and match_results['unmatched_speakers']:
            print("5️⃣ Creating new speaker profiles...")
            
            # Filter unknown speakers by quality
            quality_unknown = [
                speaker for speaker in match_results['unmatched_speakers']
                if speaker in qualified_speakers
            ]
            
            if quality_unknown:
                save_results = self.saver.save_unidentified_speakers(
                    embeddings, quality_unknown, auto_name=True, name_prefix=custom_unknown_prefix
                )
                saved_count = len(save_results.get('saved_speakers', {}))
                print(f"   💾 Created {saved_count} new speaker profile(s)")
                
                # Show new folders created
                for speaker_id, save_info in save_results.get('saved_speakers', {}).items():
                    speaker_name = save_info['saved_as']
                    folder_path = self.speakers_folder / speaker_name
                    print(f"     📁 {speaker_name}: {folder_path}")
                
                # Reload known speakers after saving new ones
                self.matcher.reload_known_speakers()
            else:
                print("   ⚠️  No unknown speakers meet quality criteria")
        else:
            print("5️⃣ Skipping new profile creation")
        
        # Step 6: Create final speaker assignments
        print("6️⃣ Creating final speaker assignments...")
        assignments = self._create_speaker_assignments(
            embeddings, match_results, save_results, qualified_speakers
        )
        
        # Step 7: Generate comprehensive results
        results = self._create_comprehensive_results(
            audio_file, embeddings, match_results, learning_results, 
            save_results, assignments, qualified_speakers
        )
        
        # Step 8: Display summary
        self._display_summary(results)
        
        return results
    
    def _filter_speakers_by_quality(self, embeddings: Dict) -> List[str]:
        """Filter speakers by quality and speech time criteria."""
        qualified_speakers = []
        
        for speaker_id, data in embeddings['speakers'].items():
            quality = data['average_quality']
            speech_time = data['total_speech_time']
            
            meets_quality = quality >= self.min_quality
            meets_time = speech_time >= self.min_speech_time
            
            if meets_quality and meets_time:
                qualified_speakers.append(speaker_id)
                print(f"   ✅ {speaker_id}: quality={quality:.3f}, time={speech_time:.1f}s")
            else:
                reasons = []
                if not meets_quality:
                    reasons.append(f"low quality ({quality:.3f})")
                if not meets_time:
                    reasons.append(f"short speech ({speech_time:.1f}s)")
                print(f"   ❌ {speaker_id}: {', '.join(reasons)}")
        
        return qualified_speakers
    
    def _create_speaker_assignments(self, embeddings: Dict, match_results: Dict, 
                                   save_results: Dict, qualified_speakers: List[str]) -> Dict:
        """Create final speaker assignments mapping detected speakers to known names."""
        assignments = {}
        
        # Assign matched speakers
        for detected_id, match_info in match_results['matches'].items():
            if detected_id in qualified_speakers:
                known_name = match_info['matched_speaker']
                similarity = match_info['similarity']
                speaker_folder = match_info.get('speaker_folder', 'Unknown')
                assignments[detected_id] = {
                    'assigned_name': known_name,
                    'type': 'matched',
                    'similarity': similarity,
                    'confidence': 'high' if similarity > 0.85 else 'medium',
                    'speaker_folder': speaker_folder
                }
        
        # Assign newly saved speakers
        if save_results and 'saved_speakers' in save_results:
            for detected_id, save_info in save_results['saved_speakers'].items():
                if detected_id in qualified_speakers:
                    new_name = save_info['saved_as']
                    speaker_folder = str(self.speakers_folder / new_name)
                    assignments[detected_id] = {
                        'assigned_name': new_name,
                        'type': 'new_profile',
                        'similarity': 0.0,
                        'confidence': 'new',
                        'speaker_folder': speaker_folder
                    }
        
        # Handle unqualified or unprocessed speakers
        for speaker_id in embeddings['speakers']:
            if speaker_id not in assignments:
                if speaker_id not in qualified_speakers:
                    assignments[speaker_id] = {
                        'assigned_name': f"Filtered_{speaker_id}",
                        'type': 'filtered',
                        'similarity': 0.0,
                        'confidence': 'low',
                        'speaker_folder': None
                    }
                else:
                    assignments[speaker_id] = {
                        'assigned_name': f"Unprocessed_{speaker_id}",
                        'type': 'unprocessed',
                        'similarity': 0.0,
                        'confidence': 'unknown',
                        'speaker_folder': None
                    }
        
        return assignments
    
    def _create_comprehensive_results(self, audio_file: str, embeddings: Dict, 
                                     match_results: Dict, learning_results: Dict,
                                     save_results: Dict, assignments: Dict, 
                                     qualified_speakers: List[str]) -> Dict:
        """Create comprehensive results dictionary."""
        
        # Create labeled timeline
        labeled_timeline = []
        for segment in embeddings['diarization']:
            detected_id = segment['speaker']
            assignment = assignments.get(detected_id, {})
            assigned_name = assignment.get('assigned_name', detected_id)
            confidence = assignment.get('confidence', 'unknown')
            speaker_folder = assignment.get('speaker_folder')
            
            labeled_timeline.append({
                'start': segment['start'],
                'end': segment['end'],
                'duration': segment['duration'],
                'detected_speaker': detected_id,
                'assigned_speaker': assigned_name,
                'assignment_type': assignment.get('type', 'unprocessed'),
                'confidence': confidence,
                'speaker_folder': speaker_folder,
                'start_time': self._format_timestamp(segment['start']),
                'end_time': self._format_timestamp(segment['end'])
            })
        
        # Summary statistics
        summary = {
            'total_detected': len(embeddings['speakers']),
            'qualified_speakers': len(qualified_speakers),
            'matched_existing': len([a for a in assignments.values() if a['type'] == 'matched']),
            'new_profiles_created': len([a for a in assignments.values() if a['type'] == 'new_profile']),
            'filtered_out': len([a for a in assignments.values() if a['type'] == 'filtered']),
            'updated_profiles': len(learning_results.get('updated_speakers', [])),
            'total_speech_time': sum(data['total_speech_time'] for data in embeddings['speakers'].values()),
            'average_quality': np.mean([data['average_quality'] for data in embeddings['speakers'].values()])
        }
        
        return {
            'file_info': {
                'audio_file': audio_file,
                'processed_at': datetime.now().isoformat(),
                'duration': embeddings['file_info']['duration'],
                'sample_rate': embeddings['file_info']['sample_rate']
            },
            'system_config': {
                'similarity_threshold': self.similarity_threshold,
                'min_quality': self.min_quality,
                'min_speech_time': self.min_speech_time,
                'learning_rate': self.learning_rate,
                'auto_save_unknown': self.auto_save_unknown,
                'speakers_folder': str(self.speakers_folder.absolute()),
                'folder_structure': 'individual_folders'
            },
            'summary': summary,
            'speaker_assignments': assignments,
            'labeled_timeline': labeled_timeline,
            'qualified_speakers': qualified_speakers,
            'detailed_results': {
                'embeddings': embeddings,
                'match_results': match_results,
                'learning_results': learning_results,
                'save_results': save_results
            }
        }
    
    def _create_empty_result(self, audio_file: str, embeddings: Dict) -> Dict:
        """Create empty result when no speakers qualify."""
        return {
            'file_info': {
                'audio_file': audio_file,
                'processed_at': datetime.now().isoformat(),
                'duration': embeddings['file_info']['duration']
            },
            'system_config': {
                'speakers_folder': str(self.speakers_folder.absolute()),
                'folder_structure': 'individual_folders'
            },
            'summary': {
                'total_detected': len(embeddings['speakers']),
                'qualified_speakers': 0,
                'matched_existing': 0,
                'new_profiles_created': 0,
                'filtered_out': len(embeddings['speakers'])
            },
            'speaker_assignments': {},
            'labeled_timeline': [],
            'qualified_speakers': []
        }
    
    def _display_summary(self, results: Dict):
        """Display processing summary."""
        print("\n📊 PROCESSING SUMMARY")
        print("="*40)
        
        summary = results['summary']
        print(f"🔍 Total speakers detected: {summary['total_detected']}")
        print(f"✅ Qualified speakers: {summary['qualified_speakers']}")
        print(f"🎯 Matched to existing: {summary['matched_existing']}")
        print(f"🆕 New profiles created: {summary['new_profiles_created']}")
        print(f"📚 Profiles updated: {summary['updated_profiles']}")
        print(f"🚫 Filtered out: {summary['filtered_out']}")
        
        if 'average_quality' in summary:
            print(f"🏆 Average quality: {summary['average_quality']:.3f}")
        
        # Show speaker assignments with folder paths
        if results['speaker_assignments']:
            print(f"\n🏷️  SPEAKER ASSIGNMENTS:")
            for detected_id, assignment in results['speaker_assignments'].items():
                assigned_name = assignment['assigned_name']
                assignment_type = assignment['type']
                confidence = assignment['confidence']
                speaker_folder = assignment.get('speaker_folder')
                
                if assignment_type == 'matched':
                    similarity = assignment['similarity']
                    icon = "🎯"
                    details = f"(similarity: {similarity:.3f})"
                elif assignment_type == 'new_profile':
                    icon = "🆕"
                    details = "(new profile created)"
                elif assignment_type == 'filtered':
                    icon = "🚫"
                    details = "(filtered - low quality/time)"
                else:
                    icon = "❓"
                    details = "(unprocessed)"
                
                print(f"   {icon} {detected_id} → {assigned_name} {details}")
                if speaker_folder:
                    print(f"      📁 {speaker_folder}")
    
    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds as MM:SS or HH:MM:SS."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"
    
    def get_speaker_database_info(self) -> Dict:
        """Get information about the current speaker database."""
        speakers = self.saver.list_saved_speakers()
        
        if not speakers:
            return {
                'total_speakers': 0,
                'speakers': [],
                'total_speech_time': 0,
                'folder_structure': 'individual_folders',
                'speakers_folder': str(self.speakers_folder.absolute()),
                'message': 'No speakers in database'
            }
        
        total_speech_time = sum(s.get('total_speech_time', 0) for s in speakers)
        
        return {
            'total_speakers': len(speakers),
            'speakers': [s['speaker_name'] for s in speakers],
            'total_speech_time': total_speech_time,
            'average_speech_time': total_speech_time / len(speakers),
            'folder_structure': 'individual_folders',
            'speakers_folder': str(self.speakers_folder.absolute()),
            'speaker_details': speakers
        }
    
    def list_speaker_folders(self) -> List[Dict]:
        """List all speaker folders and their contents."""
        return self.matcher.list_speaker_folders()
    
    def verify_folder_structure(self) -> Dict:
        """Verify the integrity of the speaker folder structure."""
        return self.matcher.verify_folder_structure()
    
    def process_multiple_files(self, audio_files: List[str], 
                              progress_callback=None) -> Dict[str, Dict]:
        """
        Process multiple audio files in batch.
        
        Args:
            audio_files: List of audio file paths
            progress_callback: Optional callback function for progress updates
            
        Returns:
            Dictionary mapping file names to results
        """
        print(f"🎵 Processing {len(audio_files)} audio files...")
        
        all_results = {}
        
        for i, audio_file in enumerate(audio_files, 1):
            print(f"\n--- Processing file {i}/{len(audio_files)} ---")
            
            try:
                results = self.process_audio_file(audio_file)
                all_results[audio_file] = results
                
                if progress_callback:
                    progress_callback(i, len(audio_files), audio_file, results)
                    
            except Exception as e:
                print(f"❌ Error processing {audio_file}: {e}")
                all_results[audio_file] = {
                    'error': str(e),
                    'file_info': {'audio_file': audio_file}
                }
        
        # Overall summary
        total_detected = sum(r.get('summary', {}).get('total_detected', 0) for r in all_results.values() if 'error' not in r)
        total_matched = sum(r.get('summary', {}).get('matched_existing', 0) for r in all_results.values() if 'error' not in r)
        total_new = sum(r.get('summary', {}).get('new_profiles_created', 0) for r in all_results.values() if 'error' not in r)
        
        print(f"\n🎯 BATCH PROCESSING SUMMARY")
        print(f"Files processed: {len(audio_files)}")
        print(f"Total speakers detected: {total_detected}")
        print(f"Total matched to existing: {total_matched}")
        print(f"Total new profiles created: {total_new}")
        print(f"Speaker database folder: {self.speakers_folder}")
        
        return all_results
    
    def save_results(self, results: Dict, output_folder: str = "output"):
        """Save processing results to files."""
        output_path = Path(output_folder)
        output_path.mkdir(exist_ok=True)
        
        audio_file = results['file_info']['audio_file']
        base_name = Path(audio_file).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save complete results
        results_file = output_path / f"{base_name}_speaker_assignments_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(self._make_json_serializable(results), f, indent=2)
        
        # Save timeline CSV
        timeline_file = output_path / f"{base_name}_timeline_{timestamp}.csv"
        with open(timeline_file, 'w') as f:
            f.write("Start,End,Duration,DetectedSpeaker,AssignedSpeaker,Type,Confidence,SpeakerFolder\n")
            for segment in results['labeled_timeline']:
                f.write(f"{segment['start']:.2f},{segment['end']:.2f},{segment['duration']:.2f},"
                       f"{segment['detected_speaker']},{segment['assigned_speaker']},"
                       f"{segment['assignment_type']},{segment['confidence']},"
                       f"{segment.get('speaker_folder', '')}\n")
        
        print(f"💾 Results saved:")
        print(f"   - Complete results: {results_file}")
        print(f"   - Timeline CSV: {timeline_file}")
        
        return {'results_file': results_file, 'timeline_file': timeline_file}
    
    def _make_json_serializable(self, obj):
        """Convert numpy arrays to JSON-compatible format."""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif hasattr(obj, 'tolist'):
            return obj.tolist()
        elif hasattr(obj, 'item'):
            return obj.item()
        else:
            return obj


def main():
    """Command-line interface for the autom  atic speaker tagging system."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Automatic Speaker Tagging and Profile Creation System (Folder Structure)")
    parser.add_argument("audio_file", nargs='?', help="Path to audio file to process")
    parser.add_argument("--speakers-folder", default="speakers", 
                       help="Folder containing individual speaker folders")
    parser.add_argument("--similarity-threshold", type=float, default=0.75,
                       help="Similarity threshold for matching (0.0-1.0)")
    parser.add_argument("--min-quality", type=float, default=0.3,
                       help="Minimum quality threshold (0.0-1.0)")
    parser.add_argument("--min-speech-time", type=float, default=2.0,
                       help="Minimum speech time in seconds")
    parser.add_argument("--no-auto-save", action="store_true",
                       help="Don't automatically save unknown speakers")
    parser.add_argument("--no-update", action="store_true",
                       help="Don't update existing speaker profiles")
    parser.add_argument("--unknown-prefix", default="Speaker",
                       help="Prefix for naming unknown speakers")
    parser.add_argument("--save-results", action="store_true",
                       help="Save results to output folder")
    parser.add_argument("--migrate", action="store_true",
                       help="Migrate from old flat file structure to new folder structure")
    parser.add_argument("--migrate-dry-run", action="store_true",
                       help="Show what would be migrated without actually doing it")
    parser.add_argument("--verify-folders", action="store_true",
                       help="Verify the integrity of speaker folder structure")
    parser.add_argument("--list-speakers", action="store_true",
                       help="List all speaker folders and their contents")
    
    args = parser.parse_args()
    
    # Initialize system
    system = AutoSpeakerTaggingSystem(
        speakers_folder=args.speakers_folder,
        similarity_threshold=args.similarity_threshold,
        min_quality=args.min_quality,
        min_speech_time=args.min_speech_time,
        auto_save_unknown=not args.no_auto_save
    )
    
    # Handle special commands
    if args.migrate or args.migrate_dry_run:
        migration_report = system.migrate_to_folder_structure(dry_run=args.migrate_dry_run)
        if args.migrate_dry_run:
            print("\nTo perform actual migration, run with --migrate")
        return 0
    
    if args.verify_folders:
        print("🔍 VERIFYING SPEAKER FOLDER STRUCTURE")
        print("=" * 50)
        verification_report = system.verify_folder_structure()
        
        print(f"Total folders: {verification_report['total_folders']}")
        print(f"Valid folders: {verification_report['valid_folders']}")
        print(f"Loaded speakers: {verification_report['loaded_speakers']}")
        
        if verification_report['invalid_folders']:
            print(f"\n❌ Invalid folders: {verification_report['invalid_folders']}")
        
        if verification_report['missing_files']:
            print(f"\n⚠️  Missing files:")
            for missing_file in verification_report['missing_files']:
                print(f"   - {missing_file}")
        
        if verification_report['valid_folders'] == verification_report['total_folders']:
            print("\n✅ All speaker folders are valid!")
        
        return 0
    
    if args.list_speakers:
        print("📁 SPEAKER FOLDERS")
        print("=" * 50)
        folders = system.list_speaker_folders()
        
        for folder_info in folders:
            speaker_name = folder_info['speaker_name']
            is_loaded = "✅" if folder_info['is_loaded'] else "❌"
            print(f"{is_loaded} {speaker_name}")
            print(f"   📁 {folder_info['folder_path']}")
            print(f"   📄 Files: {len(folder_info['files'])}")
            for file_info in folder_info['files']:
                size_kb = file_info['size_bytes'] / 1024
                print(f"      - {file_info['filename']} ({size_kb:.1f} KB)")
            print()
        
        return 0
    
    if not args.audio_file:
        print("❌ No audio file specified. Use --help for usage information.")
        return 1
    
    # Show database info
    db_info = system.get_speaker_database_info()
    print(f"📂 Speaker database: {db_info['total_speakers']} known speakers")
    print(f"   📁 Structure: {db_info['folder_structure']}")
    print(f"   📍 Location: {db_info['speakers_folder']}")
    
    if db_info['total_speakers'] > 0:
        print(f"   Known speakers: {', '.join(db_info['speakers'][:5])}")
        if len(db_info['speakers']) > 5:
            print(f"   ... and {len(db_info['speakers']) - 5} more")
    
    # Process audio file
    try:
        results = system.process_audio_file(
            args.audio_file,
            custom_unknown_prefix=args.unknown_prefix,
            update_existing=not args.no_update
        )
        
        # Save results if requested
        if args.save_results:
            system.save_results(results)
        
        print(f"\n✅ Processing complete!")
        print(f"📁 Speaker folders location: {system.speakers_folder}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())