# speaker_saver.py
import pickle
import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

class SpeakerSaver:
    """
    Saves unidentified speaker embeddings to individual folders within the speakers directory.
    Each speaker gets their own folder containing their .pkl and .json files.
    """
    
    def __init__(self, speakers_folder: str = "speakers"):
        """
        Initialize the speaker saver.
        
        Args:
            speakers_folder: Path to folder for saving speaker embeddings
        """
        self.speakers_folder = Path(speakers_folder)
        self.speakers_folder.mkdir(exist_ok=True)
        
    def save_unidentified_speakers(self, embeddings: Dict, unmatched_speakers: List[str], 
                                 auto_name: bool = True, name_prefix: str = "Unknown") -> Dict:
        """
        Save unidentified speakers to individual folders within the speakers folder.
        
        Args:
            embeddings: Speaker embeddings dictionary
            unmatched_speakers: List of unmatched speaker IDs
            auto_name: Whether to auto-generate names for speakers
            name_prefix: Prefix for auto-generated names
            
        Returns:
            Dictionary with saved speaker information
        """
        saved_speakers = {}
        skipped_speakers = []
        
        for speaker_id in unmatched_speakers:
            if speaker_id not in embeddings['speakers']:
                print(f"Warning: Speaker {speaker_id} not found in embeddings")
                skipped_speakers.append(speaker_id)
                continue
            
            speaker_data = embeddings['speakers'][speaker_id]
            
            # Generate name for the speaker
            if auto_name:
                speaker_name = self._generate_speaker_name(name_prefix)
            else:
                speaker_name = self._prompt_for_speaker_name(speaker_id, speaker_data)
            
            # Prepare speaker data for saving
            save_data = self._prepare_speaker_data(speaker_data, embeddings['file_info'], speaker_id)
            
            # Save the speaker
            if self._save_speaker(speaker_name, save_data):
                saved_speakers[speaker_id] = {
                    'saved_as': speaker_name,
                    'total_speech_time': speaker_data['total_speech_time'],
                    'segment_count': speaker_data['segment_count']
                }
                print(f"Saved speaker {speaker_id} as '{speaker_name}'")
            else:
                skipped_speakers.append(speaker_id)
        
        return {
            'saved_speakers': saved_speakers,
            'skipped_speakers': skipped_speakers,
            'total_saved': len(saved_speakers),
            'total_skipped': len(skipped_speakers)
        }
    
    def save_speaker_with_name(self, embeddings: Dict, speaker_id: str, 
                             speaker_name: str, overwrite: bool = False) -> bool:
        """
        Save a specific speaker with a given name.
        
        Args:
            embeddings: Speaker embeddings dictionary
            speaker_id: ID of the speaker to save
            speaker_name: Name to save the speaker as
            overwrite: Whether to overwrite existing speaker
            
        Returns:
            True if saved successfully, False otherwise
        """
        if speaker_id not in embeddings['speakers']:
            print(f"Error: Speaker {speaker_id} not found in embeddings")
            return False
        
        speaker_folder = self.speakers_folder / speaker_name
        
        if speaker_folder.exists() and not overwrite:
            print(f"Error: Speaker '{speaker_name}' already exists. Use overwrite=True to replace.")
            return False
        
        speaker_data = embeddings['speakers'][speaker_id]
        save_data = self._prepare_speaker_data(speaker_data, embeddings['file_info'], speaker_id)
        
        return self._save_speaker(speaker_name, save_data)
    
    def _generate_speaker_name(self, prefix: str = "Unknown") -> str:
        """
        Generate a unique speaker name by checking for existing folders.
        
        Args:
            prefix: Prefix for the generated name
            
        Returns:
            Unique speaker name
        """
        counter = 1
        while True:
            candidate_name = f"{prefix}_{counter:03d}"
            speaker_folder = self.speakers_folder / candidate_name
            if not speaker_folder.exists():
                return candidate_name
            counter += 1
    
    def _prompt_for_speaker_name(self, speaker_id: str, speaker_data: Dict) -> str:
        """
        Prompt user for speaker name (fallback to auto-generation).
        
        Args:
            speaker_id: Original speaker ID
            speaker_data: Speaker data for context
            
        Returns:
            Speaker name
        """
        print(f"\nSpeaker {speaker_id} information:")
        print(f"  Total speech time: {speaker_data['total_speech_time']:.2f} seconds")
        print(f"  Number of segments: {speaker_data['segment_count']}")
        
        try:
            name = input(f"Enter name for speaker {speaker_id} (or press Enter for auto-name): ").strip()
            if name:
                # Check if folder already exists
                speaker_folder = self.speakers_folder / name
                if speaker_folder.exists():
                    overwrite = input(f"Speaker '{name}' already exists. Overwrite? (y/n): ").strip().lower()
                    if overwrite == 'y':
                        return name
                    else:
                        return self._generate_speaker_name("Unknown")
                return name
            else:
                return self._generate_speaker_name("Unknown")
        except (EOFError, KeyboardInterrupt):
            return self._generate_speaker_name("Unknown")
    
    def _prepare_speaker_data(self, speaker_data: Dict, file_info: Dict, original_id: str) -> Dict:
        """
        Prepare speaker data for saving.
        
        Args:
            speaker_data: Original speaker data
            file_info: File information
            original_id: Original speaker ID
            
        Returns:
            Prepared speaker data
        """
        prepared_data = {
            'embedding': speaker_data['embedding'],
            'total_samples': speaker_data['segment_count'],
            'total_speech_time': speaker_data['total_speech_time'],
            'segment_history': speaker_data['segments'],
            'source_files': [file_info['filename']],
            'original_id': original_id,
            'created_date': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            'update_count': 0,
            'file_count': 1,
            'metadata': {
                'extraction_info': {
                    'file_duration': file_info['duration'],
                    'sample_rate': file_info['sample_rate'],
                    'filepath': file_info['filepath']
                }
            }
        }
        
        return prepared_data
    
    def _save_speaker(self, speaker_name: str, speaker_data: Dict) -> bool:
        """
        Save speaker data to their individual folder.
        
        Args:
            speaker_name: Name of the speaker
            speaker_data: Speaker data to save
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Create speaker folder
            speaker_folder = self.speakers_folder / speaker_name
            speaker_folder.mkdir(exist_ok=True)
            
            # Save pickle file
            speaker_file = speaker_folder / f"{speaker_name}.pkl"
            with open(speaker_file, 'wb') as f:
                pickle.dump(speaker_data, f)
            
            # Save human-readable metadata
            metadata_file = speaker_folder / f"{speaker_name}_info.json"
            metadata = {
                'speaker_name': speaker_name,
                'total_samples': speaker_data['total_samples'],
                'total_speech_time': speaker_data['total_speech_time'],
                'source_files': speaker_data['source_files'],
                'file_count': speaker_data.get('file_count', 1),
                'created_date': speaker_data['created_date'],
                'last_updated': speaker_data['last_updated'],
                'update_count': speaker_data['update_count'],
                'embedding_shape': speaker_data['embedding'].shape,
                'original_id': speaker_data.get('original_id', 'Unknown'),
                'folder_path': str(speaker_folder.absolute())
            }
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✅ Created speaker folder: {speaker_folder}")
            print(f"   📁 Folder: {speaker_folder}")
            print(f"   🗃️  Pickle: {speaker_file.name}")
            print(f"   📄 Metadata: {metadata_file.name}")
            
            return True
            
        except Exception as e:
            print(f"Error saving speaker '{speaker_name}': {e}")
            return False
    
    def batch_save_speakers(self, embeddings: Dict, speaker_mappings: Dict, 
                           overwrite: bool = False) -> Dict:
        """
        Save multiple speakers with custom names.
        
        Args:
            embeddings: Speaker embeddings dictionary
            speaker_mappings: Dictionary mapping speaker_id to desired name
            overwrite: Whether to overwrite existing speakers
            
        Returns:
            Dictionary with save results
        """
        results = {
            'saved_speakers': {},
            'failed_speakers': [],
            'skipped_speakers': []
        }
        
        for speaker_id, speaker_name in speaker_mappings.items():
            if speaker_id not in embeddings['speakers']:
                results['failed_speakers'].append({
                    'speaker_id': speaker_id,
                    'reason': 'Speaker not found in embeddings'
                })
                continue
            
            speaker_folder = self.speakers_folder / speaker_name
            if speaker_folder.exists() and not overwrite:
                results['skipped_speakers'].append({
                    'speaker_id': speaker_id,
                    'speaker_name': speaker_name,
                    'reason': 'Speaker folder already exists'
                })
                continue
            
            if self.save_speaker_with_name(embeddings, speaker_id, speaker_name, overwrite):
                results['saved_speakers'][speaker_id] = speaker_name
            else:
                results['failed_speakers'].append({
                    'speaker_id': speaker_id,
                    'speaker_name': speaker_name,
                    'reason': 'Save operation failed'
                })
        
        return results
    
    def list_saved_speakers(self) -> List[Dict]:
        """
        List all saved speakers with their information by scanning speaker folders.
        
        Returns:
            List of speaker information dictionaries
        """
        speakers = []
        
        # Look for speaker folders
        for speaker_folder in self.speakers_folder.iterdir():
            if not speaker_folder.is_dir():
                continue
                
            speaker_name = speaker_folder.name
            
            try:
                # Try to load metadata file first
                metadata_file = speaker_folder / f"{speaker_name}_info.json"
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    speakers.append(metadata)
                else:
                    # Fallback to loading pickle file
                    speaker_file = speaker_folder / f"{speaker_name}.pkl"
                    if speaker_file.exists():
                        with open(speaker_file, 'rb') as f:
                            speaker_data = pickle.load(f)
                        
                        speakers.append({
                            'speaker_name': speaker_name,
                            'total_samples': speaker_data.get('total_samples', 0),
                            'total_speech_time': speaker_data.get('total_speech_time', 0),
                            'created_date': speaker_data.get('created_date', 'Unknown'),
                            'embedding_shape': speaker_data['embedding'].shape,
                            'folder_path': str(speaker_folder.absolute())
                        })
                    else:
                        print(f"Warning: No valid files found in speaker folder: {speaker_folder}")
                        
            except Exception as e:
                print(f"Error reading speaker folder {speaker_folder}: {e}")
        
        return sorted(speakers, key=lambda x: x.get('created_date', ''))
    
    def delete_speaker(self, speaker_name: str) -> bool:
        """
        Delete a saved speaker folder and all its contents.
        
        Args:
            speaker_name: Name of the speaker to delete
            
        Returns:
            True if deleted successfully, False otherwise
        """
        speaker_folder = self.speakers_folder / speaker_name
        
        if not speaker_folder.exists():
            print(f"Error: Speaker folder '{speaker_name}' not found")
            return False
        
        try:
            # Remove all files in the folder
            for file_path in speaker_folder.iterdir():
                if file_path.is_file():
                    file_path.unlink()
                elif file_path.is_dir():
                    # Remove subdirectories if any (recursive)
                    import shutil
                    shutil.rmtree(file_path)
            
            # Remove the folder itself
            speaker_folder.rmdir()
            print(f"Deleted speaker folder: {speaker_name}")
            return True
            
        except Exception as e:
            print(f"Error deleting speaker folder '{speaker_name}': {e}")
            return False
    
    def rename_speaker(self, old_name: str, new_name: str) -> bool:
        """
        Rename a saved speaker folder and update internal references.
        
        Args:
            old_name: Current speaker name
            new_name: New speaker name
            
        Returns:
            True if renamed successfully, False otherwise
        """
        old_speaker_folder = self.speakers_folder / old_name
        new_speaker_folder = self.speakers_folder / new_name
        
        if not old_speaker_folder.exists():
            print(f"Error: Speaker folder '{old_name}' not found")
            return False
        
        if new_speaker_folder.exists():
            print(f"Error: Speaker folder '{new_name}' already exists")
            return False
        
        try:
            # Rename the folder
            old_speaker_folder.rename(new_speaker_folder)
            
            # Update file names within the folder
            old_pkl_file = new_speaker_folder / f"{old_name}.pkl"
            new_pkl_file = new_speaker_folder / f"{new_name}.pkl"
            if old_pkl_file.exists():
                old_pkl_file.rename(new_pkl_file)
            
            old_json_file = new_speaker_folder / f"{old_name}_info.json"
            new_json_file = new_speaker_folder / f"{new_name}_info.json"
            if old_json_file.exists():
                old_json_file.rename(new_json_file)
                
                # Update metadata content
                with open(new_json_file, 'r') as f:
                    metadata = json.load(f)
                metadata['speaker_name'] = new_name
                metadata['folder_path'] = str(new_speaker_folder.absolute())
                with open(new_json_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
            
            print(f"Renamed speaker: {old_name} -> {new_name}")
            print(f"New folder: {new_speaker_folder}")
            return True
            
        except Exception as e:
            print(f"Error renaming speaker: {e}")
            return False
    
    def get_speaker_folder_path(self, speaker_name: str) -> Optional[Path]:
        """
        Get the folder path for a specific speaker.
        
        Args:
            speaker_name: Name of the speaker
            
        Returns:
            Path to speaker folder or None if not found
        """
        speaker_folder = self.speakers_folder / speaker_name
        return speaker_folder if speaker_folder.exists() else None
    
    def export_speakers_summary(self, output_file: str):
        """
        Export a summary of all saved speakers to JSON.
        
        Args:
            output_file: Path to output JSON file
        """
        speakers = self.list_saved_speakers()
        
        summary = {
            'export_date': datetime.now().isoformat(),
            'total_speakers': len(speakers),
            'speakers_folder': str(self.speakers_folder.absolute()),
            'folder_structure': 'individual_folders',
            'speakers': speakers
        }
        
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Speakers summary exported to: {output_file}")
        print(f"Total speakers: {len(speakers)}")
        return summary
    
    def backup_all_speakers(self, backup_dir: str = "speakers_backup") -> str:
        """
        Create a backup of all speaker folders.
        
        Args:
            backup_dir: Directory to save backup
            
        Returns:
            Path to the backup folder
        """
        import shutil
        
        backup_path = Path(backup_dir)
        backup_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_folder = backup_path / f"backup_{timestamp}"
        backup_folder.mkdir(exist_ok=True)
        
        # Copy all speaker folders
        copied_speakers = 0
        for speaker_folder in self.speakers_folder.iterdir():
            if speaker_folder.is_dir():
                try:
                    backup_speaker_folder = backup_folder / speaker_folder.name
                    shutil.copytree(speaker_folder, backup_speaker_folder)
                    copied_speakers += 1
                except Exception as e:
                    print(f"Error backing up {speaker_folder}: {e}")
        
        # Create backup manifest
        manifest = {
            'backup_date': datetime.now().isoformat(),
            'source_folder': str(self.speakers_folder.absolute()),
            'speakers_backed_up': copied_speakers,
            'backup_folder': str(backup_folder.absolute()),
            'folder_structure': 'individual_folders'
        }
        
        manifest_file = backup_folder / "backup_manifest.json"
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"Backup created at: {backup_folder}")
        print(f"Backed up {copied_speakers} speaker folders")
        return str(backup_folder)