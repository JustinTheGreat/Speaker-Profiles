# embedding_learner.py
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import json
import shutil

class EmbeddingLearner:
    """
    Learns and updates speaker embeddings by merging new embeddings with existing ones.
    Works with individual speaker folders structure.
    """
    
    def __init__(self, speakers_folder: str = "speakers"):
        """
        Initialize the embedding learner.
        
        Args:
            speakers_folder: Path to folder containing individual speaker folders
        """
        self.speakers_folder = Path(speakers_folder)
        self.speakers_folder.mkdir(exist_ok=True)
        
    def learn_from_embeddings(self, new_embeddings: Dict, match_results: Dict, 
                            learning_rate: float = 0.7, min_samples: int = 3) -> Dict:
        """
        Learn and update speaker embeddings from new data.
        
        Args:
            new_embeddings: New speaker embeddings to learn from
            match_results: Results from speaker matching
            learning_rate: Weight for new embeddings (0.0 = ignore new, 1.0 = only new)
            min_samples: Minimum number of samples before updating embedding
            
        Returns:
            Dictionary with learning results
        """
        learning_results = {
            'updated_speakers': [],
            'skipped_speakers': [],
            'learning_stats': {}
        }
        
        # Process matched speakers (update existing embeddings)
        for original_speaker, match_info in match_results['matches'].items():
            matched_speaker = match_info['matched_speaker']
            new_embedding = new_embeddings['speakers'][original_speaker]['embedding']
            new_segments = new_embeddings['speakers'][original_speaker]['segments']
            
            # Load existing speaker data from their folder
            speaker_folder = self.speakers_folder / matched_speaker
            speaker_file = speaker_folder / f"{matched_speaker}.pkl"
            
            if speaker_file.exists():
                existing_data = self._load_speaker_data(speaker_file)
                
                # Check if we have enough samples to update
                total_segments = len(new_segments) + existing_data.get('total_samples', 0)
                
                if len(new_segments) >= min_samples or total_segments >= min_samples:
                    # Calculate adaptive learning rate if desired
                    existing_samples = existing_data.get('total_samples', 0)
                    adaptive_rate = self.adaptive_learning_rate(
                        existing_samples, len(new_segments), learning_rate
                    )
                    
                    # Update embedding using weighted average
                    existing_embedding = existing_data['embedding']
                    updated_embedding = self._weighted_average_embedding(
                        existing_embedding, new_embedding, adaptive_rate
                    )
                    
                    # Update speaker data
                    updated_data = self._merge_speaker_data(
                        existing_data, 
                        new_embeddings['speakers'][original_speaker],
                        updated_embedding,
                        new_embeddings['file_info']
                    )
                    
                    # Save updated data to the speaker's folder
                    self._save_speaker_data(matched_speaker, updated_data, speaker_folder)
                    learning_results['updated_speakers'].append(matched_speaker)
                    
                    learning_results['learning_stats'][matched_speaker] = {
                        'old_samples': existing_data.get('total_samples', 0),
                        'new_samples': len(new_segments),
                        'total_samples': updated_data['total_samples'],
                        'learning_rate_used': adaptive_rate,
                        'similarity_score': match_info['similarity'],
                        'embedding_change': self._calculate_embedding_change(
                            existing_embedding, updated_embedding
                        ),
                        'speaker_folder': str(speaker_folder.absolute())
                    }
                    
                    print(f"Updated speaker '{matched_speaker}' with {len(new_segments)} new samples")
                    print(f"  Learning rate used: {adaptive_rate:.3f}")
                    print(f"  Embedding change: {learning_results['learning_stats'][matched_speaker]['embedding_change']:.6f}")
                    print(f"  Folder: {speaker_folder}")
                else:
                    learning_results['skipped_speakers'].append(matched_speaker)
                    print(f"Skipped updating '{matched_speaker}' (insufficient samples: {len(new_segments)})")
            else:
                print(f"Warning: Speaker file for '{matched_speaker}' not found at {speaker_file}")
                learning_results['skipped_speakers'].append(matched_speaker)
        
        return learning_results
    
    def _weighted_average_embedding(self, existing_embedding: np.ndarray, 
                                  new_embedding: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Compute weighted average of embeddings.
        
        Args:
            existing_embedding: Current embedding
            new_embedding: New embedding to incorporate
            learning_rate: Weight for new embedding
            
        Returns:
            Updated embedding
        """
        # Ensure embeddings have the same shape
        if existing_embedding.shape != new_embedding.shape:
            print(f"Warning: Embedding shape mismatch: {existing_embedding.shape} vs {new_embedding.shape}")
            # Flatten both to ensure compatibility
            existing_embedding = existing_embedding.flatten()
            new_embedding = new_embedding.flatten()
            
            # Take the minimum length to avoid index errors
            min_length = min(len(existing_embedding), len(new_embedding))
            existing_embedding = existing_embedding[:min_length]
            new_embedding = new_embedding[:min_length]
        
        return (1 - learning_rate) * existing_embedding + learning_rate * new_embedding
    
    def _merge_speaker_data(self, existing_data: Dict, new_data: Dict, 
                          updated_embedding: np.ndarray, file_info: Dict) -> Dict:
        """
        Merge existing speaker data with new data.
        
        Args:
            existing_data: Existing speaker data
            new_data: New speaker data
            updated_embedding: Updated embedding vector
            file_info: Information about the source file
            
        Returns:
            Merged speaker data
        """
        merged_data = existing_data.copy()
        
        # Update embedding
        merged_data['embedding'] = updated_embedding
        
        # Update statistics
        merged_data['total_samples'] = existing_data.get('total_samples', 0) + len(new_data['segments'])
        merged_data['total_speech_time'] = existing_data.get('total_speech_time', 0) + new_data['total_speech_time']
        
        # Add new segments to history (keep limited history)
        max_history = 100
        if 'segment_history' not in merged_data:
            merged_data['segment_history'] = []
        
        # Add new segments with source file information
        new_segments_with_source = []
        for segment in new_data['segments']:
            segment_with_source = segment.copy()
            segment_with_source['source_file'] = file_info['filename']
            segment_with_source['added_date'] = datetime.now().isoformat()
            new_segments_with_source.append(segment_with_source)
        
        merged_data['segment_history'].extend(new_segments_with_source)
        
        # Keep only the most recent segments if history is too long
        if len(merged_data['segment_history']) > max_history:
            merged_data['segment_history'] = merged_data['segment_history'][-max_history:]
        
        # Update metadata
        merged_data['last_updated'] = datetime.now().isoformat()
        if 'update_count' not in merged_data:
            merged_data['update_count'] = 0
        merged_data['update_count'] += 1
        
        # Track source files
        if 'source_files' not in merged_data:
            merged_data['source_files'] = []
        
        # Add new source file if not already present
        new_filename = file_info['filename']
        if new_filename not in merged_data['source_files']:
            merged_data['source_files'].append(new_filename)
        
        # Update file count statistics
        merged_data['file_count'] = len(merged_data['source_files'])
        
        # Track learning history
        if 'learning_history' not in merged_data:
            merged_data['learning_history'] = []
        
        learning_entry = {
            'date': datetime.now().isoformat(),
            'source_file': new_filename,
            'new_segments': len(new_data['segments']),
            'new_speech_time': new_data['total_speech_time'],
            'total_samples_after': merged_data['total_samples'],
            'total_speech_time_after': merged_data['total_speech_time']
        }
        merged_data['learning_history'].append(learning_entry)
        
        # Keep only recent learning history (last 20 updates)
        if len(merged_data['learning_history']) > 20:
            merged_data['learning_history'] = merged_data['learning_history'][-20:]
        
        return merged_data
    
    def _calculate_embedding_change(self, old_embedding: np.ndarray, new_embedding: np.ndarray) -> float:
        """
        Calculate the magnitude of change between old and new embeddings.
        
        Args:
            old_embedding: Previous embedding
            new_embedding: Updated embedding
            
        Returns:
            Magnitude of change (cosine distance)
        """
        # Flatten embeddings if needed
        old_flat = old_embedding.flatten()
        new_flat = new_embedding.flatten()
        
        # Calculate cosine similarity
        cosine_sim = np.dot(old_flat, new_flat) / (np.linalg.norm(old_flat) * np.linalg.norm(new_flat))
        
        # Return cosine distance (1 - similarity)
        return 1.0 - cosine_sim
    
    def _load_speaker_data(self, speaker_file: Path) -> Dict:
        """Load speaker data from pickle file."""
        try:
            with open(speaker_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading speaker data from {speaker_file}: {e}")
            raise e
    
    def _save_speaker_data(self, speaker_name: str, speaker_data: Dict, speaker_folder: Path):
        """Save speaker data to their individual folder."""
        try:
            # Ensure the folder exists
            speaker_folder.mkdir(exist_ok=True)
            
            # Save pickle file
            speaker_file = speaker_folder / f"{speaker_name}.pkl"
            with open(speaker_file, 'wb') as f:
                pickle.dump(speaker_data, f)
            
            # Update the JSON metadata file
            self._update_speaker_metadata(speaker_name, speaker_data, speaker_folder)
            
            print(f"✅ Updated files in folder: {speaker_folder}")
            
        except Exception as e:
            print(f"Error saving speaker data for {speaker_name}: {e}")
            raise e
    
    def _update_speaker_metadata(self, speaker_name: str, speaker_data: Dict, speaker_folder: Path):
        """Update the human-readable JSON metadata file in the speaker's folder."""
        try:
            metadata_file = speaker_folder / f"{speaker_name}_info.json"
            metadata = {
                'speaker_name': speaker_name,
                'total_samples': speaker_data.get('total_samples', 0),
                'total_speech_time': speaker_data.get('total_speech_time', 0),
                'source_files': speaker_data.get('source_files', []),
                'file_count': speaker_data.get('file_count', 0),
                'created_date': speaker_data.get('created_date', 'Unknown'),
                'last_updated': speaker_data.get('last_updated', datetime.now().isoformat()),
                'update_count': speaker_data.get('update_count', 0),
                'embedding_shape': speaker_data['embedding'].shape,
                'original_id': speaker_data.get('original_id', 'Unknown'),
                'folder_path': str(speaker_folder.absolute()),
                'learning_summary': {
                    'total_updates': speaker_data.get('update_count', 0),
                    'recent_learning_entries': len(speaker_data.get('learning_history', [])),
                    'total_segments_in_history': len(speaker_data.get('segment_history', []))
                }
            }
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            print(f"Warning: Could not update metadata for {speaker_name}: {e}")
    
    def adaptive_learning_rate(self, existing_samples: int, new_samples: int, 
                             base_rate: float = 0.5) -> float:
        """
        Calculate adaptive learning rate based on sample counts.
        
        Args:
            existing_samples: Number of existing samples
            new_samples: Number of new samples
            base_rate: Base learning rate
            
        Returns:
            Adaptive learning rate
        """
        if existing_samples == 0:
            # If no existing samples, use higher rate
            return min(base_rate + 0.3, 0.9)
        
        # Calculate ratio of new samples to total
        total_samples = existing_samples + new_samples
        new_ratio = new_samples / total_samples
        
        # Adjust learning rate based on the ratio
        # More new samples relative to existing = higher learning rate
        adaptive_rate = base_rate + (new_ratio * 0.4)  # Max boost of 0.4
        
        # Consider the absolute number of existing samples
        # If we have very few existing samples, be more aggressive
        if existing_samples < 5:
            adaptive_rate += 0.2
        elif existing_samples < 10:
            adaptive_rate += 0.1
        
        # Cap the learning rate
        return min(adaptive_rate, 0.95)
    
    def get_speaker_stats(self, speaker_name: str) -> Optional[Dict]:
        """
        Get comprehensive statistics for a specific speaker from their folder.
        
        Args:
            speaker_name: Name of the speaker
            
        Returns:
            Speaker statistics or None if not found
        """
        speaker_folder = self.speakers_folder / speaker_name
        speaker_file = speaker_folder / f"{speaker_name}.pkl"
        
        if not speaker_file.exists():
            return None
        
        try:
            speaker_data = self._load_speaker_data(speaker_file)
            
            stats = {
                'basic_info': {
                    'speaker_name': speaker_name,
                    'speaker_folder': str(speaker_folder.absolute()),
                    'total_samples': speaker_data.get('total_samples', 0),
                    'total_speech_time': speaker_data.get('total_speech_time', 0),
                    'file_count': speaker_data.get('file_count', 0),
                    'embedding_shape': speaker_data['embedding'].shape
                },
                'dates': {
                    'created_date': speaker_data.get('created_date', 'Unknown'),
                    'last_updated': speaker_data.get('last_updated', 'Unknown'),
                    'update_count': speaker_data.get('update_count', 0)
                },
                'sources': {
                    'source_files': speaker_data.get('source_files', []),
                    'original_id': speaker_data.get('original_id', 'Unknown')
                },
                'learning_stats': {
                    'learning_history': speaker_data.get('learning_history', []),
                    'recent_updates': len(speaker_data.get('learning_history', [])),
                    'segment_history_count': len(speaker_data.get('segment_history', []))
                },
                'embedding_stats': {
                    'embedding_mean': float(np.mean(speaker_data['embedding'])),
                    'embedding_std': float(np.std(speaker_data['embedding'])),
                    'embedding_norm': float(np.linalg.norm(speaker_data['embedding']))
                },
                'folder_info': {
                    'folder_exists': speaker_folder.exists(),
                    'pickle_file_exists': speaker_file.exists(),
                    'json_file_exists': (speaker_folder / f"{speaker_name}_info.json").exists()
                }
            }
            
            return stats
            
        except Exception as e:
            print(f"Error getting stats for speaker {speaker_name}: {e}")
            return None
    
    def backup_speakers(self, backup_dir: str = "speakers_backup") -> str:
        """
        Create a backup of all speaker folders.
        
        Args:
            backup_dir: Directory to save backup
            
        Returns:
            Path to the backup folder
        """
        backup_path = Path(backup_dir)
        backup_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_folder = backup_path / f"backup_{timestamp}"
        backup_folder.mkdir(exist_ok=True)
        
        # Copy all speaker folders
        copied_folders = 0
        for speaker_folder in self.speakers_folder.iterdir():
            if speaker_folder.is_dir():
                try:
                    backup_speaker_folder = backup_folder / speaker_folder.name
                    shutil.copytree(speaker_folder, backup_speaker_folder)
                    copied_folders += 1
                except Exception as e:
                    print(f"Error backing up {speaker_folder}: {e}")
        
        # Create backup manifest
        manifest = {
            'backup_date': datetime.now().isoformat(),
            'source_folder': str(self.speakers_folder.absolute()),
            'folders_backed_up': copied_folders,
            'backup_folder': str(backup_folder.absolute()),
            'folder_structure': 'individual_folders'
        }
        
        manifest_file = backup_folder / "backup_manifest.json"
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"Backup created at: {backup_folder}")
        print(f"Backed up {copied_folders} speaker folders")
        return str(backup_folder)
    
    def export_learning_report(self, learning_results: Dict, output_file: str):
        """
        Export detailed learning results to JSON file.
        
        Args:
            learning_results: Results from learn_from_embeddings
            output_file: Path to output JSON file
        """
        # Create comprehensive report
        report = {
            'report_info': {
                'timestamp': datetime.now().isoformat(),
                'report_type': 'embedding_learning_report',
                'folder_structure': 'individual_folders'
            },
            'summary': {
                'total_speakers_processed': len(learning_results.get('learning_stats', {})),
                'speakers_updated': len(learning_results['updated_speakers']),
                'speakers_skipped': len(learning_results['skipped_speakers']),
                'success_rate': len(learning_results['updated_speakers']) / max(1, len(learning_results.get('learning_stats', {})))
            },
            'updated_speakers': learning_results['updated_speakers'],
            'skipped_speakers': learning_results['skipped_speakers'],
            'detailed_stats': learning_results['learning_stats']
        }
        
        # Add aggregate statistics
        if learning_results['learning_stats']:
            stats_values = learning_results['learning_stats'].values()
            
            report['aggregate_stats'] = {
                'average_learning_rate': np.mean([s['learning_rate_used'] for s in stats_values]),
                'average_new_samples': np.mean([s['new_samples'] for s in stats_values]),
                'average_embedding_change': np.mean([s['embedding_change'] for s in stats_values]),
                'total_new_samples': sum([s['new_samples'] for s in stats_values]),
                'average_similarity_score': np.mean([s['similarity_score'] for s in stats_values]),
                'speaker_folders_updated': [s['speaker_folder'] for s in stats_values]
            }
        
        try:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"Learning report exported to: {output_file}")
            
        except Exception as e:
            print(f"Error exporting learning report: {e}")
    
    def analyze_learning_trends(self, speaker_name: str) -> Optional[Dict]:
        """
        Analyze learning trends for a specific speaker.
        
        Args:
            speaker_name: Name of the speaker to analyze
            
        Returns:
            Analysis results or None if speaker not found
        """
        stats = self.get_speaker_stats(speaker_name)
        if not stats:
            return None
        
        learning_history = stats['learning_stats']['learning_history']
        if not learning_history:
            return {'message': 'No learning history available'}
        
        # Calculate trends
        dates = [entry['date'] for entry in learning_history]
        sample_counts = [entry['new_segments'] for entry in learning_history]
        speech_times = [entry['new_speech_time'] for entry in learning_history]
        
        analysis = {
            'speaker_name': speaker_name,
            'speaker_folder': stats['basic_info']['speaker_folder'],
            'total_updates': len(learning_history),
            'date_range': {
                'first_update': dates[0] if dates else None,
                'last_update': dates[-1] if dates else None
            },
            'sample_trends': {
                'average_samples_per_update': np.mean(sample_counts),
                'total_samples_learned': sum(sample_counts),
                'max_samples_in_update': max(sample_counts),
                'min_samples_in_update': min(sample_counts)
            },
            'speech_time_trends': {
                'average_speech_time_per_update': np.mean(speech_times),
                'total_speech_time_learned': sum(speech_times),
                'max_speech_time_in_update': max(speech_times),
                'min_speech_time_in_update': min(speech_times)
            },
            'recent_activity': learning_history[-5:] if len(learning_history) > 5 else learning_history
        }
        
        return analysis
    
    def migrate_old_structure(self, dry_run: bool = True) -> Dict:
        """
        Migrate speakers from old flat file structure to new folder structure.
        
        Args:
            dry_run: If True, only show what would be migrated without actually doing it
            
        Returns:
            Migration report
        """
        migration_report = {
            'dry_run': dry_run,
            'found_old_files': [],
            'migration_plan': [],
            'migrated_speakers': [],
            'failed_migrations': [],
            'skipped_speakers': []
        }
        
        # Look for old-style pickle files in the speakers folder root
        for pkl_file in self.speakers_folder.glob("*.pkl"):
            speaker_name = pkl_file.stem
            migration_report['found_old_files'].append(str(pkl_file))
            
            # Check if speaker folder already exists
            speaker_folder = self.speakers_folder / speaker_name
            json_file = self.speakers_folder / f"{speaker_name}_info.json"
            
            if speaker_folder.exists():
                migration_report['skipped_speakers'].append({
                    'speaker_name': speaker_name,
                    'reason': 'Folder already exists'
                })
                continue
            
            migration_plan_item = {
                'speaker_name': speaker_name,
                'old_pkl_file': str(pkl_file),
                'old_json_file': str(json_file) if json_file.exists() else None,
                'new_folder': str(speaker_folder),
                'new_pkl_file': str(speaker_folder / f"{speaker_name}.pkl"),
                'new_json_file': str(speaker_folder / f"{speaker_name}_info.json")
            }
            migration_report['migration_plan'].append(migration_plan_item)
            
            if not dry_run:
                try:
                    # Create speaker folder
                    speaker_folder.mkdir(exist_ok=True)
                    
                    # Move pickle file
                    new_pkl_file = speaker_folder / f"{speaker_name}.pkl"
                    shutil.move(str(pkl_file), str(new_pkl_file))
                    
                    # Move JSON file if it exists
                    if json_file.exists():
                        new_json_file = speaker_folder / f"{speaker_name}_info.json"
                        shutil.move(str(json_file), str(new_json_file))
                        
                        # Update JSON file with folder path
                        try:
                            with open(new_json_file, 'r') as f:
                                metadata = json.load(f)
                            metadata['folder_path'] = str(speaker_folder.absolute())
                            with open(new_json_file, 'w') as f:
                                json.dump(metadata, f, indent=2)
                        except Exception as e:
                            print(f"Warning: Could not update metadata for {speaker_name}: {e}")
                    
                    migration_report['migrated_speakers'].append(migration_plan_item)
                    print(f"✅ Migrated {speaker_name} to folder structure")
                    
                except Exception as e:
                    migration_report['failed_migrations'].append({
                        'speaker_name': speaker_name,
                        'error': str(e)
                    })
                    print(f"❌ Failed to migrate {speaker_name}: {e}")
        
        # Print summary
        if dry_run:
            print(f"\n🔍 MIGRATION DRY RUN REPORT")
            print(f"Found {len(migration_report['found_old_files'])} old-style files")
            print(f"Would migrate {len(migration_report['migration_plan'])} speakers")
            print(f"Would skip {len(migration_report['skipped_speakers'])} speakers")
            print("\nSet dry_run=False to perform actual migration")
        else:
            print(f"\n✅ MIGRATION COMPLETE")
            print(f"Successfully migrated: {len(migration_report['migrated_speakers'])}")
            print(f"Failed migrations: {len(migration_report['failed_migrations'])}")
            print(f"Skipped: {len(migration_report['skipped_speakers'])}")
        
        return migration_report