# speaker_matcher.py
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import json

class SpeakerMatcher:
    """
    Matches speakers from new audio files with known speakers from individual folders
    in the speakers directory.
    """
    
    def __init__(self, speakers_folder: str = "speakers"):
        """
        Initialize the speaker matcher.
        
        Args:
            speakers_folder: Path to folder containing known speaker folders
        """
        self.speakers_folder = Path(speakers_folder)
        self.speakers_folder.mkdir(exist_ok=True)
        self.known_speakers = self._load_known_speakers()
        
    def _load_known_speakers(self) -> Dict:
        """Load all known speaker embeddings from individual speaker folders."""
        known_speakers = {}
        
        # Look for speaker folders
        for speaker_folder in self.speakers_folder.iterdir():
            if not speaker_folder.is_dir():
                continue
                
            speaker_name = speaker_folder.name
            speaker_file = speaker_folder / f"{speaker_name}.pkl"
            
            if not speaker_file.exists():
                print(f"Warning: No pickle file found for speaker {speaker_name} in {speaker_folder}")
                continue
                
            try:
                with open(speaker_file, 'rb') as f:
                    speaker_data = pickle.load(f)
                    
                    # Debug: Check the loaded embedding
                    embedding = speaker_data['embedding']
                    print(f"Loaded speaker {speaker_name}: shape={embedding.shape}, "
                          f"mean={np.mean(embedding):.6f}, std={np.std(embedding):.6f}")
                    
                    # Add folder path info to speaker data
                    speaker_data['folder_path'] = str(speaker_folder.absolute())
                    known_speakers[speaker_name] = speaker_data
                    
            except Exception as e:
                print(f"Error loading {speaker_file}: {e}")
        
        print(f"Loaded {len(known_speakers)} known speakers from individual folders")
        return known_speakers
    
    def match_speakers(self, new_embeddings: Dict, similarity_threshold: float = 0.75) -> Dict:
        """
        Match speakers in new embeddings with known speakers.
        """
        matches = {}
        unmatched_speakers = []
        similarity_scores = {}
        
        if not self.known_speakers:
            print("No known speakers found. All speakers will be marked as unmatched.")
            for speaker in new_embeddings['speakers']:
                unmatched_speakers.append(speaker)
            return {
                'matches': matches,
                'unmatched_speakers': unmatched_speakers,
                'similarity_scores': similarity_scores,
                'relabeled_speakers': {}
            }
        
        print(f"\nMatching {len(new_embeddings['speakers'])} new speakers with {len(self.known_speakers)} known speakers")
        
        # Compare each new speaker with all known speakers
        for new_speaker, new_data in new_embeddings['speakers'].items():
            new_embedding = new_data['embedding']
            
            # Debug new embedding
            print(f"\nNew speaker {new_speaker}: shape={new_embedding.shape}, "
                  f"mean={np.mean(new_embedding):.6f}, std={np.std(new_embedding):.6f}")
            
            # Ensure embedding is 2D for cosine_similarity
            if new_embedding.ndim == 1:
                new_embedding_2d = new_embedding.reshape(1, -1)
            else:
                new_embedding_2d = new_embedding
            
            best_match = None
            best_similarity = 0
            speaker_similarities = {}
            
            for known_speaker, known_data in self.known_speakers.items():
                known_embedding = known_data['embedding']
                
                # Ensure known embedding is 2D for cosine_similarity
                if known_embedding.ndim == 1:
                    known_embedding_2d = known_embedding.reshape(1, -1)
                else:
                    known_embedding_2d = known_embedding
                
                # Check if embeddings have the same dimension
                if new_embedding_2d.shape[1] != known_embedding_2d.shape[1]:
                    print(f"Dimension mismatch: {new_speaker}({new_embedding_2d.shape[1]}) vs {known_speaker}({known_embedding_2d.shape[1]})")
                    continue
                
                # Calculate cosine similarity
                try:
                    similarity_matrix = cosine_similarity(new_embedding_2d, known_embedding_2d)
                    similarity = similarity_matrix[0][0]
                    
                    # Debug similarity calculation
                    print(f"Similarity {new_speaker} vs {known_speaker}: {similarity:.6f}")
                    
                    # Additional verification: manual calculation
                    new_flat = new_embedding.flatten()
                    known_flat = known_embedding.flatten()
                    
                    # Check for zero vectors
                    new_norm = np.linalg.norm(new_flat)
                    known_norm = np.linalg.norm(known_flat)
                    
                    if new_norm == 0 or known_norm == 0:
                        print(f"WARNING: Zero vector detected! New norm: {new_norm}, Known norm: {known_norm}")
                        similarity = 0.0
                    else:
                        manual_similarity = np.dot(new_flat, known_flat) / (new_norm * known_norm)
                        
                        # Check if they match
                        if abs(similarity - manual_similarity) > 1e-6:
                            print(f"WARNING: Similarity calculation mismatch! sklearn: {similarity:.6f}, manual: {manual_similarity:.6f}")
                    
                    # Check for nearly identical embeddings
                    if np.allclose(new_flat, known_flat, atol=1e-8):
                        print(f"WARNING: Embeddings are nearly identical!")
                    
                    speaker_similarities[known_speaker] = similarity
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = known_speaker
                        
                except Exception as e:
                    print(f"Error calculating similarity between {new_speaker} and {known_speaker}: {e}")
                    speaker_similarities[known_speaker] = 0.0
            
            similarity_scores[new_speaker] = speaker_similarities
            
            # Check if best match meets threshold
            if best_match and best_similarity >= similarity_threshold:
                matches[new_speaker] = {
                    'matched_speaker': best_match,
                    'similarity': best_similarity,
                    'speaker_folder': self.known_speakers[best_match].get('folder_path', 'Unknown')
                }
                print(f"✅ Matched {new_speaker} -> {best_match} (similarity: {best_similarity:.6f})")
                print(f"   📁 Speaker folder: {matches[new_speaker]['speaker_folder']}")
            else:
                unmatched_speakers.append(new_speaker)
                print(f"❌ No match found for {new_speaker} (best similarity: {best_similarity:.6f})")
        
        # Create relabeled speakers dictionary
        relabeled_speakers = {}
        for new_speaker, speaker_data in new_embeddings['speakers'].items():
            if new_speaker in matches:
                matched_name = matches[new_speaker]['matched_speaker']
                relabeled_speakers[matched_name] = speaker_data
            else:
                relabeled_speakers[new_speaker] = speaker_data
        
        print(f"\nMatching Summary:")
        print(f"  Matches: {len(matches)}")
        print(f"  Unmatched: {len(unmatched_speakers)}")
        
        return {
            'matches': matches,
            'unmatched_speakers': unmatched_speakers,
            'similarity_scores': similarity_scores,
            'relabeled_speakers': relabeled_speakers
        }
    
    def relabel_diarization(self, diarization_segments: List[Dict], 
                          speaker_mapping: Dict) -> List[Dict]:
        """
        Relabel diarization segments based on speaker matching results.
        """
        relabeled_segments = []
        
        for segment in diarization_segments:
            original_speaker = segment['speaker']
            new_segment = segment.copy()
            
            # Find the new label for this speaker
            if original_speaker in speaker_mapping:
                new_segment['speaker'] = speaker_mapping[original_speaker]
                new_segment['original_speaker'] = original_speaker
            
            relabeled_segments.append(new_segment)
        
        return relabeled_segments
    
    def get_speaker_mapping_from_matches(self, match_results: Dict) -> Dict:
        """
        Create a speaker mapping dictionary from match results.
        """
        speaker_mapping = {}
        
        # Add matched speakers
        for original_speaker, match_info in match_results['matches'].items():
            speaker_mapping[original_speaker] = match_info['matched_speaker']
        
        # Keep unmatched speakers with their original labels
        for unmatched_speaker in match_results['unmatched_speakers']:
            speaker_mapping[unmatched_speaker] = unmatched_speaker
        
        return speaker_mapping
    
    def reload_known_speakers(self):
        """Reload known speakers from the speaker folders."""
        print("🔄 Reloading known speakers from folders...")
        self.known_speakers = self._load_known_speakers()
    
    def get_known_speaker_list(self) -> List[str]:
        """Get list of known speaker names."""
        return list(self.known_speakers.keys())
    
    def get_speaker_info(self, speaker_name: str) -> Optional[Dict]:
        """
        Get detailed information about a specific known speaker.
        
        Args:
            speaker_name: Name of the speaker
            
        Returns:
            Speaker information or None if not found
        """
        if speaker_name not in self.known_speakers:
            return None
            
        speaker_data = self.known_speakers[speaker_name]
        folder_path = Path(speaker_data.get('folder_path', ''))
        
        # Try to load metadata file for additional info
        metadata_file = folder_path / f"{speaker_name}_info.json"
        metadata = {}
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load metadata for {speaker_name}: {e}")
        
        return {
            'speaker_name': speaker_name,
            'folder_path': str(folder_path),
            'embedding_shape': speaker_data['embedding'].shape,
            'total_samples': speaker_data.get('total_samples', 0),
            'total_speech_time': speaker_data.get('total_speech_time', 0),
            'source_files': speaker_data.get('source_files', []),
            'created_date': speaker_data.get('created_date', 'Unknown'),
            'last_updated': speaker_data.get('last_updated', 'Unknown'),
            'metadata': metadata
        }
    
    def save_matching_report(self, match_results: Dict, output_file: str):
        """
        Save a detailed matching report to JSON file.
        """
        # Enhance report with folder information
        enhanced_matches = {}
        for speaker, match_info in match_results['matches'].items():
            enhanced_matches[speaker] = match_info.copy()
            matched_speaker = match_info['matched_speaker']
            speaker_info = self.get_speaker_info(matched_speaker)
            if speaker_info:
                enhanced_matches[speaker]['matched_speaker_info'] = {
                    'folder_path': speaker_info['folder_path'],
                    'total_samples': speaker_info['total_samples'],
                    'total_speech_time': speaker_info['total_speech_time'],
                    'source_files': speaker_info['source_files'],
                    'created_date': speaker_info['created_date']
                }
        
        report = {
            'matches': enhanced_matches,
            'unmatched_speakers': match_results['unmatched_speakers'],
            'similarity_scores': match_results['similarity_scores'],
            'known_speakers': list(self.known_speakers.keys()),
            'known_speaker_folders': [self.get_speaker_info(name)['folder_path'] 
                                    for name in self.known_speakers.keys()],
            'total_new_speakers': len(match_results['similarity_scores']),
            'total_matches': len(match_results['matches']),
            'total_unmatched': len(match_results['unmatched_speakers']),
            'speakers_folder_structure': 'individual_folders',
            'speakers_base_folder': str(self.speakers_folder.absolute())
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Matching report saved to: {output_file}")
    
    def debug_embeddings(self, new_embeddings: Dict):
        """
        Debug function to analyze embeddings in detail.
        """
        print("\n🔍 EMBEDDING DEBUG ANALYSIS")
        print("=" * 50)
        
        # Analyze new embeddings
        print("New Embeddings:")
        for speaker, data in new_embeddings['speakers'].items():
            embedding = data['embedding']
            print(f"  {speaker}:")
            print(f"    Shape: {embedding.shape}")
            print(f"    Mean: {np.mean(embedding):.6f}")
            print(f"    Std: {np.std(embedding):.6f}")
            print(f"    Min: {np.min(embedding):.6f}")
            print(f"    Max: {np.max(embedding):.6f}")
            print(f"    Norm: {np.linalg.norm(embedding):.6f}")
            print(f"    First 5 values: {embedding.flatten()[:5]}")
        
        # Analyze known embeddings
        print("\nKnown Embeddings (from folders):")
        for speaker, data in self.known_speakers.items():
            embedding = data['embedding']
            folder_path = data.get('folder_path', 'Unknown')
            print(f"  {speaker}:")
            print(f"    Folder: {folder_path}")
            print(f"    Shape: {embedding.shape}")
            print(f"    Mean: {np.mean(embedding):.6f}")
            print(f"    Std: {np.std(embedding):.6f}")
            print(f"    Min: {np.min(embedding):.6f}")
            print(f"    Max: {np.max(embedding):.6f}")
            print(f"    Norm: {np.linalg.norm(embedding):.6f}")
            print(f"    First 5 values: {embedding.flatten()[:5]}")
        
        # Cross-compare all embeddings
        print("\nCross-comparison Matrix:")
        all_speakers = {}
        all_speakers.update({f"NEW_{k}": v['embedding'] for k, v in new_embeddings['speakers'].items()})
        all_speakers.update({f"KNOWN_{k}": v['embedding'] for k, v in self.known_speakers.items()})
        
        speaker_names = list(all_speakers.keys())
        for i, speaker1 in enumerate(speaker_names):
            for j, speaker2 in enumerate(speaker_names):
                if i < j:  # Only upper triangle
                    emb1 = all_speakers[speaker1].flatten()
                    emb2 = all_speakers[speaker2].flatten()
                    
                    if len(emb1) == len(emb2):
                        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                        print(f"  {speaker1} vs {speaker2}: {similarity:.6f}")
                    else:
                        print(f"  {speaker1} vs {speaker2}: DIMENSION MISMATCH")
    
    def list_speaker_folders(self) -> List[Dict]:
        """
        List all speaker folders and their contents.
        
        Returns:
            List of folder information
        """
        folders_info = []
        
        for speaker_folder in self.speakers_folder.iterdir():
            if not speaker_folder.is_dir():
                continue
                
            speaker_name = speaker_folder.name
            folder_info = {
                'speaker_name': speaker_name,
                'folder_path': str(speaker_folder.absolute()),
                'files': []
            }
            
            # List files in the folder
            for file_path in speaker_folder.iterdir():
                if file_path.is_file():
                    folder_info['files'].append({
                        'filename': file_path.name,
                        'size_bytes': file_path.stat().st_size,
                        'modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                    })
            
            # Check if this speaker is loaded
            folder_info['is_loaded'] = speaker_name in self.known_speakers
            
            folders_info.append(folder_info)
        
        return sorted(folders_info, key=lambda x: x['speaker_name'])
    
    def verify_folder_structure(self) -> Dict:
        """
        Verify the integrity of the speaker folder structure.
        
        Returns:
            Verification report
        """
        report = {
            'total_folders': 0,
            'valid_folders': 0,
            'invalid_folders': [],
            'missing_files': [],
            'loaded_speakers': len(self.known_speakers),
            'folder_details': []
        }
        
        for speaker_folder in self.speakers_folder.iterdir():
            if not speaker_folder.is_dir():
                continue
                
            report['total_folders'] += 1
            speaker_name = speaker_folder.name
            
            folder_detail = {
                'speaker_name': speaker_name,
                'folder_path': str(speaker_folder),
                'has_pickle': False,
                'has_json': False,
                'is_valid': False,
                'issues': []
            }
            
            # Check for required files
            pickle_file = speaker_folder / f"{speaker_name}.pkl"
            json_file = speaker_folder / f"{speaker_name}_info.json"
            
            if pickle_file.exists():
                folder_detail['has_pickle'] = True
            else:
                folder_detail['issues'].append(f"Missing pickle file: {pickle_file.name}")
                report['missing_files'].append(str(pickle_file))
            
            if json_file.exists():
                folder_detail['has_json'] = True
            else:
                folder_detail['issues'].append(f"Missing JSON file: {json_file.name}")
                report['missing_files'].append(str(json_file))
            
            # Check if folder is valid (has at least pickle file)
            if folder_detail['has_pickle']:
                folder_detail['is_valid'] = True
                report['valid_folders'] += 1
            else:
                report['invalid_folders'].append(speaker_name)
            
            report['folder_details'].append(folder_detail)
        
        return report