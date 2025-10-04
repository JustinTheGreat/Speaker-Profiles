# simple_auto_tagging_example.py
"""
Simple examples of using the automatic speaker tagging system.
"""

from auto_speaker_tagging_system import AutoSpeakerTaggingSystem
from pathlib import Path

def simple_auto_tag_example(audio_file: str):
    """
    Simplest example: Auto-tag speakers with default settings.
    
    This will:
    1. Extract speakers from the audio file
    2. Match them against existing speaker profiles
    3. Update existing profiles if matches found
    4. Create new profiles for unknown speakers
    """
    print("🎙️ SIMPLE AUTO-TAGGING EXAMPLE")
    print("="*50)
    
    # Initialize the system with default settings
    system = AutoSpeakerTaggingSystem()
    
    # Process the audio file
    results = system.process_audio_file(audio_file)
    
    # Show the speaker assignments
    print("\n🏷️  FINAL SPEAKER ASSIGNMENTS:")
    for segment in results['labeled_timeline']:
        assigned_speaker = segment['assigned_speaker']
        assignment_type = segment['assignment_type']
        start_time = segment['start_time']
        end_time = segment['end_time']
        
        if assignment_type == 'matched':
            icon = "🎯"
        elif assignment_type == 'new_profile':
            icon = "🆕"
        else:
            icon = "🚫"
        
        print(f"{icon} {start_time}-{end_time}: {assigned_speaker}")
    
    return results

def custom_settings_example(audio_file: str):
    """
    Example with custom settings for stricter matching.
    """
    print("🎙️ CUSTOM SETTINGS EXAMPLE")
    print("="*50)
    
    # Initialize with custom settings
    system = AutoSpeakerTaggingSystem(
        similarity_threshold=0.8,  # Higher threshold for stricter matching
        min_quality=0.5,          # Higher quality requirement
        min_speech_time=3.0,      # Longer minimum speech time
        auto_save_unknown=True    # Still save unknown speakers
    )
    
    # Process the audio file
    results = system.process_audio_file(
        audio_file,
        custom_unknown_prefix="Unknown_Person",  # Custom prefix for new speakers
        update_existing=True                      # Update existing profiles
    )
    
    # Save results to files
    system.save_results(results)
    
    return results

def batch_processing_example(audio_files: list):
    """
    Example of processing multiple audio files in batch.
    """
    print("🎙️ BATCH PROCESSING EXAMPLE")
    print("="*50)
    
    # Initialize system
    system = AutoSpeakerTaggingSystem(
        similarity_threshold=0.75,
        min_quality=0.4,
        auto_save_unknown=True
    )
    
    # Process all files
    all_results = system.process_multiple_files(audio_files)
    
    # Show summary for each file
    print("\n📊 BATCH SUMMARY:")
    for audio_file, results in all_results.items():
        if 'error' in results:
            print(f"❌ {Path(audio_file).name}: ERROR - {results['error']}")
        else:
            summary = results['summary']
            print(f"✅ {Path(audio_file).name}: "
                  f"{summary['matched_existing']} matched, "
                  f"{summary['new_profiles_created']} new")
    
    return all_results

def monitor_speaker_database():
    """
    Monitor the speaker database and show current status.
    """
    print("📂 SPEAKER DATABASE STATUS")
    print("="*50)
    
    system = AutoSpeakerTaggingSystem()
    db_info = system.get_speaker_database_info()
    
    if db_info['total_speakers'] == 0:
        print("📭 No speakers in database yet")
        print("   Add some audio files to start building speaker profiles!")
    else:
        print(f"👥 Total speakers: {db_info['total_speakers']}")
        print(f"⏱️  Total speech time: {db_info['total_speech_time']:.1f} seconds")
        print(f"📊 Average per speaker: {db_info['average_speech_time']:.1f} seconds")
        
        print(f"\n🎤 Known speakers:")
        for speaker in db_info['speakers']:
            print(f"   - {speaker}")
    
    return db_info

def interactive_example():
    """
    Interactive example that asks user for input.
    """
    print("🎙️ INTERACTIVE AUTO-TAGGING")
    print("="*50)
    
    # Get audio file from user
    audio_file = input("Enter path to audio file: ").strip()
    
    if not Path(audio_file).exists():
        print(f"❌ File not found: {audio_file}")
        return
    
    # Ask for settings
    print("\nChoose settings:")
    print("1. Default settings (recommended)")
    print("2. Strict matching (higher similarity threshold)")
    print("3. Lenient matching (lower similarity threshold)")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "2":
        # Strict settings
        system = AutoSpeakerTaggingSystem(
            similarity_threshold=0.85,
            min_quality=0.6,
            min_speech_time=3.0
        )
        print("🔒 Using strict matching settings")
    elif choice == "3":
        # Lenient settings
        system = AutoSpeakerTaggingSystem(
            similarity_threshold=0.65,
            min_quality=0.2,
            min_speech_time=1.0
        )
        print("🔓 Using lenient matching settings")
    else:
        # Default settings
        system = AutoSpeakerTaggingSystem()
        print("⚙️ Using default settings")
    
    # Show current database
    monitor_speaker_database()
    
    # Process the file
    print(f"\nProcessing: {Path(audio_file).name}")
    results = system.process_audio_file(audio_file)
    
    # Ask if user wants to save results
    save_choice = input("\nSave results to files? (y/n): ").strip().lower()
    if save_choice == 'y':
        system.save_results(results)
    
    return results

def main():
    """
    Main function with examples.
    """
    import sys
    
    print("🎙️ AUTOMATIC SPEAKER TAGGING SYSTEM")
    print("="*60)
    print("This system automatically identifies speakers and manages profiles")
    print()
    
    if len(sys.argv) > 1:
        # Command line usage
        audio_file = sys.argv[1]
        
        if not Path(audio_file).exists():
            print(f"❌ File not found: {audio_file}")
            return 1
        
        print(f"Processing: {audio_file}")
        
        # Check if this is batch mode
        if len(sys.argv) > 2 and sys.argv[2] == "--batch":
            # Treat as first of multiple files
            audio_files = sys.argv[1:]
            audio_files.remove("--batch")
            batch_processing_example(audio_files)
        else:
            # Single file processing
            simple_auto_tag_example(audio_file)
    
    else:
        # Interactive mode
        print("Choose an example:")
        print("1. 📁 Show speaker database status")
        print("2. 🎙️  Process single audio file (interactive)")
        print("3. 🎯 Simple auto-tagging example")
        print("4. ⚙️  Custom settings example")
        print("5. 📋 Batch processing example")
        print("0. 🚪 Exit")
        
        while True:
            choice = input("\nEnter choice (0-5): ").strip()
            
            if choice == "0":
                print("👋 Goodbye!")
                break
            
            elif choice == "1":
                monitor_speaker_database()
            
            elif choice == "2":
                interactive_example()
            
            elif choice == "3":
                audio_file = input("Enter audio file path: ").strip()
                if Path(audio_file).exists():
                    simple_auto_tag_example(audio_file)
                else:
                    print(f"❌ File not found: {audio_file}")
            
            elif choice == "4":
                audio_file = input("Enter audio file path: ").strip()
                if Path(audio_file).exists():
                    custom_settings_example(audio_file)
                else:
                    print(f"❌ File not found: {audio_file}")
            
            elif choice == "5":
                print("Enter audio file paths (one per line, empty line to finish):")
                audio_files = []
                while True:
                    file_path = input().strip()
                    if not file_path:
                        break
                    if Path(file_path).exists():
                        audio_files.append(file_path)
                    else:
                        print(f"⚠️  File not found: {file_path}")
                
                if audio_files:
                    batch_processing_example(audio_files)
                else:
                    print("❌ No valid files provided")
            
            else:
                print("❌ Invalid choice")
            
            input("\nPress Enter to continue...")
    
    return 0


if __name__ == "__main__":
    exit(main())