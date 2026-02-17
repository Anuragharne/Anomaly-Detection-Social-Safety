import os

# CONFIGURATION
DATA_DIR = r"01_Data"  # Path to your data folder

def sanitize_filenames(root_dir):
    print(f"🧹 Starting cleanup in: {root_dir}...")
    
    # Walk through all folders (train, val) and subfolders (Fight, Safe)
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # We only care about the bottom-level folders where videos are
        if not filenames:
            continue
            
        # Get the class name (e.g., "Fight" or "Safe") from the folder name
        class_name = os.path.basename(dirpath)
        print(f"\n📂 Processing: {dirpath} ({len(filenames)} files)")
        
        count = 1
        for filename in filenames:
            # Get full old path
            old_path = os.path.join(dirpath, filename)
            
            # Get extension (e.g., .avi, .mp4)
            _, extension = os.path.splitext(filename)
            
            # Create NEW clean name: ClassName_0001.avi
            new_filename = f"{class_name}_{count:05d}{extension}"
            new_path = os.path.join(dirpath, new_filename)
            
            # Rename
            try:
                os.rename(old_path, new_path)
                # Print only every 100th file to keep console clean
                if count % 100 == 0:
                    print(f"   ✅ Renamed {count}: {filename[:20]}... -> {new_filename}")
                count += 1
            except OSError as e:
                print(f"   ❌ Error renaming {filename}: {e}")

    print("\n✨ ALL DONE! Your dataset is now clean and ready for upload.")

if __name__ == "__main__":
    # Double check path exists before running
    if os.path.exists(DATA_DIR):
        confirm = input(f"⚠️ This will rename ALL files in '{DATA_DIR}'. Type 'yes' to proceed: ")
        if confirm.lower() == "yes":
            sanitize_filenames(DATA_DIR)
        else:
            print("❌ Operation cancelled.")
    else:
        print(f"❌ Error: Folder '{DATA_DIR}' not found.")