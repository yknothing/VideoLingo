import os
import glob
from core._1_ytdlp import find_video_files
from core.utils.config_utils import get_storage_paths
import shutil

def cleanup(history_dir="history"):
    # Get configured storage paths
    paths = get_storage_paths()
    
    # Get video file name
    video_file = find_video_files()
    video_name = os.path.basename(video_file)
    video_name = os.path.splitext(video_name)[0]
    video_name = sanitize_filename(video_name)
    
    # Create required folders
    os.makedirs(history_dir, exist_ok=True)
    video_history_dir = os.path.join(history_dir, video_name)
    log_dir = os.path.join(video_history_dir, "log")
    gpt_log_dir = os.path.join(video_history_dir, "gpt_log")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(gpt_log_dir, exist_ok=True)

    # Move files from temp directory (processing files)
    temp_dir = paths['temp']
    if os.path.exists(temp_dir):
        # Move non-log files from temp
        for file in glob.glob(f"{temp_dir}/*"):
            if not file.endswith(('log', 'gpt_log')):
                move_file(file, video_history_dir)

        # Move log files from temp
        temp_log_dir = os.path.join(temp_dir, "log")
        if os.path.exists(temp_log_dir):
            for file in glob.glob(f"{temp_log_dir}/*"):
                move_file(file, log_dir)

        # Move gpt_log files from temp
        temp_gpt_log_dir = os.path.join(temp_dir, "gpt_log")
        if os.path.exists(temp_gpt_log_dir):
            for file in glob.glob(f"{temp_gpt_log_dir}/*"):
                move_file(file, gpt_log_dir)

    # Move files from output directory (final results)
    output_dir = paths['output']
    if os.path.exists(output_dir):
        for file in glob.glob(f"{output_dir}/*"):
            if not file.endswith(('log', 'gpt_log')):
                move_file(file, video_history_dir)

    # SAFETY: Only clean up empty directories, never delete configured directories
    # Clean up empty subdirectories but preserve the main structure

def move_file(src, dst):
    try:
        # Get the source file name
        src_filename = os.path.basename(src)
        # Use os.path.join to ensure correct path and include file name
        dst = os.path.join(dst, sanitize_filename(src_filename))
        
        if os.path.exists(dst):
            if os.path.isdir(dst):
                # If destination is a folder, try to delete its contents
                shutil.rmtree(dst, ignore_errors=True)
            else:
                # If destination is a file, try to delete it
                os.remove(dst)
        
        shutil.move(src, dst, copy_function=shutil.copy2)
        print(f"✅ Moved: {src} -> {dst}")
    except PermissionError:
        print(f"⚠️ Permission error: Cannot delete {dst}, attempting to overwrite")
        try:
            shutil.copy2(src, dst)
            os.remove(src)
            print(f"✅ Copied and deleted source file: {src} -> {dst}")
        except Exception as e:
            print(f"❌ Move failed: {src} -> {dst}")
            print(f"Error message: {str(e)}")
    except Exception as e:
        print(f"❌ Move failed: {src} -> {dst}")
        print(f"Error message: {str(e)}")

def sanitize_filename(filename):
    # Remove or replace disallowed characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename

if __name__ == "__main__":
    cleanup()