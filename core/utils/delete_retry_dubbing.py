import os
import shutil
from core.utils.path_adapter import get_path_adapter
from core.utils.video_manager import get_video_manager

def reset_dubbing_for_current_video():
    """
    新架构：重置当前视频的配音相关文件（覆盖模式，不删除）
    只重置当前视频的临时文件，保留input和历史数据
    """
    adapter = get_path_adapter()
    video_mgr = get_video_manager()
    
    current_video_id = adapter.get_current_video_id()
    if not current_video_id:
        print("Warning: No current video found for dubbing reset")
        return
    
    print(f"🔄 Resetting dubbing files for video: {current_video_id}")
    
    # 重置音频相关的临时文件
    audio_segs_dir = adapter.get_audio_segs_dir()
    if os.path.exists(audio_segs_dir):
        try:
            # 清空音频片段目录内容
            for item in os.listdir(audio_segs_dir):
                item_path = os.path.join(audio_segs_dir, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
            print(f"✅ Reset audio segments directory: {audio_segs_dir}")
        except Exception as e:
            print(f"❌ Error resetting audio segments: {str(e)}")
    
    # 重置配音相关的输出文件（允许覆盖）
    try:
        video_mgr.safe_overwrite_output_files(current_video_id)
        print(f"✅ Reset output files for video: {current_video_id}")
    except Exception as e:
        print(f"❌ Error resetting output files: {str(e)}")
    
    print("🎯 Dubbing reset completed - ready for new dubbing process")

def delete_dubbing_files():
    """
    DEPRECATED: 旧的删除方法，已被新的重置方法取代
    保留用于向后兼容
    """
    print("⚠️ Warning: delete_dubbing_files is deprecated. Use reset_dubbing_for_current_video instead.")
    reset_dubbing_for_current_video()

if __name__ == "__main__":
    reset_dubbing_for_current_video()