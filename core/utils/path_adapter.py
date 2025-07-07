"""
VideoLingo 路径适配器
为现有模块提供新架构兼容的路径管理

这个适配器确保：
1. 现有代码可以无缝迁移到新架构
2. 基于视频ID的路径管理
3. 向后兼容旧代码
"""

import os
import functools
from typing import Optional, Dict, Any
from core.utils.video_manager import get_video_manager
from core.utils.config_utils import get_storage_paths

class PathAdapter:
    """路径适配器，为旧代码提供新架构的路径"""
    
    def __init__(self):
        self.video_mgr = get_video_manager()
        self.legacy_paths = get_storage_paths()
        self._current_video_id = None
    
    def get_current_video_id(self) -> Optional[str]:
        """获取当前处理的视频ID"""
        if self._current_video_id is None:
            self._current_video_id = self.video_mgr.get_current_video_id()
        return self._current_video_id
    
    def set_current_video_id(self, video_id: str):
        """设置当前处理的视频ID"""
        self._current_video_id = video_id
    
    def get_input_path(self) -> str:
        """获取当前视频的input路径"""
        video_id = self.get_current_video_id()
        if video_id:
            try:
                paths = self.video_mgr.get_video_paths(video_id)
                return paths['input']
            except Exception as e:
                print(f"Warning: Failed to get video paths for {video_id}: {e}")
        return self.legacy_paths['input']
    
    def get_temp_dir(self) -> str:
        """获取当前视频的temp目录"""
        video_id = self.get_current_video_id()
        if video_id:
            return self.video_mgr.get_temp_dir(video_id)
        return self.legacy_paths['temp']
    
    def get_temp_file(self, filename: str) -> str:
        """获取temp目录中的文件路径"""
        video_id = self.get_current_video_id()
        if video_id:
            return self.video_mgr.get_temp_file(video_id, filename)
        return os.path.join(self.legacy_paths['temp'], filename)
    
    def get_output_dir(self) -> str:
        """获取输出目录"""
        return self.legacy_paths['output']
    
    def get_output_file(self, suffix: str, extension: str = '.mp4') -> str:
        """获取输出文件路径"""
        video_id = self.get_current_video_id()
        if video_id:
            return self.video_mgr.get_output_file(video_id, suffix, extension)
        # 降级到旧格式
        filename = f"output_{suffix}{extension}"
        return os.path.join(self.legacy_paths['output'], filename)
    
    def get_log_dir(self) -> str:
        """获取日志目录"""
        video_id = self.get_current_video_id()
        if video_id:
            temp_dir = self.video_mgr.get_temp_dir(video_id)
            log_dir = os.path.join(temp_dir, 'log')
            os.makedirs(log_dir, exist_ok=True)
            return log_dir
        # 降级到旧格式
        log_dir = os.path.join(self.legacy_paths['temp'], 'log')
        os.makedirs(log_dir, exist_ok=True)
        return log_dir
    
    def get_gpt_log_dir(self) -> str:
        """获取GPT日志目录"""
        video_id = self.get_current_video_id()
        if video_id:
            temp_dir = self.video_mgr.get_temp_dir(video_id)
            gpt_log_dir = os.path.join(temp_dir, 'gpt_log')
            os.makedirs(gpt_log_dir, exist_ok=True)
            return gpt_log_dir
        # 降级到旧格式
        gpt_log_dir = os.path.join(self.legacy_paths['temp'], 'gpt_log')
        os.makedirs(gpt_log_dir, exist_ok=True)
        return gpt_log_dir
    
    def get_audio_dir(self) -> str:
        """获取音频处理目录"""
        video_id = self.get_current_video_id()
        if video_id:
            temp_dir = self.video_mgr.get_temp_dir(video_id)
            audio_dir = os.path.join(temp_dir, 'audio')
            os.makedirs(audio_dir, exist_ok=True)
            return audio_dir
        # 降级到旧格式
        audio_dir = os.path.join(self.legacy_paths['temp'], 'audio')
        os.makedirs(audio_dir, exist_ok=True)
        return audio_dir
    
    def get_audio_segs_dir(self) -> str:
        """获取音频片段目录"""
        audio_dir = self.get_audio_dir()
        segs_dir = os.path.join(audio_dir, 'segs')
        os.makedirs(segs_dir, exist_ok=True)
        return segs_dir
    
    def get_audio_refers_dir(self) -> str:
        """获取参考音频目录"""
        audio_dir = self.get_audio_dir()
        refers_dir = os.path.join(audio_dir, 'refers')
        os.makedirs(refers_dir, exist_ok=True)
        return refers_dir
    
    def reset_for_new_video(self, video_path: str = None) -> str:
        """为新视频重置适配器状态"""
        if video_path:
            # 注册新视频
            video_id = self.video_mgr.register_video(video_path)
            self.set_current_video_id(video_id)
            return video_id
        else:
            # 清除当前视频ID，让系统自动检测
            self._current_video_id = None
            return self.get_current_video_id()

# 全局路径适配器实例
_path_adapter = PathAdapter()

def get_path_adapter() -> PathAdapter:
    """获取全局路径适配器实例"""
    return _path_adapter

# 便捷函数，用于在现有代码中快速使用
def get_current_video_paths() -> Dict[str, str]:
    """获取当前视频的所有路径"""
    adapter = get_path_adapter()
    return {
        'input': adapter.get_input_path(),
        'temp_dir': adapter.get_temp_dir(),
        'output_dir': adapter.get_output_dir(),
        'log_dir': adapter.get_log_dir(),
        'gpt_log_dir': adapter.get_gpt_log_dir(),
        'audio_dir': adapter.get_audio_dir(),
        'audio_segs_dir': adapter.get_audio_segs_dir(),
        'audio_refers_dir': adapter.get_audio_refers_dir()
    }

def with_video_context(func):
    """
    装饰器：确保函数在正确的视频上下文中执行
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        adapter = get_path_adapter()
        
        # 尝试获取当前视频ID
        current_id = adapter.get_current_video_id()
        if not current_id:
            print("Warning: No current video ID found in context")
        
        return func(*args, **kwargs)
    return wrapper