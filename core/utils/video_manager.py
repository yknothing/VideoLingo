"""
VideoLingo 视频文件管理系统 - 架构师重新设计
基于视频ID的1:1:n文件映射关系

架构原则:
1. 每个视频有唯一ID (基于内容hash + 时间戳)
2. 文件组织: input/video_id.ext -> temp/video_id/ -> output/video_id_*.ext
3. 只允许覆盖，禁止删除
4. 完全隔离不同视频的处理过程
"""

import os
import hashlib
import time
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from core.utils.config_utils import get_storage_paths

class VideoFileManager:
    """视频文件管理器 - 基于ID的文件组织系统"""
    
    def __init__(self):
        self.paths = get_storage_paths()
        self.ensure_directory_structure()
        
    def ensure_directory_structure(self):
        """确保目录结构存在"""
        for path in self.paths.values():
            os.makedirs(path, exist_ok=True)
            
    def generate_video_id(self, video_path: str) -> str:
        """
        为视频生成唯一ID
        基于文件内容hash + 文件名 + 时间戳
        """
        try:
            # 读取文件前1MB计算hash (避免大文件性能问题)
            with open(video_path, 'rb') as f:
                first_chunk = f.read(1024 * 1024)  # 1MB
                file_hash = hashlib.md5(first_chunk).hexdigest()[:8]
            
            # 文件名hash
            filename = os.path.basename(video_path)
            name_hash = hashlib.md5(filename.encode()).hexdigest()[:4]
            
            # 时间戳
            timestamp = str(int(time.time()))[-6:]  # 后6位时间戳
            
            # 组合ID: hash_name_timestamp
            video_id = f"{file_hash}_{name_hash}_{timestamp}"
            return video_id
            
        except Exception as e:
            # 降级方案：仅基于文件名和时间
            filename = os.path.basename(video_path)
            name_hash = hashlib.md5(filename.encode()).hexdigest()[:8]
            timestamp = str(int(time.time()))[-6:]
            return f"fallback_{name_hash}_{timestamp}"
    
    def register_video(self, video_path: str, video_id: Optional[str] = None) -> str:
        """
        注册新视频到系统中
        返回视频ID
        """
        if video_id is None:
            video_id = self.generate_video_id(video_path)
        
        # 确保input目录中有正确的文件
        input_dir = self.paths['input']
        ext = os.path.splitext(video_path)[1]
        target_input_path = os.path.join(input_dir, f"{video_id}{ext}")
        
        # 智能文件管理：避免不必要的复制
        if os.path.abspath(video_path) != os.path.abspath(target_input_path):
            # 检查是否已经在input目录中
            if os.path.dirname(os.path.abspath(video_path)) == os.path.abspath(input_dir):
                # 文件已在input目录，直接重命名
                os.rename(video_path, target_input_path)
            else:
                # 文件在其他目录，移动到input目录（避免复制）
                shutil.move(video_path, target_input_path)
        
        # 创建对应的temp目录
        temp_dir = self.get_temp_dir(video_id)
        os.makedirs(temp_dir, exist_ok=True)
        
        # 创建视频元数据
        self._save_video_metadata(video_id, {
            'original_path': video_path,
            'original_filename': os.path.basename(video_path),
            'registered_time': time.time(),
            'input_path': target_input_path,
            'temp_dir': temp_dir,
            'file_extension': ext,
            'moved_from': video_path if video_path != target_input_path else None
        })
        
        return video_id
    
    def get_video_paths(self, video_id: str) -> Dict[str, str]:
        """
        获取视频ID对应的所有路径
        """
        metadata = self._load_video_metadata(video_id)
        if not metadata:
            raise ValueError(f"Video ID {video_id} not found")
        
        return {
            'input': metadata['input_path'],
            'temp_dir': self.get_temp_dir(video_id),
            'output_dir': self.paths['output'],
            'metadata_file': self._get_metadata_path(video_id)
        }
    
    def get_temp_dir(self, video_id: str) -> str:
        """获取视频的临时文件目录"""
        return os.path.join(self.paths['temp'], video_id)
    
    def get_temp_file(self, video_id: str, filename: str) -> str:
        """获取temp目录中的特定文件路径"""
        return os.path.join(self.get_temp_dir(video_id), filename)
    
    def get_output_file(self, video_id: str, suffix: str, extension: str = '.mp4') -> str:
        """获取输出文件路径"""
        filename = f"{video_id}_{suffix}{extension}"
        return os.path.join(self.paths['output'], filename)
    
    def list_temp_files(self, video_id: str) -> List[str]:
        """列出视频的所有临时文件"""
        temp_dir = self.get_temp_dir(video_id)
        if not os.path.exists(temp_dir):
            return []
        
        files = []
        for root, dirs, filenames in os.walk(temp_dir):
            for filename in filenames:
                files.append(os.path.join(root, filename))
        return files
    
    def list_output_files(self, video_id: str) -> List[str]:
        """列出视频的所有输出文件"""
        output_dir = self.paths['output']
        if not os.path.exists(output_dir):
            return []
        
        files = []
        for filename in os.listdir(output_dir):
            if filename.startswith(f"{video_id}_"):
                files.append(os.path.join(output_dir, filename))
        return files
    
    def safe_overwrite_temp_files(self, video_id: str):
        """
        安全地覆盖临时文件（不删除，只清空内容用于新处理）
        """
        temp_dir = self.get_temp_dir(video_id)
        if os.path.exists(temp_dir):
            # 创建覆盖日志
            self._log_overwrite_operation(video_id, 'temp', self.list_temp_files(video_id))
            
            # 清空temp目录内容，但保留目录结构
            for item in os.listdir(temp_dir):
                item_path = os.path.join(temp_dir, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
    
    def safe_overwrite_output_files(self, video_id: str):
        """
        安全地覆盖输出文件（用于重新生成输出）
        """
        output_files = self.list_output_files(video_id)
        if output_files:
            # 创建覆盖日志
            self._log_overwrite_operation(video_id, 'output', output_files)
            
            # 删除旧的输出文件
            for file_path in output_files:
                if os.path.exists(file_path):
                    os.remove(file_path)
    
    def get_current_video_id(self) -> Optional[str]:
        """
        获取当前正在处理的视频ID（从input目录推断）
        """
        input_dir = self.paths['input']
        if not os.path.exists(input_dir):
            return None
        
        # 查找input目录中的视频文件
        video_files = []
        for filename in os.listdir(input_dir):
            if any(filename.lower().endswith(ext) for ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']):
                video_files.append(filename)
        
        if not video_files:
            return None
        
        # 如果只有一个视频文件，提取其ID
        if len(video_files) == 1:
            filename = video_files[0]
            # 假设文件名格式为 video_id.ext
            video_id = os.path.splitext(filename)[0]
            return video_id
        
        # 多个文件时返回最新的
        newest_file = max(video_files, key=lambda f: os.path.getmtime(os.path.join(input_dir, f)))
        return os.path.splitext(newest_file)[0]
    
    def _save_video_metadata(self, video_id: str, metadata: Dict):
        """保存视频元数据"""
        metadata_path = self._get_metadata_path(video_id)
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    def _load_video_metadata(self, video_id: str) -> Optional[Dict]:
        """加载视频元数据"""
        metadata_path = self._get_metadata_path(video_id)
        if not os.path.exists(metadata_path):
            return None
        
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None
    
    def _get_metadata_path(self, video_id: str) -> str:
        """获取元数据文件路径"""
        temp_dir = self.get_temp_dir(video_id)
        return os.path.join(temp_dir, '.metadata.json')
    
    def _log_overwrite_operation(self, video_id: str, operation_type: str, affected_files: List[str]):
        """记录覆盖操作日志"""
        log_dir = os.path.join(self.get_temp_dir(video_id), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, 'overwrite_operations.log')
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        
        log_entry = {
            'timestamp': timestamp,
            'video_id': video_id,
            'operation_type': operation_type,
            'affected_files': affected_files
        }
        
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')

# 全局视频管理器实例
video_manager = VideoFileManager()

def get_video_manager() -> VideoFileManager:
    """获取全局视频管理器实例"""
    return video_manager