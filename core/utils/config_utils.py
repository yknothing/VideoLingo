from ruamel.yaml import YAML
import threading
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

CONFIG_PATH = 'config.yaml'
lock = threading.Lock()

yaml = YAML()
yaml.preserve_quotes = True

# ------------
# Environment variable mapping for sensitive data
# ------------
ENV_MAPPINGS = {
    'api.key': 'API_KEY',
    'api.base_url': 'API_BASE_URL',
    'video_storage.base_path': 'VIDEO_STORAGE_BASE_PATH',
    'youtube.cookies_path': 'YOUTUBE_COOKIES_PATH',
    'youtube.proxy': 'YOUTUBE_PROXY',
    'model_dir': 'WHISPER_MODEL_DIR'
}

# -----------------------
# load & update config
# -----------------------

def load_key(key):
    # First check if this key should be loaded from environment
    if key in ENV_MAPPINGS:
        env_value = os.getenv(ENV_MAPPINGS[key])
        if env_value:
            # Special handling for empty string in config (means read from env)
            return env_value
    
    with lock:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as file:
            data = yaml.load(file)

    keys = key.split('.')
    value = data
    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            raise KeyError(f"Key '{k}' not found in configuration")
    
    # If config value is empty string and env mapping exists, try environment
    if value == '' and key in ENV_MAPPINGS:
        env_value = os.getenv(ENV_MAPPINGS[key])
        if env_value:
            return env_value
    
    return value

def update_key(key, new_value):
    with lock:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as file:
            data = yaml.load(file)

        keys = key.split('.')
        current = data
        for k in keys[:-1]:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return False

        if isinstance(current, dict) and keys[-1] in current:
            current[keys[-1]] = new_value
            with open(CONFIG_PATH, 'w', encoding='utf-8') as file:
                yaml.dump(data, file)
            return True
        else:
            raise KeyError(f"Key '{keys[-1]}' not found in configuration")
        
# basic utils
def get_joiner(language):
    if language in load_key('language_split_with_space'):
        return " "
    elif language in load_key('language_split_without_space'):
        return ""
    else:
        raise ValueError(f"Unsupported language code: {language}")

# directory management utils
def get_storage_paths():
    """Get the configured storage paths for video processing"""
    import os
    try:
        base_path = load_key('video_storage.base_path')
        input_dir = load_key('video_storage.input_dir')
        temp_dir = load_key('video_storage.temp_dir')
        output_dir = load_key('video_storage.output_dir')
        
        return {
            'base': base_path,
            'input': os.path.join(base_path, input_dir),
            'temp': os.path.join(base_path, temp_dir),
            'output': os.path.join(base_path, output_dir)
        }
    except KeyError:
        # Fallback with separate directories for safety - never mix input with processing directories
        return {
            'base': 'output',
            'input': 'input',      # Use separate input directory to prevent accidental deletion
            'temp': 'temp',        # Use separate temp directory 
            'output': 'output'     # Use separate output directory
        }

def ensure_storage_dirs():
    """Create storage directories and required subdirectories if they don't exist"""
    import os
    paths = get_storage_paths()
    
    # Create main directories
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    
    # Create required subdirectories
    # Temp directory subdirectories
    temp_subdirs = ['log', 'gpt_log', 'audio', 'audio/refers', 'audio/segs', 'audio/tmp']
    for subdir in temp_subdirs:
        os.makedirs(os.path.join(paths['temp'], subdir), exist_ok=True)
    
    # Output directory (keep simple, just the main directory)

def is_protected_directory(path):
    """
    Check if a directory is part of the protected storage structure
    Enhanced security: prevents deletion of system directories and parent paths
    """
    import os
    
    # Get configured paths
    paths = get_storage_paths()
    protected_paths = list(paths.values())
    
    # Add system-level protection
    system_protected = ['/', '/home', '/Users', '/System', '/usr', '/var', '/etc']
    protected_paths.extend(system_protected)
    
    abs_path = os.path.abspath(path)
    
    # Check if path is exactly a protected path
    for protected in protected_paths:
        protected_abs = os.path.abspath(protected)
        if abs_path == protected_abs:
            return True
    
    # Check if path is a parent of any protected path (critical safety check)
    for protected in protected_paths:
        protected_abs = os.path.abspath(protected)
        if protected_abs.startswith(abs_path + os.sep) or protected_abs == abs_path:
            return True
    
    # Check directory depth - never allow deletion of directories with depth < 2
    path_parts = abs_path.split(os.sep)
    if len(path_parts) < 3:  # ['', 'Users', 'username'] = depth 3
        return True
    
    return False

def clean_directory_contents(directory_path, preserve_structure=True):
    """
    Clean directory contents while preserving the directory structure if specified
    Enhanced security: multiple safety checks to prevent accidental deletion
    """
    import os
    import shutil
    
    # Critical safety checks
    if not directory_path or directory_path.strip() == '':
        raise ValueError("Directory path cannot be empty")
    
    abs_dir_path = os.path.abspath(directory_path)
    
    # Never allow cleaning of protected directories
    if is_protected_directory(abs_dir_path):
        raise ValueError(f"Cannot clean protected directory: {abs_dir_path}")
    
    # Additional safety: ensure we're only cleaning within project boundaries
    project_root = os.path.abspath('.')
    if not abs_dir_path.startswith(project_root):
        raise ValueError(f"Cannot clean directory outside project root: {abs_dir_path}")
    
    if not os.path.exists(directory_path):
        return
        
    try:
        for item in os.listdir(directory_path):
            item_path = os.path.join(directory_path, item)
            
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                # Only remove subdirectories if not preserving structure
                # or if it's not a protected directory
                if not preserve_structure or not is_protected_directory(item_path):
                    shutil.rmtree(item_path)
                else:
                    # Recursively clean but preserve the directory itself
                    clean_directory_contents(item_path, preserve_structure)
    except Exception as e:
        # Log error but don't raise - partial cleanup is better than failure
        print(f"Warning: Error cleaning directory {directory_path}: {str(e)}")

def safe_clean_processing_directories(exclude_input=True):
    """
    DEPRECATED: 旧的清理方法，已被新的视频管理系统取代
    此函数保留用于向后兼容，但应避免使用
    """
    print("⚠️ Warning: safe_clean_processing_directories is deprecated. Use VideoFileManager.safe_overwrite_* methods instead.")
    
    import os
    import shutil
    
    paths = get_storage_paths()
    
    for path_type, path in paths.items():
        # CRITICAL SAFETY: Never clean input directory
        if exclude_input and path_type == 'input':
            continue
            
        # Only clean temp and output directories
        if path_type in ['temp', 'output'] and path != paths['input'] and os.path.exists(path):
            for item in os.listdir(path):
                item_path = os.path.join(path, item)
                try:
                    if os.path.isfile(item_path):
                        os.remove(item_path)
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                except Exception as e:
                    print(f"Warning: Could not remove {item_path}: {str(e)}")

def get_video_specific_paths(video_id: str):
    """
    获取特定视频ID的文件路径
    新架构：每个视频有独立的文件组织结构
    """
    from core.utils.video_manager import get_video_manager
    manager = get_video_manager()
    return manager.get_video_paths(video_id)

def get_or_create_video_id(video_path: str = None) -> str:
    """
    获取或创建当前视频的ID
    这是新架构的入口点
    """
    import os
    from core.utils.video_manager import get_video_manager
    manager = get_video_manager()
    
    # 尝试获取当前视频ID
    current_id = manager.get_current_video_id()
    if current_id:
        return current_id
    
    # 如果没有当前视频且提供了路径，注册新视频
    if video_path and os.path.exists(video_path):
        return manager.register_video(video_path)
    
    raise ValueError("No current video found and no valid video path provided")

if __name__ == "__main__":
    print(load_key('language_split_with_space'))
