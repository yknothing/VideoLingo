# ------------
# Security utilities for VideoLingo
# ------------

import os
import re
import uuid
from pathlib import Path, PurePath
from urllib.parse import urlparse
import unicodedata

def sanitize_filename(filename, max_length=200):
    """
    Sanitize filename to prevent directory traversal and ensure cross-platform compatibility
    保持文件名的可读性，只移除真正危险的字符
    
    Args:
        filename: Original filename
        max_length: Maximum allowed filename length
    
    Returns:
        Sanitized filename safe for filesystem use
    """
    if not filename:
        return str(uuid.uuid4())
    
    # ------------
    # 保持可读性的文件名清理策略
    # ------------
    
    # 只移除真正危险的字符，保留更多可读字符
    # 允许：字母、数字、空格、点、连字符、下划线、括号、中文等
    dangerous_chars = r'[<>:"/\\|?*\x00-\x1f]'
    sanitized = re.sub(dangerous_chars, '', filename)
    
    # 将多个空格替换为单个空格
    sanitized = re.sub(r'\s+', ' ', sanitized)
    
    # 移除目录遍历尝试
    sanitized = sanitized.replace('..', '')
    
    # Unicode标准化
    sanitized = unicodedata.normalize('NFKD', sanitized)
    
    # 移除首尾的点和空格（Windows兼容性）
    sanitized = sanitized.strip('. ')
    
    # 检查Windows保留名称
    reserved_names = {
        'CON', 'PRN', 'AUX', 'NUL', 'COM1', 'COM2', 'COM3', 'COM4', 'COM5',
        'COM6', 'COM7', 'COM8', 'COM9', 'LPT1', 'LPT2', 'LPT3', 'LPT4',
        'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'
    }
    
    name_without_ext = os.path.splitext(sanitized)[0].upper()
    if name_without_ext in reserved_names:
        sanitized = f"video_{sanitized}"
    
    # 长度截断，优先保留文件扩展名
    if len(sanitized) > max_length:
        name, ext = os.path.splitext(sanitized)
        available_length = max_length - len(ext)
        if available_length > 0:
            sanitized = name[:available_length] + ext
        else:
            # 如果扩展名太长，截断整个文件名
            sanitized = sanitized[:max_length]
    
    # 如果清理后为空，使用原始文件名的安全版本
    if not sanitized:
        # 提取文件扩展名
        _, ext = os.path.splitext(filename)
        safe_ext = re.sub(dangerous_chars, '', ext)
        sanitized = f"video_{str(uuid.uuid4())[:8]}{safe_ext}"
    
    return sanitized

def sanitize_path(base_path, user_path):
    """
    Safely combine base path with user-provided path to prevent directory traversal
    
    Args:
        base_path: Trusted base directory path
        user_path: User-provided path component
    
    Returns:
        Safe absolute path within base_path
    """
    if not base_path or not user_path:
        raise ValueError("Both base_path and user_path must be provided")
    
    # Convert to Path objects for safe handling
    base = Path(base_path).resolve()
    
    # Extract just the filename from user path to prevent traversal
    safe_name = PurePath(user_path).name
    
    # Sanitize the filename
    safe_name = sanitize_filename(safe_name)
    
    # Combine safely
    result_path = base / safe_name
    
    # Ensure the result is within base_path
    try:
        result_path.resolve().relative_to(base)
    except ValueError:
        # Path is outside base_path, use fallback
        result_path = base / str(uuid.uuid4())
    
    return result_path

def generate_safe_filename(original_name=None, extension=None):
    """
    Generate a safe filename using UUID with optional original name preservation
    
    Args:
        original_name: Original filename (optional)
        extension: File extension (optional)
    
    Returns:
        Safe filename with UUID
    """
    safe_id = str(uuid.uuid4())
    
    if extension:
        if not extension.startswith('.'):
            extension = '.' + extension
        return safe_id + extension
    
    if original_name:
        _, ext = os.path.splitext(original_name)
        if ext:
            return safe_id + ext
    
    return safe_id

def validate_url(url):
    """
    Validate URL format and ensure it uses safe protocols
    
    Args:
        url: URL to validate
    
    Returns:
        bool: True if URL is safe to use
    """
    if not url:
        return False
    
    try:
        parsed = urlparse(url)
        
        # Only allow http/https protocols
        if parsed.scheme not in ['http', 'https']:
            return False
        
        # Must have a netloc (domain)
        if not parsed.netloc:
            return False
        
        # Basic domain validation
        if not re.match(r'^[a-zA-Z0-9.-]+$', parsed.netloc.split(':')[0]):
            return False
        
        return True
    except Exception:
        return False

def validate_proxy_url(proxy_url):
    """
    Validate proxy URL format
    
    Args:
        proxy_url: Proxy URL to validate
    
    Returns:
        bool: True if proxy URL is safe to use
    """
    if not proxy_url:
        return False
    
    try:
        parsed = urlparse(proxy_url)
        
        # Only allow http/https/socks5 protocols for proxy
        if parsed.scheme not in ['http', 'https', 'socks5']:
            return False
        
        # Must have a netloc
        if not parsed.netloc:
            return False
        
        return True
    except Exception:
        return False

def safe_join_path(*args):
    """
    Safely join path components, preventing directory traversal
    
    Args:
        *args: Path components to join
    
    Returns:
        Safe joined path
    """
    if not args:
        return ""
    
    # Start with first argument as base
    result = Path(args[0])
    
    # Add each subsequent component safely
    for component in args[1:]:
        if component:
            # Remove any directory traversal attempts
            safe_component = str(component).replace('..', '').strip('/')
            if safe_component:
                result = result / safe_component
    
    return str(result)

def is_safe_path(path, allowed_roots=None):
    """
    Check if a path is safe (no traversal, within allowed roots)
    
    Args:
        path: Path to check
        allowed_roots: List of allowed root directories
    
    Returns:
        bool: True if path is safe
    """
    if not path:
        return False
    
    try:
        abs_path = Path(path).resolve()
        
        # Check for directory traversal indicators
        if '..' in str(abs_path):
            return False
        
        # If allowed_roots specified, ensure path is within one of them
        if allowed_roots:
            for root in allowed_roots:
                try:
                    abs_path.relative_to(Path(root).resolve())
                    return True
                except ValueError:
                    continue
            return False
        
        return True
    except Exception:
        return False 