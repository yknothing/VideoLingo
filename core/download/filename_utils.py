"""
Filename utilities for video download operations
Single responsibility: Handle filename sanitization and validation
"""

import re
import os


def sanitize_filename(filename):
    """
    Sanitize filename by removing illegal characters
    
    Args:
        filename (str): Original filename
        
    Returns:
        str: Sanitized filename safe for filesystem
    """
    # Remove or replace illegal characters
    filename = re.sub(r'[<>:"/\\|?*]', '', filename)
    # Ensure filename doesn't start or end with a dot or space
    filename = filename.strip('. ')
    # Use default name if filename is empty
    return filename if filename else 'video'


def validate_filename_safety(filename):
    """
    Validate that filename is safe for cross-platform use
    
    Args:
        filename (str): Filename to validate
        
    Returns:
        tuple: (is_valid, error_message)
    """
    if not filename:
        return False, "Filename cannot be empty"
        
    # Check for reserved names on Windows
    reserved_names = {
        'CON', 'PRN', 'AUX', 'NUL',
        'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9',
        'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'
    }
    
    name_without_ext = os.path.splitext(filename)[0].upper()
    if name_without_ext in reserved_names:
        return False, f"Filename '{filename}' uses reserved name"
        
    # Check length (255 chars is typical filesystem limit)
    if len(filename) > 255:
        return False, "Filename too long (max 255 characters)"
        
    # Check for illegal characters
    illegal_chars = '<>:"/\\|?*'
    for char in illegal_chars:
        if char in filename:
            return False, f"Filename contains illegal character: {char}"
            
    return True, "Filename is valid"


def generate_safe_filename(original_name, max_length=200):
    """
    Generate a safe filename from original name
    
    Args:
        original_name (str): Original filename
        max_length (int): Maximum length for filename
        
    Returns:
        str: Safe filename
    """
    # Sanitize first
    safe_name = sanitize_filename(original_name)
    
    # Truncate if too long, preserving extension
    if len(safe_name) > max_length:
        name, ext = os.path.splitext(safe_name)
        # Reserve space for extension
        max_name_length = max_length - len(ext)
        safe_name = name[:max_name_length] + ext
        
    return safe_name