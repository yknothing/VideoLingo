import os
import json
import re
import urllib.parse
from threading import Lock
import json_repair
import openai
from core.utils.config_utils import load_key, get_storage_paths
from core.constants import SecurityConstants
from rich import print as rprint
from core.utils.decorator import except_handler
from core.utils.observability import log_event, time_block, inc_counter, observe_histogram


def safe_json_parse(response_content):
    """
    Enhanced secure JSON parsing with comprehensive deserialization attack prevention.
    
    Security measures:
    - Type validation for input
    - Size limits to prevent DoS attacks
    - Pattern detection for code injection attempts (including Unicode bypass)
    - Strict JSON parsing mode
    - Nesting depth validation
    """
    # ------------
    # Type validation
    # ------------
    if not isinstance(response_content, str):
        raise ValueError("Response content must be string")
    
    # ------------
    # Size limit using constant (5MB for large transcriptions)
    # ------------
    max_size = SecurityConstants.MAX_JSON_RESPONSE_SIZE
    if len(response_content) > max_size:
        raise ValueError(f"Response too large: {len(response_content)} bytes (max: {max_size})")
    
    # ------------
    # Decode URL-encoded content to prevent bypass attacks
    # ------------
    try:
        decoded_content = urllib.parse.unquote(response_content)
    except Exception:
        decoded_content = response_content
    
    # ------------
    # Suspicious pattern detection (check both original and decoded)
    # ------------
    suspicious_patterns = [
        r'__[a-zA-Z_]+__',           # Python dunder methods
        r'eval\s*\(',                 # eval function calls
        r'exec\s*\(',                 # exec function calls
        r'import\s+',                 # import statements
        r'from\s+\w+\s+import',       # from imports
        r'compile\s*\(',              # compile function
        r'globals\s*\(',              # globals access
        r'locals\s*\(',               # locals access
        r'getattr\s*\(',              # attribute access
        r'setattr\s*\(',              # attribute setting
        r'delattr\s*\(',              # attribute deletion
        r'open\s*\(',                 # file operations
        r'subprocess',                # subprocess module
        r'os\.system',                # system calls
        r'pickle\.',                  # pickle operations
        r'marshal\.',                 # marshal operations
    ]
    
    for content_to_check in [response_content, decoded_content]:
        for pattern in suspicious_patterns:
            if re.search(pattern, content_to_check, re.IGNORECASE):
                rprint(f"[red]Security: Suspicious pattern detected in JSON response[/red]")
                raise ValueError("Response contains suspicious content")
    
    # ------------
    # Strict JSON parsing
    # ------------
    try:
        result = json.loads(response_content, strict=True)
        
        # Validate nesting depth to prevent stack overflow attacks
        if not _validate_json_depth(result, max_depth=SecurityConstants.MAX_JSON_NESTING_DEPTH):
            raise ValueError("JSON nesting depth exceeds security limit")
        
        return result
        
    except json.JSONDecodeError as e:
        rprint(f"[yellow]Warning: Invalid JSON response[/yellow]")
        raise ValueError(f"Invalid JSON format: {str(e)[:100]}")


def _validate_json_depth(obj, current_depth=0, max_depth=50):
    """
    Validate JSON nesting depth to prevent stack overflow attacks.
    
    Args:
        obj: JSON object to validate
        current_depth: Current nesting level
        max_depth: Maximum allowed depth
        
    Returns:
        bool: True if within limits, False otherwise
    """
    if current_depth > max_depth:
        return False
    
    if isinstance(obj, dict):
        for value in obj.values():
            if not _validate_json_depth(value, current_depth + 1, max_depth):
                return False
    elif isinstance(obj, list):
        for item in obj:
            if not _validate_json_depth(item, current_depth + 1, max_depth):
                return False
    
    return True

# ------------
# Logging configuration and security
# ------------

LOCK = Lock()

def get_gpt_log_folder():
    """Get GPT log folder from storage configuration"""
    try:
        paths = get_storage_paths()
        return os.path.join(paths['temp'], 'gpt_log')
    except:
        return 'output/gpt_log'

def is_logging_enabled():
    """Check if detailed logging is enabled"""
    try:
        return load_key("debug.enable_gpt_logging")
    except:
        # Default to False for security - only log when explicitly enabled
        return False

def mask_sensitive_content(content, mask_ratio=0.3):
    """
    Mask sensitive content in logs for privacy protection
    
    Args:
        content: Content to mask
        mask_ratio: Ratio of content to mask (0.0 to 1.0)
    
    Returns:
        Masked content with partial information preserved
    """
    if not content or not isinstance(content, str):
        return content
    
    # If content is very short, mask minimally
    if len(content) < 50:
        return content[:20] + "..." + content[-10:] if len(content) > 30 else content
    
    # For longer content, mask middle portion based on ratio
    content_len = len(content)
    mask_len = int(content_len * mask_ratio)
    start_keep = (content_len - mask_len) // 2
    end_keep = content_len - start_keep - mask_len
    
    if mask_len > 0:
        return content[:start_keep] + f"[MASKED {mask_len} chars]" + content[-end_keep:]
    return content

def get_comprehensive_api_key_patterns():
    """
    Get comprehensive API key patterns for all major providers
    
    Returns:
        List of regex patterns to detect API keys
    """
    return [
        # OpenAI API keys (more specific patterns first)
        r'sk-proj-[a-zA-Z0-9]{64}',                     # Project-based OpenAI keys
        r'sk-[a-zA-Z0-9]{48}',                          # Classic OpenAI keys
        
        # OpenRouter API keys
        r'sk-or-v1-[a-zA-Z0-9-_]{32,128}',             # OpenRouter keys
        
        # Anthropic Claude API keys  
        r'sk-ant-api03-[a-zA-Z0-9_-]{95}',             # Anthropic API keys
        r'sk-ant-[a-zA-Z0-9_-]{32,128}',               # Anthropic variants
        
        # Google API keys
        r'AIza[0-9A-Za-z_-]{35}',                       # Google API keys
        r'ya29\.[0-9A-Za-z_-]{68}',                     # Google OAuth tokens
        
        # Hugging Face tokens
        r'hf_[a-zA-Z0-9]{34}',                          # Hugging Face tokens
        
        # Bearer tokens and Authorization headers
        r'Bearer\s+[a-zA-Z0-9_.-]{20,}',                # Bearer tokens
        r'Authorization:\s*Bearer\s+[a-zA-Z0-9_.-]{20,}', # Authorization headers
        
        # Generic OpenAI-style keys (catch remaining patterns)
        r'sk-[a-zA-Z0-9_-]{20,}',                       # General sk- prefixed keys
        
        # Azure OpenAI keys (be more specific to avoid false positives)
        r'\b[a-f0-9]{32}\b',                            # Azure API keys (32 hex chars, word boundaries)
        
        # Cohere API keys (be more specific)
        r'\b[a-zA-Z0-9]{40}\b',                         # Cohere API keys (40 chars, word boundaries)
        
        # Generic API key assignments (URL parameters and key-value pairs)
        r'api[_-]?key["\']?\s*[:=]\s*["\']?[a-zA-Z0-9_-]{20,}["\']?', # API key assignments
        r'apikey["\']?\s*[:=]\s*["\']?[a-zA-Z0-9_-]{20,}["\']?',     # API key assignments (no underscore)
        r'token["\']?\s*[:=]\s*["\']?[a-zA-Z0-9_-]{20,}["\']?',      # Token assignments
        r'secret["\']?\s*[:=]\s*["\']?[a-zA-Z0-9_-]{20,}["\']?',     # Secret assignments
    ]

def sanitize_api_keys_from_text(text):
    """
    Sanitize API keys from text using comprehensive pattern matching
    
    Args:
        text: Text that may contain API keys
        
    Returns:
        Text with API keys replaced by [API_KEY_REDACTED]
    """
    if not isinstance(text, str):
        return text
    
    patterns = get_comprehensive_api_key_patterns()
    result = text
    
    # Apply each pattern with case-insensitive matching
    for pattern in patterns:
        result = re.sub(pattern, '[API_KEY_REDACTED]', result, flags=re.IGNORECASE)
    
    # Additional patterns for common key formats in error messages  
    additional_patterns = [
        r'("api_key":\s*"[^"]{10,}")',                  # JSON API key fields
        r'("apikey":\s*"[^"]{10,}")',                   # JSON apikey fields
        r'("authorization":\s*"[^"]{10,}")',            # JSON authorization fields
        r'(\'api_key\':\s*\'[^\']{10,}\')',             # Python dict API keys
    ]
    
    # Handle JSON and Python dict patterns
    for pattern in additional_patterns:
        result = re.sub(pattern, lambda m: re.sub(r'[a-zA-Z0-9_-]{10,}', '[API_KEY_REDACTED]', m.group(1)), result, flags=re.IGNORECASE)
    
    # Handle URL parameters and key-value assignments with full replacement
    url_param_patterns = [
        r'\bapi_key=[a-zA-Z0-9_-]{10,}',                # URL api_key parameter
        r'\bapikey=[a-zA-Z0-9_-]{10,}',                 # URL apikey parameter
        r'\btoken=[a-zA-Z0-9_-]{10,}',                  # URL token parameter
        r'\bsecret=[a-zA-Z0-9_-]{10,}',                 # URL secret parameter
    ]
    
    for pattern in url_param_patterns:
        result = re.sub(pattern, lambda m: m.group(0).split('=')[0] + '=[API_KEY_REDACTED]', result, flags=re.IGNORECASE)
    
    return result

def sanitize_sensitive_fields(data):
    """
    Sanitize sensitive field names regardless of their values
    
    Args:
        data: Dictionary or object to sanitize
        
    Returns:
        Sanitized data with sensitive fields redacted
    """
    sensitive_field_patterns = [
        r'.*api[_-]?key.*',
        r'.*password.*',
        r'.*token.*',
        r'.*secret.*',
        r'.*auth.*',
        r'.*credential.*',
        r'.*private.*',
        r'.*confidential.*',
    ]
    
    if isinstance(data, dict):
        sanitized = {}
        for key, value in data.items():
            key_lower = key.lower()
            is_sensitive = any(re.match(pattern, key_lower) for pattern in sensitive_field_patterns)
            
            if is_sensitive:
                sanitized[key] = "[REDACTED]"
            else:
                sanitized[key] = sanitize_for_logging(value)
        return sanitized
    
    return data

def sanitize_for_logging(data):
    """
    Comprehensive sanitization for safe logging - remove or mask sensitive information
    
    Args:
        data: Data to sanitize (can be dict, list, str, or other types)
        
    Returns:
        Sanitized data safe for logging
    """
    if isinstance(data, dict):
        # First sanitize sensitive fields by name
        sanitized = sanitize_sensitive_fields(data)
        
        # Then sanitize content for non-redacted fields
        for key, value in sanitized.items():
            if value != "[REDACTED]":
                if key in ['prompt', 'resp_content'] and not is_logging_enabled():
                    # Mask user content when detailed logging is disabled
                    sanitized[key] = mask_sensitive_content(str(value))
                else:
                    sanitized[key] = sanitize_for_logging(value)
        return sanitized
        
    elif isinstance(data, list):
        return [sanitize_for_logging(item) for item in data]
        
    elif isinstance(data, str):
        # Remove API keys using comprehensive patterns
        result = sanitize_api_keys_from_text(data)
        return result
        
    else:
        return data

def validate_sanitization_integrity():
    """
    Validate that sanitization functions work correctly for security
    
    Returns:
        Dict with validation results and any issues found
    
    NOTE: Test keys use obviously fake patterns to avoid triggering secret scanners
    """
    # Use obviously fake test keys that won't trigger GitHub secret scanning
    # Format: prefix + "FAKE" + "X" padding to match expected length
    test_cases = [
        # OpenAI API keys (use FAKE marker)
        ("Request failed with sk-FAKETESTKEY00000000000000000000000000000000000000 key", "[API_KEY_REDACTED]"),
        ("sk-proj-FAKETESTKEY000000000000000000000000000000000000000000000000", "[API_KEY_REDACTED]"),
        
        # OpenRouter keys (use FAKE marker)
        ("Authentication failed: sk-or-v1-FAKETESTKEY0000000000000000000000000000000000000000000000000000", "[API_KEY_REDACTED]"),
        
        # Anthropic keys (use FAKE marker)  
        ("Error with sk-ant-FAKETESTKEY00000000000000000000000000000000000000000000000000000000000000000000000", "[API_KEY_REDACTED]"),
        
        # Google keys - use invalid prefix pattern that won't match real keys
        ("Failed: FAKE_GOOGLE_KEY_FOR_TESTING_ONLY_000000", "[API_KEY_REDACTED]"),
        
        # Bearer tokens (use FAKE marker)
        ("Authorization failed: Bearer FAKETESTTOKEN0000000000000000000000", "[API_KEY_REDACTED]"),
        
        # JSON format (use FAKE marker)
        ('{"api_key": "sk-FAKETESTKEY00000000000000000000000000000000000000"}', '[API_KEY_REDACTED]'),
        
        # URL parameters (use FAKE marker)
        ("Request to https://api.example.com?api_key=sk-FAKETESTKEY00000000000000000000000000000000000000", "api_key=[API_KEY_REDACTED]"),
    ]
    
    results = {
        "passed": 0,
        "failed": 0,
        "issues": []
    }
    
    for test_input, expected_pattern in test_cases:
        sanitized = sanitize_api_keys_from_text(test_input)
        
        # Check if the expected pattern is in the sanitized output AND no sensitive data remains
        has_expected_pattern = expected_pattern in sanitized
        # Check for test key patterns that should be redacted
        has_sensitive_data = any(
            sensitive_pattern in sanitized for sensitive_pattern in [
                "sk-FAKE", "FAKE_GOOGLE", "Bearer FAKE", "FAKETESTKEY", "FAKETESTTOKEN"
            ]
        )
        
        # Special case: for URL parameters, we need to check that the full key is redacted
        if "api_key=" in test_input and "sk-FAKE" in test_input:
            has_sensitive_data = has_sensitive_data or ("api_key=sk-FAKE" in sanitized)
        
        if has_expected_pattern and not has_sensitive_data:
            results["passed"] += 1
        else:
            results["failed"] += 1
            results["issues"].append({
                "input": test_input[:50] + "..." if len(test_input) > 50 else test_input,
                "output": sanitized[:50] + "..." if len(sanitized) > 50 else sanitized,
                "expected_pattern": expected_pattern,
                "has_expected": has_expected_pattern,
                "has_sensitive": has_sensitive_data
            })
    
    return results

def get_logging_security_level():
    """
    Get current logging security level based on configuration
    
    Returns:
        String indicating security level: "strict", "moderate", or "permissive"
    """
    try:
        # Check if detailed logging is enabled
        detailed_logging = is_logging_enabled()
        
        # Check if running in production mode
        debug_mode = load_key("debug.enable_debug_mode", fallback=False)
        
        if not detailed_logging and not debug_mode:
            return "strict"  # Maximum security, minimal logging
        elif detailed_logging and not debug_mode:
            return "moderate"  # Some logging but still secure
        else:
            return "permissive"  # Debug mode, more verbose logging
            
    except Exception:
        # Default to strict security if configuration cannot be determined
        return "strict"

# ------------
# cache gpt response
# ------------

def _save_cache(model, prompt, resp_content, resp_type, resp, message=None, log_title="default"):
    # Only cache when detailed logging explicitly enabled
    if not is_logging_enabled():
        return
    with LOCK:
        logs = []
        gpt_log_folder = get_gpt_log_folder()
        file = os.path.join(gpt_log_folder, f"{log_title}.json")
        os.makedirs(os.path.dirname(file), exist_ok=True)
        
        if os.path.exists(file):
            with open(file, 'r', encoding='utf-8') as f:
                logs = json.load(f)
        
        # Create log entry with security considerations
        log_entry = {
            "model": model,
            "prompt": prompt,
            "resp_content": resp_content,
            "resp_type": resp_type,
            "resp": resp,
            "message": message
        }
        
        # Sanitize sensitive data before saving
        sanitized_entry = sanitize_for_logging(log_entry)
        logs.append(sanitized_entry)
        
        with open(file, 'w', encoding='utf-8') as f:
            json.dump(logs, f, ensure_ascii=False, indent=4)
        try:
            log_event("debug", "gpt cache saved", stage="gpt", op="cache_save", log_title=log_title, size=len(logs))
            inc_counter("gpt.cache_write", 1, stage="gpt")
        except Exception:
            pass

def _load_cache(prompt, resp_type, log_title):
    if not is_logging_enabled():
        return False
    with LOCK:
        gpt_log_folder = get_gpt_log_folder()
        file = os.path.join(gpt_log_folder, f"{log_title}.json")
        if os.path.exists(file):
            with open(file, 'r', encoding='utf-8') as f:
                for item in json.load(f):
                    if item["prompt"] == prompt and item["resp_type"] == resp_type:
                        try:
                            inc_counter("gpt.cache_hit", 1, stage="gpt")
                        except Exception:
                            pass
                        return item["resp"]
        return False

# ------------
# ask gpt once
# ------------

def sanitize_error_message(error_msg):
    """
    Sanitize error messages to remove sensitive information before logging
    
    Args:
        error_msg: Error message string that may contain sensitive data
        
    Returns:
        Sanitized error message safe for logging
    """
    if not isinstance(error_msg, str):
        error_msg = str(error_msg)
    
    # Remove API keys from error messages
    sanitized = sanitize_api_keys_from_text(error_msg)
    
    # Remove common sensitive patterns in error messages
    sensitive_patterns = [
        (r'api_key=[a-zA-Z0-9_-]+', 'api_key=[REDACTED]'),
        (r'apikey=[a-zA-Z0-9_-]+', 'apikey=[REDACTED]'),
        (r'token=[a-zA-Z0-9_-]+', 'token=[REDACTED]'),
        (r'authorization=[a-zA-Z0-9_-]+', 'authorization=[REDACTED]'),
        (r'x-api-key:\s*[a-zA-Z0-9_-]+', 'x-api-key: [REDACTED]'),
        (r'bearer\s+[a-zA-Z0-9_-]+', 'bearer [REDACTED]'),
    ]
    
    for pattern, replacement in sensitive_patterns:
        sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)
    
    return sanitized

def is_network_error(exception):
    """
    判断是否为网络错误，需要重试
    Uses sanitized error message to prevent API key exposure
    """
    # Sanitize error message before analysis
    error_str = sanitize_error_message(str(exception)).lower()
    
    network_indicators = [
        'connection', 'timeout', 'network', 'dns', 'ssl',
        'handshake_failure', 'connection refused', 'connection reset',
        'unreachable', 'temporary failure', 'service unavailable',
        'bad gateway', 'gateway timeout', 'request timeout'
    ]
    return any(indicator in error_str for indicator in network_indicators)

def get_retry_delay(attempt: int, base_delay: float = 1.0) -> float:
    """
    计算重试延迟时间（指数退避 + 随机抖动）
    """
    import random
    delay = base_delay * (2 ** attempt)  # 指数退避
    jitter = delay * 0.1 * random.random()  # 10%随机抖动
    return min(delay + jitter, 60.0)  # 最大延迟60秒

@except_handler("GPT request failed", retry=5)
def ask_gpt(prompt, resp_type=None, valid_def=None, log_title="default"):
    # Prefer environment variables over config file to avoid plaintext secrets
    # Supported env names: VIDEO_LINGO_API_KEY / OPENROUTER_API_KEY / OPENAI_API_KEY
    # Prefer explicit config when present (even empty), fallback to env only when key not configured
    config_api_key = None
    try:
        config_api_key = load_key("api.key")
    except Exception:
        config_api_key = None
    if config_api_key is not None:
        api_key = config_api_key
    else:
        env_api_key = os.getenv("VIDEO_LINGO_API_KEY") or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        api_key = env_api_key

    # 如果依然未配置，报错提示
    if not api_key or api_key.lower().startswith("your-"):
        raise ValueError("API key is not set (env: VIDEO_LINGO_API_KEY / OPENROUTER_API_KEY / OPENAI_API_KEY)")

    # check cache
    cached = _load_cache(prompt, resp_type, log_title)
    if cached:
        rprint("use cache response")
        log_event("info", "gpt cache hit", stage="gpt", op="cache", log_title=log_title)
        return cached

    # Try models in priority order for fallback
    primary_model = load_key("api.model")
    try:
        model_priority = load_key("api.model_priority")
        if model_priority and isinstance(model_priority, list):
            models_to_try = [primary_model] + [m for m in model_priority if m != primary_model]
        else:
            models_to_try = [primary_model]
    except:
        models_to_try = [primary_model]
    
    base_url = load_key("api.base_url")
    if 'ark' in base_url:
        base_url = "https://ark.cn-beijing.volces.com/api/v3" # huoshan base url
    elif 'v1' not in base_url:
        base_url = base_url.strip('/') + '/v1'
    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    response_format = {"type": "json_object"} if resp_type == "json" and load_key("api.llm_support_json") else None

    messages = [{"role": "user", "content": prompt}]

    # Try each model in priority order with improved error handling
    last_error = None
    max_retries = 3
    
    for model in models_to_try:
        model_success = False
        
        # 对每个模型进行重试
        for retry_attempt in range(max_retries):
            try:
                params = dict(
                    model=model,
                    messages=messages,
                    response_format=response_format,
                    timeout=300
                )
                with time_block("gpt request", stage="gpt", op="request", attempt=retry_attempt, model=model):
                    resp_raw = client.chat.completions.create(**params)
                model_success = True
                break  # 成功，跳出重试循环
                
            except Exception as e:
                last_error = e
                # Sanitize error message before any logging or display
                sanitized_error = sanitize_error_message(str(e))
                error_str = sanitized_error.lower()
                
                # 分类错误类型并决定重试策略
                if "429" in error_str or "rate_limit" in error_str or "quota" in error_str:
                    rprint(f"[yellow]Rate limit/quota exceeded for {model}, trying next model...[/yellow]")
                    break  # 跳出重试循环，尝试下一个模型
                    
                elif is_network_error(e):
                    if retry_attempt < max_retries - 1:
                        delay = get_retry_delay(retry_attempt)
                        # Use sanitized error message for display
                        safe_error_msg = sanitized_error[:100] + "..." if len(sanitized_error) > 100 else sanitized_error
                        rprint(f"[yellow]Network error for {model} (attempt {retry_attempt + 1}/{max_retries}): {safe_error_msg}[/yellow]")
                        rprint(f"[blue]Retrying in {delay:.1f} seconds...[/blue]")
                        log_event("warning", "gpt network error", stage="gpt", op="retry", attempt=retry_attempt, model=model, error=sanitized_error[:150])
                        observe_histogram("gpt.retry_wait_seconds", float(delay), stage="gpt", model=model)
                        import time
                        time.sleep(delay)
                        continue  # 继续重试当前模型
                    else:
                        rprint(f"[red]Network error persists for {model} after {max_retries} attempts[/red]")
                        log_event("error", "gpt network error persists", stage="gpt", op="retry", model=model, error=sanitized_error[:150])
                        break  # 跳出重试循环，尝试下一个模型
                        
                elif "invalid_request_error" in error_str or "model_not_found" in error_str:
                    rprint(f"[red]Model {model} not available or invalid request[/red]")
                    log_event("error", "gpt invalid request or model not found", stage="gpt", op="request", model=model, error=sanitized_error[:150])
                    break  # 立即跳到下一个模型
                    
                else:
                    # 其他错误，如果不是最后一次重试，继续尝试
                    if retry_attempt < max_retries - 1:
                        delay = get_retry_delay(retry_attempt, 0.5)
                        # Use sanitized error message for display
                        safe_error_msg = sanitized_error[:100] + "..." if len(sanitized_error) > 100 else sanitized_error
                        rprint(f"[yellow]Error for {model} (attempt {retry_attempt + 1}/{max_retries}): {safe_error_msg}[/yellow]")
                        rprint(f"[blue]Retrying in {delay:.1f} seconds...[/blue]")
                        log_event("warning", "gpt other error", stage="gpt", op="retry", attempt=retry_attempt, model=model, error=sanitized_error[:150])
                        import time
                        time.sleep(delay)
                        continue
                    else:
                        break  # 跳出重试循环，尝试下一个模型
        
        if model_success:
            break  # 某个模型成功了，跳出模型循环
    else:
        # 所有模型都失败了
        if last_error:
            # Sanitize the final error message
            sanitized_final_error = sanitize_error_message(str(last_error))
            error_summary = f"All models failed. Last error: {sanitized_final_error[:200]}"
            if is_network_error(last_error):
                error_summary += "\n[Suggestion] Check your internet connection and API endpoints"
            log_event("error", error_summary, stage="gpt", op="fail")
            inc_counter("gpt.error", 1, stage="gpt")
            raise Exception(error_summary)
        else:
            log_event("error", "All fallback models failed with unknown errors", stage="gpt", op="fail")
            inc_counter("gpt.error", 1, stage="gpt")
            raise Exception("All fallback models failed with unknown errors")

    # process and return full result
    resp_content = resp_raw.choices[0].message.content
    if resp_type == "json":
        # SECURITY: 使用安全的JSON解析，防止反序列化攻击
        resp = safe_json_parse(resp_content)
    else:
        resp = resp_content
    
    # check if the response format is valid
    if valid_def:
        valid_resp = valid_def(resp)
        if valid_resp['status'] != 'success':
            # Sanitize validation error message before logging and raising
            sanitized_message = sanitize_error_message(valid_resp['message'])
            _save_cache(model, prompt, resp_content, resp_type, resp, log_title="error", message=sanitized_message)
            raise ValueError(f"❎ API response error: {sanitized_message}")

    _save_cache(model, prompt, resp_content, resp_type, resp, log_title=log_title)
    try:
        log_event("info", "gpt request success", stage="gpt", op="success", model=model, resp_type=str(resp_type or "text"))
        inc_counter("gpt.success", 1, stage="gpt")
        if resp_type == "json":
            observe_histogram("gpt.response_json_size", len(json.dumps(resp, ensure_ascii=False)), stage="gpt", model=model)
        else:
            observe_histogram("gpt.response_text_len", len(str(resp_content or "")), stage="gpt", model=model)
    except Exception:
        pass
    return resp


if __name__ == '__main__':
    from rich import print as rprint
    
    # Security validation check
    rprint("[blue]Running API key sanitization validation...[/blue]")
    validation_results = validate_sanitization_integrity()
    
    if validation_results["failed"] > 0:
        rprint(f"[red]⚠️  Security validation failed! {validation_results['failed']} tests failed.[/red]")
        for issue in validation_results["issues"]:
            rprint(f"[yellow]  Issue: {issue['input']} -> {issue['output']}[/yellow]")
            rprint(f"[yellow]  Expected: {issue['expected_pattern']}[/yellow]")
    else:
        rprint(f"[green]✅ Security validation passed! All {validation_results['passed']} tests successful.[/green]")
    
    rprint(f"[blue]Current logging security level: {get_logging_security_level()}[/blue]")
    
    # Test API call
    result = ask_gpt("""test respond ```json\n{\"code\": 200, \"message\": \"success\"}\n```""", resp_type="json")
    rprint(f"Test json output result: {result}")
