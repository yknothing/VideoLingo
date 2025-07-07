import os
import json
import re
from threading import Lock
import json_repair
from openai import OpenAI
from core.utils.config_utils import load_key, get_storage_paths
from rich import print as rprint
from core.utils.decorator import except_handler

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

def sanitize_for_logging(data):
    """
    Sanitize data for safe logging - remove or mask sensitive information
    """
    if isinstance(data, dict):
        sanitized = {}
        for key, value in data.items():
            if key.lower() in ['api_key', 'password', 'token', 'secret']:
                sanitized[key] = "[REDACTED]"
            elif key in ['prompt', 'resp_content'] and not is_logging_enabled():
                # Mask user content when detailed logging is disabled
                sanitized[key] = mask_sensitive_content(str(value))
            else:
                sanitized[key] = sanitize_for_logging(value)
        return sanitized
    elif isinstance(data, list):
        return [sanitize_for_logging(item) for item in data]
    elif isinstance(data, str):
        # Remove potential API keys or sensitive patterns
        patterns_to_mask = [
            r'sk-[a-zA-Z0-9]{48}',  # OpenAI API keys
            r'sk-or-v1-[a-zA-Z0-9-]{64}',  # OpenRouter API keys
            r'Bearer [a-zA-Z0-9-_\.]{20,}',  # Bearer tokens
        ]
        
        result = data
        for pattern in patterns_to_mask:
            result = re.sub(pattern, '[API_KEY_REDACTED]', result)
        return result
    else:
        return data

# ------------
# cache gpt response
# ------------

def _save_cache(model, prompt, resp_content, resp_type, resp, message=None, log_title="default"):
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

def _load_cache(prompt, resp_type, log_title):
    with LOCK:
        gpt_log_folder = get_gpt_log_folder()
        file = os.path.join(gpt_log_folder, f"{log_title}.json")
        if os.path.exists(file):
            with open(file, 'r', encoding='utf-8') as f:
                for item in json.load(f):
                    if item["prompt"] == prompt and item["resp_type"] == resp_type:
                        return item["resp"]
        return False

# ------------
# ask gpt once
# ------------

@except_handler("GPT request failed", retry=5)
def ask_gpt(prompt, resp_type=None, valid_def=None, log_title="default"):
    # Prefer environment variables over config file to avoid plaintext secrets
    # Supported env names: VIDEO_LINGO_API_KEY / OPENROUTER_API_KEY / OPENAI_API_KEY
    env_api_key = os.getenv("VIDEO_LINGO_API_KEY") or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    api_key = env_api_key if env_api_key else load_key("api.key")

    # 如果依然未配置，报错提示
    if not api_key or api_key.lower().startswith("your-"):
        raise ValueError("API key is not set (env: VIDEO_LINGO_API_KEY / OPENROUTER_API_KEY / OPENAI_API_KEY)")

    # check cache
    cached = _load_cache(prompt, resp_type, log_title)
    if cached:
        rprint("use cache response")
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
    client = OpenAI(api_key=api_key, base_url=base_url)
    response_format = {"type": "json_object"} if resp_type == "json" and load_key("api.llm_support_json") else None

    messages = [{"role": "user", "content": prompt}]

    # Try each model in priority order
    last_error = None
    for model in models_to_try:
        try:
            params = dict(
                model=model,
                messages=messages,
                response_format=response_format,
                timeout=300
            )
            resp_raw = client.chat.completions.create(**params)
            # Success - break out of retry loop
            break
        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            if "429" in error_str or "rate_limit" in error_str or "quota" in error_str:
                rprint(f"[yellow]Rate limit/quota exceeded for {model}, trying next model...[/yellow]")
                if model != models_to_try[-1]:  # Not the last model
                    continue
            # For non-rate-limit errors or if this is the last model, re-raise
            raise e
    else:
        # All models failed
        raise last_error or Exception("All fallback models failed")

    # process and return full result
    resp_content = resp_raw.choices[0].message.content
    if resp_type == "json":
        resp = json_repair.loads(resp_content)
    else:
        resp = resp_content
    
    # check if the response format is valid
    if valid_def:
        valid_resp = valid_def(resp)
        if valid_resp['status'] != 'success':
            _save_cache(model, prompt, resp_content, resp_type, resp, log_title="error", message=valid_resp['message'])
            raise ValueError(f"❎ API response error: {valid_resp['message']}")

    _save_cache(model, prompt, resp_content, resp_type, resp, log_title=log_title)
    return resp


if __name__ == '__main__':
    from rich import print as rprint
    
    result = ask_gpt("""test respond ```json\n{\"code\": 200, \"message\": \"success\"}\n```""", resp_type="json")
    rprint(f"Test json output result: {result}")
