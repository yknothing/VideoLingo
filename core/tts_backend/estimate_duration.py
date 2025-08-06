
# TTS Duration estimation - 安全版本
from core.constants import AudioConstants
def estimate_duration(text, language='en'):
    '''简单的持续时间估算'''
    # 基于字符数的简单估算
    chars_per_second = 15  # 平均每秒字符数
    base_duration = len(text) / chars_per_second
    return max(AudioConstants.MIN_BASE_DURATION, base_duration)  # 最少1秒

def init_estimator(language='en'):
    '''初始化估算器 - 简化版本'''
    return True
