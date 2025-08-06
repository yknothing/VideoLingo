# ------------
# VideoLingo Constants Management
# 统一管理所有硬编码数值，提高代码可维护性
# ------------

class AudioConstants:
    """音频处理相关常量"""
    # 时长相关
    MIN_AUDIO_DURATION = 3.0              # 最小音频时长（秒）
    MIN_BASE_DURATION = 1.0               # 基础最小时长（秒）
    TIME_TOLERANCE = 0.1                  # 时间误差容忍度（秒）
    DURATION_MULTIPLIER = 1.02            # 时长乘数
    TIME_DIFF_TOLERANCE = 0.6             # 时间差容忍度（秒）
    MIN_SUBTITLE_DURATION = 2.5           # 最小字幕时长（秒）
    TIME_EXTENSION_TOLERANCE = 1.5        # 时间扩展容忍度（秒）
    
    # 速度相关
    SPEED_FACTOR_PRECISION = 0.001        # 速度因子比较精度
    BASE_SPEED_FACTOR = 1.0               # 基准速度因子
    MAX_ACCEPTABLE_SPEED = 1.2            # 最大可接受速度
    MIN_SPEED_THRESHOLD = 0.7             # 最小速度阈值
    
    # 音频处理
    SILENT_AUDIO_DURATION = 100           # 静音音频时长（毫秒）
    SAMPLE_RATE_DEFAULT = 16000           # 默认采样率
    
class RetryConstants:
    """重试机制相关常量"""
    # 重试次数
    DEFAULT_MAX_RETRIES = 3               # 默认最大重试次数
    GPT_MAX_RETRIES = 5                   # GPT最大重试次数
    AUDIO_SPEED_MAX_RETRIES = 2           # 音频速度调整最大重试次数
    DOWNLOAD_MAX_RETRIES = 3              # 下载最大重试次数
    
    # 延迟时间
    DEFAULT_DELAY = 1.0                   # 默认延迟时间（秒）
    SHORT_DELAY = 0.1                     # 短延迟时间（秒）
    NETWORK_RETRY_DELAY = 0.5             # 网络重试延迟（秒）
    
class SubtitleConstants:
    """字幕处理相关常量"""
    # 长度限制
    DEFAULT_MAX_LENGTH = 75               # 默认最大字幕长度
    TARGET_MULTIPLIER = 1.2               # 目标语言长度乘数
    
    # 时间相关
    MIN_TRIM_DURATION = 3.5               # 最小修剪时长（秒）
    
    # 分割相关
    MAX_MERGE_COUNT = 5                   # 最大合并数量
    MIN_MERGE_THRESHOLD = 2               # 最小合并阈值
    
class NetworkConstants:
    """网络相关常量"""
    # 磁盘空间
    MIN_FREE_SPACE_GB = 1.0               # 最小可用空间（GB）
    
    # 网络速度
    MIN_DOWNLOAD_SPEED_MBPS = 0.1         # 最小下载速度（MB/s）
    
    # 服务器相关
    SERVER_STARTUP_TIMEOUT = 50           # 服务器启动超时（秒）
    DEFAULT_HOST = '127.0.0.1'            # 默认主机地址
    DEFAULT_PORT = 9880                   # 默认端口
    
class SimilarityConstants:
    """相似度和质量相关常量"""
    # 相似度阈值
    MIN_SIMILARITY_THRESHOLD = 0.9        # 最小相似度阈值
    
    # 质量评分阈值
    HIGH_QUALITY_THRESHOLD = 0.9          # 高质量阈值
    GOOD_QUALITY_THRESHOLD = 0.8          # 良好质量阈值
    ACCEPTABLE_QUALITY_THRESHOLD = 0.75   # 可接受质量阈值
    
class ProcessingConstants:
    """处理流程相关常量"""
    # 批处理
    DEFAULT_BATCH_SIZE = 15               # 默认批处理大小
    
    # 摘要
    DEFAULT_SUMMARY_LENGTH = 8000         # 默认摘要长度
    
    # 分割
    DEFAULT_MAX_SPLIT_LENGTH = 20         # 默认最大分割长度
    
    # 并发
    WARMUP_SIZE = 5                       # 预热大小
    
class FileConstants:
    """文件处理相关常量"""
    # 文件名长度
    MAX_FILENAME_LENGTH = 200             # 最大文件名长度
    
    # 用户输入限制
    MAX_USER_INPUT_LENGTH = 10000         # 最大用户输入长度
    
class LanguageConstants:
    """语言处理相关常量"""
    # 时长估算（每个音节的时长，秒）
    DURATION_PER_SYLLABLE = {
        'en': 0.225,                      # 英语
        'zh': 0.21,                       # 中文
        'default': 0.22                   # 默认值
    }
    
    # 字符权重（用于长度计算）
    CJK_CHAR_WEIGHT = 1.75                # 中日韩字符权重
    KOREAN_CHAR_WEIGHT = 1.5              # 韩语字符权重
    DEFAULT_CHAR_WEIGHT = 1.0             # 默认字符权重

# ------------
# 向后兼容的常量导出
# ------------

# 音频处理常量
MIN_AUDIO_DURATION = AudioConstants.MIN_AUDIO_DURATION
TIME_TOLERANCE = AudioConstants.TIME_TOLERANCE
DURATION_MULTIPLIER = AudioConstants.DURATION_MULTIPLIER
TIME_DIFF_TOLERANCE = AudioConstants.TIME_DIFF_TOLERANCE
SPEED_FACTOR_PRECISION = AudioConstants.SPEED_FACTOR_PRECISION

# 重试常量
DEFAULT_MAX_RETRIES = RetryConstants.DEFAULT_MAX_RETRIES
GPT_MAX_RETRIES = RetryConstants.GPT_MAX_RETRIES
DEFAULT_DELAY = RetryConstants.DEFAULT_DELAY

# 字幕常量
MAX_MERGE_COUNT = SubtitleConstants.MAX_MERGE_COUNT
MIN_MERGE_THRESHOLD = SubtitleConstants.MIN_MERGE_THRESHOLD

# 网络常量
MIN_FREE_SPACE_GB = NetworkConstants.MIN_FREE_SPACE_GB
MIN_DOWNLOAD_SPEED_MBPS = NetworkConstants.MIN_DOWNLOAD_SPEED_MBPS

# 相似度常量
MIN_SIMILARITY_THRESHOLD = SimilarityConstants.MIN_SIMILARITY_THRESHOLD