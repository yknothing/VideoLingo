# ------------
# Time Utilities - 统一时间处理工具
# 提取重复的时间解析和格式化代码
# ------------

import datetime
from typing import Union


class TimeUtils:
    """统一的时间处理工具类"""
    
    @staticmethod
    def srt_to_seconds(time_str: str) -> float:
        """
        将SRT时间格式转换为秒数
        格式: HH:MM:SS.mmm 或 HH:MM:SS,mmm
        """
        time_str = time_str.strip()
        
        # 处理不同的分隔符 (. 或 ,)
        if ',' in time_str:
            time_part, ms_part = time_str.rsplit(',', 1)
        else:
            time_part, ms_part = time_str.rsplit('.', 1)
            
        hours, minutes, seconds = time_part.split(':')
        
        total_seconds = (
            int(hours) * 3600 + 
            int(minutes) * 60 + 
            int(seconds) + 
            int(ms_part) / 1000
        )
        
        return total_seconds
    
    @staticmethod
    def seconds_to_srt(seconds: float) -> str:
        """
        将秒数转换为SRT时间格式
        格式: HH:MM:SS,mmm
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"
    
    @staticmethod
    def time_diff_seconds(time1_str: str, time2_str: str) -> float:
        """
        计算两个时间字符串之间的差值（秒）
        """
        time1 = TimeUtils.srt_to_seconds(time1_str)
        time2 = TimeUtils.srt_to_seconds(time2_str)
        return abs(time2 - time1)
    
    @staticmethod
    def format_duration(seconds: float) -> str:
        """
        格式化时长为可读格式
        """
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"
    
    @staticmethod
    def time_to_samples(time_str: str, sample_rate: int = 16000) -> int:
        """
        将时间字符串转换为音频采样点数
        """
        seconds = TimeUtils.srt_to_seconds(time_str)
        return int(seconds * sample_rate)
    
    @staticmethod
    def samples_to_time(samples: int, sample_rate: int = 16000) -> str:
        """
        将音频采样点数转换为时间字符串
        """
        seconds = samples / sample_rate
        return TimeUtils.seconds_to_srt(seconds)


# ------------
# 向后兼容的函数导出
# ------------

def parse_df_srt_time(time_str: str) -> float:
    """向后兼容：解析SRT时间格式"""
    return TimeUtils.srt_to_seconds(time_str)

def format_time(seconds: float) -> str:
    """向后兼容：格式化时间"""
    return TimeUtils.seconds_to_srt(seconds)

def time_diff_seconds(dt1: Union[str, datetime.datetime], dt2: Union[str, datetime.datetime], 
                     today: datetime.date = None) -> float:
    """向后兼容：计算时间差"""
    if isinstance(dt1, str) and isinstance(dt2, str):
        return TimeUtils.time_diff_seconds(dt1, dt2)
    
    # 处理datetime对象
    if isinstance(dt1, datetime.datetime) and isinstance(dt2, datetime.datetime):
        return abs((dt2 - dt1).total_seconds())
    
    raise ValueError("Unsupported time format")