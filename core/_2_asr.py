from core.utils import *
# 可选导入，避免启动时的依赖问题
try:
    from core.asr_backend.demucs_vl import demucs_audio
except ImportError:
    rprint("[yellow]Warning: demucs not available, some audio separation features will be disabled[/yellow]")
    demucs_audio = None

from core.asr_backend.audio_preprocess import process_transcription, convert_video_to_audio, split_audio, save_results, normalize_audio_volume
from core._1_ytdlp import find_video_files
from core.utils.models import *
import psutil
import gc

def check_memory_usage():
    """检查当前内存使用情况"""
    try:
        memory = psutil.virtual_memory()
        return {
            'available_mb': memory.available / (1024 * 1024),
            'used_percent': memory.percent,
            'available_percent': 100 - memory.percent
        }
    except Exception as e:
        rprint(f"[yellow]Memory check failed: {e}[/yellow]")
        return {'available_mb': -1, 'used_percent': -1, 'available_percent': -1}

def monitor_memory_and_warn(stage: str, min_required_mb: int = 2048):
    """
    监控内存使用并发出警告
    Args:
        stage: 处理阶段名称（用于日志）
        min_required_mb: 建议的最小可用内存(MB)
    """
    memory_info = check_memory_usage()
    
    if memory_info['available_mb'] > 0:
        if memory_info['available_mb'] < min_required_mb:
            rprint(f"[red]Warning: Low memory at {stage}. Available: {memory_info['available_mb']:.0f}MB, Recommended: {min_required_mb}MB[/red]")
            rprint(f"[yellow]Consider closing other applications or processing smaller audio segments[/yellow]")
        elif memory_info['used_percent'] > 85:
            rprint(f"[yellow]High memory usage at {stage}: {memory_info['used_percent']:.1f}% used[/yellow]")
        else:
            rprint(f"[green]Memory status at {stage}: {memory_info['available_mb']:.0f}MB available ({memory_info['available_percent']:.1f}% free)[/green]")

@check_file_exists(_2_CLEANED_CHUNKS)
def transcribe():
    # 初始内存检查
    monitor_memory_and_warn("transcription start", 2048)
    
    # 1. video to audio
    video_file = find_video_files()
    convert_video_to_audio(video_file)
    monitor_memory_and_warn("after video conversion", 1024)

    # 2. Demucs vocal separation:
    if load_key("demucs") and demucs_audio is not None:
        rprint("[cyan]🎵 Starting vocal separation with Demucs...[/cyan]")
        monitor_memory_and_warn("before Demucs", 4096)  # Demucs需要更多内存
        demucs_audio()
        vocal_audio = normalize_audio_volume(_VOCAL_AUDIO_FILE, _VOCAL_AUDIO_FILE, format="mp3")
        monitor_memory_and_warn("after Demucs", 1024)
        # 强制垃圾回收释放Demucs占用的内存
        gc.collect()
    elif load_key("demucs") and demucs_audio is None:
        rprint("[yellow]⚠️  Demucs is enabled in config but not available. Skipping vocal separation.[/yellow]")
        vocal_audio = output_audio
    else:
        vocal_audio = _RAW_AUDIO_FILE

    # 3. Extract audio
    segments = split_audio(_RAW_AUDIO_FILE)
    monitor_memory_and_warn("after audio segmentation", 1024)
    
    # 4. Transcribe audio by clips
    all_results = []
    runtime = load_key("whisper.runtime")
    if runtime == "local":
        from core.asr_backend.whisperX_local import transcribe_audio as ts
        rprint("[cyan]🎤 Transcribing audio with local model...[/cyan]")
        monitor_memory_and_warn("before local transcription", 3072)  # 本地模型需要更多内存
    elif runtime == "cloud":
        from core.asr_backend.whisperX_302 import transcribe_audio_302 as ts
        rprint("[cyan]🎤 Transcribing audio with 302 API...[/cyan]")
        monitor_memory_and_warn("before cloud transcription", 512)  # 云端API内存需求较低
    elif runtime == "elevenlabs":
        from core.asr_backend.elevenlabs_asr import transcribe_audio_elevenlabs as ts
        rprint("[cyan]🎤 Transcribing audio with ElevenLabs API...[/cyan]")
        monitor_memory_and_warn("before ElevenLabs transcription", 512)

    # 分段处理并监控内存
    for i, (start, end) in enumerate(segments):
        if i % 10 == 0 and i > 0:  # 每10个片段检查一次内存
            monitor_memory_and_warn(f"transcription segment {i}/{len(segments)}", 1024)
            gc.collect()  # 定期清理内存
        
        result = ts(_RAW_AUDIO_FILE, vocal_audio, start, end)
        all_results.append(result)
    
    monitor_memory_and_warn("after transcription", 1024)
    
    # 5. Combine results
    combined_result = {'segments': []}
    for result in all_results:
        combined_result['segments'].extend(result['segments'])
    
    # 6. Process df
    df = process_transcription(combined_result)
    save_results(df)
    
    # 最终内存状态
    monitor_memory_and_warn("transcription complete", 512)
    gc.collect()  # 最终清理
        
if __name__ == "__main__":
    transcribe()