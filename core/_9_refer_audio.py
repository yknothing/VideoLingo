import os
from rich.panel import Panel
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from core.utils import *
from core.utils.models import *
import pandas as pd
import soundfile as sf
console = Console()

# Optional demucs import with fallback
try:
    from core.asr_backend.demucs_vl import demucs_audio
except ImportError:
    rprint("[yellow]Warning: demucs not available, some audio separation features will be disabled[/yellow]")
    demucs_audio = None

def time_to_samples(time_str, sr):
    """Unified time conversion function"""
    h, m, s = time_str.split(':')
    s, ms = s.split(',') if ',' in s else (s, '0')
    seconds = int(h) * 3600 + int(m) * 60 + float(s) + float(ms) / 1000
    return int(seconds * sr)

def extract_audio(audio_data, sr, start_time, end_time, out_file):
    """Simplified audio extraction function"""
    start = time_to_samples(start_time, sr)
    end = time_to_samples(end_time, sr)
    sf.write(out_file, audio_data[start:end], sr)

def extract_refer_audio_main():
    # Run demucs audio separation if available
    if demucs_audio is not None:
        demucs_audio() #!!! in case demucs not run
    else:
        rprint("[yellow]⚠️ Demucs is not available. Skipping audio separation step.[/yellow]")
    
    if os.path.exists(os.path.join(_AUDIO_SEGS_DIR, '1.wav')):
        rprint(Panel("Audio segments already exist, skipping extraction", title="Info", border_style="blue"))
        return

    # Create output directory
    os.makedirs(_AUDIO_REFERS_DIR, exist_ok=True)
    
    # Read task file and audio data
    df = pd.read_excel(_8_1_AUDIO_TASK)
    
    # Use vocal audio if available, otherwise fallback to raw audio
    audio_file = _VOCAL_AUDIO_FILE if os.path.exists(_VOCAL_AUDIO_FILE) else _RAW_AUDIO_FILE
    data, sr = sf.read(audio_file)
    
    if audio_file == _RAW_AUDIO_FILE:
        rprint("[yellow]ℹ️ Using raw audio file (demucs separation not available)[/yellow]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    ) as progress:
        task = progress.add_task("Extracting audio segments...", total=len(df))
        
        for _, row in df.iterrows():
            out_file = os.path.join(_AUDIO_REFERS_DIR, f"{row['number']}.wav")
            extract_audio(data, sr, row['start_time'], row['end_time'], out_file)
            progress.update(task, advance=1)
            
    rprint(Panel(f"Audio segments saved to {_AUDIO_REFERS_DIR}", title="Success", border_style="green"))

if __name__ == "__main__":
    extract_refer_audio_main()