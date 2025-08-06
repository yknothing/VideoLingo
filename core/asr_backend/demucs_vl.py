import os
import torch
from rich.console import Console
from rich import print as rprint
from torch.cuda import is_available as is_cuda_available
from typing import Optional
import gc
from core.utils.models import *

# Optional demucs imports with fallbacks
try:
    from demucs.pretrained import get_model
    from demucs.audio import save_audio
    from demucs.api import Separator
    from demucs.apply import BagOfModels
    DEMUCS_AVAILABLE = True
except ImportError:
    rprint("[yellow]Warning: demucs not available, audio separation features will be disabled[/yellow]")
    DEMUCS_AVAILABLE = False
    get_model = None
    save_audio = None
    Separator = None
    BagOfModels = None

def create_preloaded_separator(model, shifts: int = 1, overlap: float = 0.25,
                              split: bool = True, segment: Optional[int] = None, jobs: int = 0):
    """Create a preloaded separator if demucs is available"""
    if not DEMUCS_AVAILABLE or Separator is None:
        raise ImportError("demucs is not available")
    
    separator = Separator(model)
    device = "cuda" if is_cuda_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    separator.update_parameter(device=device, shifts=shifts, overlap=overlap, split=split,
                             segment=segment, jobs=jobs, progress=True, callback=None, callback_arg=None)
    return separator

def is_demucs_available():
    """Check if demucs is available"""
    return DEMUCS_AVAILABLE

def demucs_audio():
    if not DEMUCS_AVAILABLE:
        rprint("[yellow]⚠️ Demucs is not available. Audio separation will be skipped.[/yellow]")
        rprint("[cyan]💡 To enable audio separation, install demucs: pip install demucs[/cyan]")
        return
    
    if os.path.exists(_VOCAL_AUDIO_FILE) and os.path.exists(_BACKGROUND_AUDIO_FILE):
        rprint(f"[yellow]⚠️ {_VOCAL_AUDIO_FILE} and {_BACKGROUND_AUDIO_FILE} already exist, skip Demucs processing.[/yellow]")
        return
    
    console = Console()
    os.makedirs(_AUDIO_DIR, exist_ok=True)
    
    try:
        console.print("🤖 Loading <htdemucs> model...")
        model = get_model('htdemucs')
        separator = create_preloaded_separator(model, shifts=1, overlap=0.25)
        
        console.print("🎵 Separating audio...")
        _, outputs = separator.separate_audio_file(_RAW_AUDIO_FILE)
        
        kwargs = {"samplerate": model.samplerate, "bitrate": 128, "preset": 2, 
                 "clip": "rescale", "as_float": False, "bits_per_sample": 16}
        
        console.print("🎤 Saving vocals track...")
        save_audio(outputs['vocals'].cpu(), _VOCAL_AUDIO_FILE, **kwargs)
        
        console.print("🎹 Saving background music...")
        background = sum(audio for source, audio in outputs.items() if source != 'vocals')
        save_audio(background.cpu(), _BACKGROUND_AUDIO_FILE, **kwargs)
        
        # Clean up memory
        del outputs, background, model, separator
        gc.collect()
        
        console.print("[green]✨ Audio separation completed![/green]")
    except Exception as e:
        rprint(f"[red]❌ Error during audio separation: {e}[/red]")
        rprint("[yellow]Continuing without audio separation...[/yellow]")

if __name__ == "__main__":
    demucs_audio()
