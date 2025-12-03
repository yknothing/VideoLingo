import os, subprocess
import pandas as pd
from typing import Dict, List, Tuple
from pydub import AudioSegment
from core.utils import *
from core.utils.models import *
from pydub import AudioSegment
from pydub.silence import detect_silence
from pydub.utils import mediainfo
from rich import print as rprint
import psutil
import gc
import contextlib
import tempfile
import weakref


class AudioMemoryManager:
    """Memory manager specifically for audio processing operations"""
    
    def __init__(self):
        self.active_audio_objects = set()
        self.memory_threshold_mb = 2048  # 2GB threshold for audio operations
        
    def register_audio_object(self, audio_obj):
        """Register an audio object for tracking"""
        # Use weak references to avoid circular references
        self.active_audio_objects.add(weakref.ref(audio_obj, self._audio_cleanup_callback))
        
    def _audio_cleanup_callback(self, weak_ref):
        """Callback when audio object is garbage collected"""
        self.active_audio_objects.discard(weak_ref)
        
    def check_audio_memory(self, operation: str) -> bool:
        """Check if we have sufficient memory for audio operations"""
        try:
            memory = psutil.virtual_memory()
            available_mb = memory.available / (1024 * 1024)
            
            if available_mb < self.memory_threshold_mb:
                rprint(f"[yellow]⚠️ Low memory for {operation}: {available_mb:.0f}MB available[/yellow]")
                return False
                
            return True
        except Exception as e:
            rprint(f"[yellow]Memory check failed for {operation}: {e}[/yellow]")
            return True  # Assume it's safe if we can't check
            
    def cleanup_audio_objects(self):
        """Force cleanup of tracked audio objects"""
        # Clear dead references
        self.active_audio_objects = {ref for ref in self.active_audio_objects if ref() is not None}
        
        # Force garbage collection
        gc.collect()
        
        rprint(f"[cyan]🧹 Audio memory cleanup: {len(self.active_audio_objects)} objects tracked[/cyan]")


# Global audio memory manager
_audio_memory_manager = AudioMemoryManager()


@contextlib.contextmanager
def memory_safe_audio_processing(operation_name: str):
    """Context manager for memory-safe audio processing"""
    if not _audio_memory_manager.check_audio_memory(operation_name):
        rprint(f"[yellow]⚠️ Proceeding with {operation_name} despite low memory[/yellow]")
        
    try:
        yield _audio_memory_manager
    finally:
        # Cleanup after audio processing
        _audio_memory_manager.cleanup_audio_objects()


def normalize_audio_volume(audio_path, output_path, target_db=-20.0, format="wav"):
    """Memory-efficient audio normalization with automatic cleanup"""
    with memory_safe_audio_processing("audio normalization"):
        try:
            audio = AudioSegment.from_file(audio_path)
            _audio_memory_manager.register_audio_object(audio)
            
            change_in_dBFS = target_db - audio.dBFS
            normalized_audio = audio.apply_gain(change_in_dBFS)
            _audio_memory_manager.register_audio_object(normalized_audio)
            
            normalized_audio.export(output_path, format=format)
            rprint(f"[green]✅ Audio normalized from {audio.dBFS:.1f}dB to {target_db:.1f}dB[/green]")
            
            # Explicit cleanup
            del audio, normalized_audio
            gc.collect()
            
            return output_path
            
        except Exception as e:
            rprint(f"[red]❌ Audio normalization failed: {e}[/red]")
            raise


def convert_video_to_audio(video_file: str):
    """Memory-efficient video to audio conversion using FFmpeg"""
    with memory_safe_audio_processing("video to audio conversion"):
        os.makedirs(_AUDIO_DIR, exist_ok=True)
        if not os.path.exists(_RAW_AUDIO_FILE):
            rprint(f"[blue]🎬➡️🎵 Converting to high quality audio with FFmpeg ......[/blue]")
            
            # Use FFmpeg directly to avoid loading large files into memory
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    video_file,
                    "-vn",
                    "-c:a",
                    "libmp3lame",
                    "-b:a",
                    "32k",
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    "-metadata",
                    "encoding=UTF-8",
                    _RAW_AUDIO_FILE,
                ],
                check=True,
                stderr=subprocess.PIPE,
            )
            rprint(f"[green]🎬➡️🎵 Converted <{video_file}> to <{_RAW_AUDIO_FILE}> with FFmpeg\n[/green]")


def get_audio_duration(audio_file: str) -> float:
    """Get the duration of an audio file using ffmpeg (memory efficient)."""
    cmd = ["ffmpeg", "-i", audio_file]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    _, stderr = process.communicate()
    output = stderr.decode("utf-8", errors="ignore")

    try:
        duration_str = [line for line in output.split("\n") if "Duration" in line][0]
        duration_parts = duration_str.split("Duration: ")[1].split(",")[0].split(":")
        duration = (
            float(duration_parts[0]) * 3600
            + float(duration_parts[1]) * 60
            + float(duration_parts[2])
        )
    except Exception as e:
        rprint(f"[red]❌ Error: Failed to get audio duration: {e}[/red]")
        duration = 0
    return duration


def split_audio(
    audio_file: str, target_len: float = 30 * 60, win: float = 60
) -> List[Tuple[float, float]]:
    """Memory-efficient audio splitting with proper cleanup"""
    
    rprint(f"[blue]🎙️ Starting audio segmentation {audio_file} {target_len} {win}[/blue]")
    
    with memory_safe_audio_processing("audio segmentation"):
        # Get duration without loading full audio into memory
        duration = float(mediainfo(audio_file)["duration"])
        
        if duration <= target_len + win:
            return [(0, duration)]
            
        # Memory check before loading audio
        if not _audio_memory_manager.check_audio_memory("audio loading"):
            rprint("[yellow]⚠️ Low memory for audio loading, using conservative processing[/yellow]")
            
        # Load audio with explicit memory management
        try:
            audio = AudioSegment.from_file(audio_file)
            _audio_memory_manager.register_audio_object(audio)
            
            segments, pos = [], 0.0
            safe_margin = 0.5  # 静默点前后安全边界，单位秒
            
            while pos < duration:
                # Memory check during processing
                if not _audio_memory_manager.check_audio_memory(f"segment processing at {pos:.1f}s"):
                    rprint("[yellow]⚠️ Low memory during segmentation, forcing cleanup[/yellow]")
                    _audio_memory_manager.cleanup_audio_objects()
                    
                if duration - pos <= target_len:
                    segments.append((pos, duration))
                    break

                threshold = pos + target_len
                ws, we = int((threshold - win) * 1000), int((threshold + win) * 1000)

                # Use memory-efficient slicing
                try:
                    audio_slice = audio[ws:we]
                    _audio_memory_manager.register_audio_object(audio_slice)
                    
                    # 获取完整的静默区域
                    silence_regions = detect_silence(
                        audio_slice, min_silence_len=int(safe_margin * 1000), silence_thresh=-30
                    )
                    
                    # Clean up slice immediately after use
                    del audio_slice
                    
                except MemoryError:
                    rprint(f"[yellow]⚠️ Memory error during silence detection at {pos:.1f}s, using threshold[/yellow]")
                    silence_regions = []
                    
                silence_regions = [
                    (s / 1000 + (threshold - win), e / 1000 + (threshold - win)) for s, e in silence_regions
                ]
                
                # 筛选长度足够（至少1秒）且位置适合的静默区域
                valid_regions = [
                    (start, end)
                    for start, end in silence_regions
                    if (end - start) >= (safe_margin * 2)
                    and threshold <= start + safe_margin <= threshold + win
                ]

                if valid_regions:
                    start, end = valid_regions[0]
                    split_at = start + safe_margin  # 在静默区域起始点后0.5秒处切分
                else:
                    rprint(
                        f"[yellow]⚠️ No valid silence regions found for {audio_file} at {threshold}s, using threshold[/yellow]"
                    )
                    split_at = threshold

                segments.append((pos, split_at))
                pos = split_at
                
                # Periodic memory cleanup during long processing
                if len(segments) % 10 == 0:
                    _audio_memory_manager.cleanup_audio_objects()

            # Final cleanup
            del audio
            _audio_memory_manager.cleanup_audio_objects()
            
            rprint(f"[green]🎙️ Audio split completed {len(segments)} segments[/green]")
            return segments
            
        except Exception as e:
            rprint(f"[red]❌ Audio segmentation failed: {e}[/red]")
            _audio_memory_manager.cleanup_audio_objects()
            raise


def process_transcription(result: Dict) -> pd.DataFrame:
    """Memory-efficient transcription processing with input validation"""
    
    if not result or "segments" not in result:
        raise ValueError("Invalid transcription result: missing segments")
        
    all_words = []
    max_words_per_batch = 1000  # Process in batches to manage memory
    
    rprint(f"[cyan]📝 Processing {len(result['segments'])} transcription segments...[/cyan]")
    
    try:
        for segment_idx, segment in enumerate(result["segments"]):
            # Memory check for large batches
            if len(all_words) > 0 and len(all_words) % max_words_per_batch == 0:
                memory = psutil.virtual_memory()
                if memory.percent > 85:
                    rprint(f"[yellow]⚠️ High memory usage during transcription processing: {memory.percent:.1f}%[/yellow]")
                    gc.collect()
                    
            # Get speaker_id, if not exists, set to None
            speaker_id = segment.get("speaker_id", None)
            
            if "words" not in segment:
                rprint(f"[yellow]⚠️ Segment {segment_idx} missing words, skipping[/yellow]")
                continue

            for word_idx, word in enumerate(segment["words"]):
                try:
                    # Input validation and sanitization
                    word_text = word.get("word", "").strip()
                    
                    # Check word length (security: prevent memory exhaustion)
                    if len(word_text) > 30:
                        rprint(
                            f"[yellow]⚠️ Warning: Word longer than 30 characters in segment {segment_idx}, truncating: {word_text[:30]}...[/yellow]"
                        )
                        word_text = word_text[:30]
                        
                    # Skip empty words
                    if not word_text:
                        continue

                    # ! For French, we need to convert guillemets to empty strings
                    word_text = word_text.replace("»", "").replace("«", "")

                    if "start" not in word and "end" not in word:
                        if all_words:
                            # Assign the end time of the previous word as the start and end time of the current word
                            word_dict = {
                                "text": word_text,
                                "start": all_words[-1]["end"],
                                "end": all_words[-1]["end"],
                                "speaker_id": speaker_id,
                            }
                            all_words.append(word_dict)
                        else:
                            # If it's the first word, look next for a timestamp then assign it to the current word
                            next_word = next(
                                (w for w in segment["words"] if "start" in w and "end" in w), None
                            )
                            if next_word:
                                word_dict = {
                                    "text": word_text,
                                    "start": next_word["start"],
                                    "end": next_word["end"],
                                    "speaker_id": speaker_id,
                                }
                                all_words.append(word_dict)
                            else:
                                rprint(f"[yellow]⚠️ No timestamp found for word in segment {segment_idx}, skipping[/yellow]")
                                continue
                    else:
                        # Normal case, with start and end times
                        word_dict = {
                            "text": word_text,
                            "start": word.get("start", all_words[-1]["end"] if all_words else 0),
                            "end": word["end"],
                            "speaker_id": speaker_id,
                        }
                        all_words.append(word_dict)
                        
                except Exception as e:
                    rprint(f"[yellow]⚠️ Error processing word {word_idx} in segment {segment_idx}: {e}[/yellow]")
                    continue
                    
        if not all_words:
            raise ValueError("No valid words found in transcription result")
            
        rprint(f"[green]✅ Processed {len(all_words)} words from transcription[/green]")
        
        # Convert to DataFrame with memory monitoring
        memory_before = psutil.virtual_memory().percent
        df = pd.DataFrame(all_words)
        memory_after = psutil.virtual_memory().percent
        
        if memory_after - memory_before > 5:
            rprint(f"[yellow]⚠️ DataFrame creation increased memory usage by {memory_after - memory_before:.1f}%[/yellow]")
            
        # Clear the word list to free memory
        del all_words
        gc.collect()
        
        return df
        
    except Exception as e:
        rprint(f"[red]❌ Transcription processing failed: {e}[/red]")
        # Emergency cleanup
        if 'all_words' in locals():
            del all_words
        gc.collect()
        raise


def save_results(df: pd.DataFrame):
    """Memory-efficient result saving with validation and cleanup"""
    from core.utils.config_utils import get_storage_paths

    if df is None or df.empty:
        raise ValueError("Cannot save empty DataFrame")
        
    rprint(f"[cyan]💾 Saving {len(df)} transcription results...[/cyan]")
    
    try:
        paths = get_storage_paths()
        log_dir = os.path.join(paths["temp"], "log")
        os.makedirs(log_dir, exist_ok=True)

        # Remove rows where 'text' is empty
        initial_rows = len(df)
        df = df[df["text"].str.len() > 0]
        removed_rows = initial_rows - len(df)
        if removed_rows > 0:
            rprint(f"[blue]ℹ️ Removed {removed_rows} row(s) with empty text.[/blue]")

        # Check for and remove words longer than 30 characters (security)
        long_words = df[df["text"].str.len() > 30]
        if not long_words.empty:
            rprint(
                f"[yellow]⚠️ Warning: Detected {len(long_words)} word(s) longer than 30 characters. These will be truncated.[/yellow]"
            )
            df["text"] = df["text"].str.slice(0, 30)

        # Add quotes to text for proper CSV formatting
        df["text"] = df["text"].apply(lambda x: f'"{x}"')
        
        # Save with memory monitoring
        memory_before = psutil.virtual_memory().percent
        df.to_excel(_2_CLEANED_CHUNKS, index=False)
        memory_after = psutil.virtual_memory().percent
        
        rprint(f"[green]📊 Excel file saved to {_2_CLEANED_CHUNKS}[/green]")
        
        if memory_after > memory_before:
            rprint(f"[cyan]📊 File save memory impact: +{memory_after - memory_before:.1f}%[/cyan]")
            
    except Exception as e:
        rprint(f"[red]❌ Failed to save results: {e}[/red]")
        raise
    finally:
        # Force cleanup after save
        gc.collect()


def save_language(language: str):
    """Save detected language with validation"""
    if not language or not isinstance(language, str):
        rprint("[yellow]⚠️ Invalid language detected, using default[/yellow]")
        language = "en"
        
    # Sanitize language code (security)
    language = language.lower().strip()[:10]  # Max 10 chars
    update_key("whisper.detected_language", language)
    rprint(f"[green]🌐 Language saved: {language}[/green]")
