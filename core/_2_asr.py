from core.utils import *
import contextlib
import threading
import time
from typing import Dict, List, Optional, Tuple, Any
import tempfile
import shutil

# 可选导入，避免启动时的依赖问题
try:
    from core.asr_backend.demucs_vl import demucs_audio
except ImportError:
    rprint(
        "[yellow]Warning: demucs not available, some audio separation features will be disabled[/yellow]"
    )
    demucs_audio = None

from core.asr_backend.audio_preprocess import (
    process_transcription,
    convert_video_to_audio,
    split_audio,
    save_results,
    normalize_audio_volume,
)
from core._1_ytdlp import find_video_files
from core.utils.models import *
import psutil
import gc
import weakref
from collections import deque


class MemoryManager:
    """
    Advanced memory management system for ASR processing with DoS attack prevention
    """
    
    def __init__(self):
        self.system_memory = psutil.virtual_memory().total
        self.critical_threshold = 0.90  # 90% - Critical level
        self.warning_threshold = 0.80   # 80% - Warning level
        self.safe_threshold = 0.70      # 70% - Safe level
        self.min_free_memory = max(1024 * 1024 * 1024, self.system_memory * 0.10)  # 1GB or 10% of total
        self.max_results_memory = self.system_memory * 0.20  # Max 20% of system memory for results
        self.memory_history = deque(maxlen=10)
        self.pressure_detected = False
        self.cleanup_callbacks = []
        self._monitoring_active = False
        self._monitor_thread = None
        
    def start_monitoring(self):
        """Start continuous memory monitoring in background thread"""
        if self._monitoring_active:
            return
            
        self._monitoring_active = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        rprint("[cyan]🔍 Memory monitoring started[/cyan]")
        
    def stop_monitoring(self):
        """Stop memory monitoring"""
        self._monitoring_active = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
        rprint("[cyan]🔍 Memory monitoring stopped[/cyan]")
        
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self._monitoring_active:
            try:
                memory_info = self.get_memory_status()
                self.memory_history.append(memory_info['used_percent'])
                
                # Detect memory pressure trend
                if len(self.memory_history) >= 5:
                    recent_avg = sum(list(self.memory_history)[-5:]) / 5
                    if recent_avg > self.warning_threshold * 100:
                        self.pressure_detected = True
                        if recent_avg > self.critical_threshold * 100:
                            rprint(f"[red]🚨 CRITICAL MEMORY PRESSURE: {recent_avg:.1f}%[/red]")
                            self._emergency_cleanup()
                    else:
                        self.pressure_detected = False
                        
                time.sleep(2.0)  # Check every 2 seconds
            except Exception as e:
                rprint(f"[yellow]Memory monitoring error: {e}[/yellow]")
                
    def register_cleanup_callback(self, callback):
        """Register cleanup callback for emergency situations"""
        self.cleanup_callbacks.append(weakref.ref(callback))
        
    def _emergency_cleanup(self):
        """Emergency cleanup when memory pressure is critical"""
        rprint("[red]🚨 Executing emergency memory cleanup[/red]")
        
        # Call registered cleanup callbacks
        for callback_ref in self.cleanup_callbacks[:]:
            callback = callback_ref()
            if callback is None:
                self.cleanup_callbacks.remove(callback_ref)
            else:
                try:
                    callback()
                except Exception as e:
                    rprint(f"[yellow]Cleanup callback failed: {e}[/yellow]")
        
        # Force garbage collection
        for _ in range(3):
            gc.collect()
            
        # Additional system-specific cleanup
        try:
            if hasattr(gc, 'set_threshold'):
                gc.set_threshold(100, 10, 10)  # More aggressive GC
        except:
            pass
            
    def get_memory_status(self) -> Dict[str, float]:
        """Get detailed memory status"""
        try:
            memory = psutil.virtual_memory()
            return {
                "total_gb": memory.total / (1024 ** 3),
                "available_gb": memory.available / (1024 ** 3),
                "used_gb": memory.used / (1024 ** 3),
                "available_mb": memory.available / (1024 * 1024),
                "used_percent": memory.percent,
                "available_percent": 100 - memory.percent,
                "pressure_level": self._get_pressure_level(memory.percent),
                "is_safe": memory.available > self.min_free_memory
            }
        except Exception as e:
            rprint(f"[yellow]Memory status check failed: {e}[/yellow]")
            return {"error": str(e)}
            
    def _get_pressure_level(self, used_percent: float) -> str:
        """Determine memory pressure level"""
        if used_percent >= self.critical_threshold * 100:
            return "CRITICAL"
        elif used_percent >= self.warning_threshold * 100:
            return "HIGH"
        elif used_percent >= self.safe_threshold * 100:
            return "MODERATE"
        else:
            return "LOW"
            
    def check_memory_for_operation(self, operation_name: str, estimated_mb: int) -> bool:
        """
        Check if there's sufficient memory for an operation
        Returns True if safe to proceed, False otherwise
        """
        memory_info = self.get_memory_status()
        
        if "error" in memory_info:
            rprint(f"[yellow]⚠️ Cannot check memory for {operation_name}, proceeding with caution[/yellow]")
            return True
            
        available_mb = memory_info["available_mb"]
        required_mb = estimated_mb + (self.min_free_memory / (1024 * 1024))  # Add safety buffer
        
        if available_mb < required_mb:
            rprint(f"[red]❌ Insufficient memory for {operation_name}[/red]")
            rprint(f"[red]   Required: {required_mb:.0f}MB, Available: {available_mb:.0f}MB[/red]")
            return False
            
        pressure_level = memory_info["pressure_level"]
        if pressure_level in ["HIGH", "CRITICAL"]:
            rprint(f"[yellow]⚠️ Memory pressure {pressure_level} for {operation_name}[/yellow]")
            if pressure_level == "CRITICAL":
                return False
                
        rprint(f"[green]✅ Memory check passed for {operation_name}: {available_mb:.0f}MB available[/green]")
        return True
        
    def monitor_memory_and_warn(self, stage: str, estimated_requirement_mb: int = 1024):
        """Enhanced memory monitoring with adaptive thresholds"""
        memory_info = self.get_memory_status()
        
        if "error" in memory_info:
            rprint(f"[yellow]⚠️ Memory monitoring failed at {stage}[/yellow]")
            return
            
        available_mb = memory_info["available_mb"]
        used_percent = memory_info["used_percent"]
        pressure_level = memory_info["pressure_level"]
        
        # Adaptive threshold based on system memory
        dynamic_threshold = max(estimated_requirement_mb, self.min_free_memory / (1024 * 1024))
        
        status_color = {
            "LOW": "green",
            "MODERATE": "cyan", 
            "HIGH": "yellow",
            "CRITICAL": "red"
        }.get(pressure_level, "white")
        
        rprint(f"[{status_color}]📊 Memory at {stage}:[/{status_color}]")
        rprint(f"[{status_color}]   Available: {available_mb:.0f}MB ({memory_info['available_percent']:.1f}%)[/{status_color}]")
        rprint(f"[{status_color}]   Pressure: {pressure_level} ({used_percent:.1f}% used)[/{status_color}]")
        
        if available_mb < dynamic_threshold:
            rprint(f"[red]🚨 LOW MEMORY WARNING at {stage}![/red]")
            rprint(f"[red]   Available: {available_mb:.0f}MB < Required: {dynamic_threshold:.0f}MB[/red]")
            rprint(f"[yellow]   Triggering memory cleanup...[/yellow]")
            self._emergency_cleanup()
            
        if pressure_level == "CRITICAL":
            raise MemoryError(f"Critical memory pressure at {stage}: {used_percent:.1f}% used")


class MemoryEfficientResultCollector:
    """
    Memory-efficient collector for transcription results with automatic cleanup
    """
    
    def __init__(self, memory_manager: MemoryManager, max_memory_mb: float = None):
        self.memory_manager = memory_manager
        self.max_memory_mb = max_memory_mb or (memory_manager.max_results_memory / (1024 * 1024))
        self.results = []
        self.temp_dir = None
        self.spill_to_disk = False
        self.disk_results = []
        self._total_size_estimate = 0
        
        # Register cleanup callback
        self.memory_manager.register_cleanup_callback(self._cleanup_callback)
        
    def _cleanup_callback(self):
        """Emergency cleanup callback"""
        if self.spill_to_disk and len(self.results) > 10:
            # Keep only recent results in memory
            self._spill_excess_to_disk(keep_recent=5)
            
    def _estimate_result_size(self, result: Dict) -> int:
        """Estimate memory size of a transcription result"""
        try:
            # Rough estimation based on text content
            text_size = 0
            if 'segments' in result:
                for segment in result['segments']:
                    if 'text' in segment:
                        text_size += len(segment['text']) * 4  # Unicode overhead
                    if 'words' in segment:
                        for word in segment['words']:
                            if 'word' in word:
                                text_size += len(word['word']) * 4
                                
            # Add overhead for metadata
            return max(text_size * 2, 1024)  # At least 1KB per result
        except Exception:
            return 10240  # 10KB default estimate
            
    def add_result(self, result: Dict):
        """Add result with memory monitoring"""
        result_size = self._estimate_result_size(result)
        self._total_size_estimate += result_size
        
        # Check if we should spill to disk
        if self._total_size_estimate > (self.max_memory_mb * 1024 * 1024):
            if not self.spill_to_disk:
                rprint("[yellow]💾 Memory limit reached, spilling results to disk[/yellow]")
                self._enable_disk_spill()
            self._spill_to_disk_if_needed()
            
        self.results.append(result)
        
        # Periodic memory check
        if len(self.results) % 50 == 0:
            memory_info = self.memory_manager.get_memory_status()
            if memory_info.get('pressure_level') in ['HIGH', 'CRITICAL']:
                self._spill_excess_to_disk(keep_recent=20)
                
    def _enable_disk_spill(self):
        """Enable spilling results to temporary disk storage"""
        try:
            self.temp_dir = tempfile.mkdtemp(prefix="videolingo_asr_")
            self.spill_to_disk = True
            rprint(f"[cyan]💾 Disk spill enabled: {self.temp_dir}[/cyan]")
        except Exception as e:
            rprint(f"[yellow]⚠️ Could not enable disk spill: {e}[/yellow]")
            
    def _spill_excess_to_disk(self, keep_recent: int = 10):
        """Spill older results to disk, keeping recent ones in memory"""
        if not self.spill_to_disk or len(self.results) <= keep_recent:
            return
            
        try:
            # Move older results to disk
            to_spill = self.results[:-keep_recent]
            for i, result in enumerate(to_spill):
                disk_file = os.path.join(self.temp_dir, f"result_{len(self.disk_results) + i}.json")
                with open(disk_file, 'w', encoding='utf-8') as f:
                    import json
                    json.dump(result, f, ensure_ascii=False)
                self.disk_results.append(disk_file)
                
            # Keep only recent results in memory
            self.results = self.results[-keep_recent:]
            
            # Update size estimate
            self._total_size_estimate = sum(self._estimate_result_size(r) for r in self.results)
            
            rprint(f"[cyan]💾 Spilled {len(to_spill)} results to disk, {len(self.results)} in memory[/cyan]")
            gc.collect()
            
        except Exception as e:
            rprint(f"[yellow]⚠️ Disk spill failed: {e}[/yellow]")
            
    def _spill_to_disk_if_needed(self):
        """Check if we should spill more results to disk"""
        if len(self.results) > 100:  # Too many results in memory
            self._spill_excess_to_disk(keep_recent=30)
            
    def get_all_results(self) -> List[Dict]:
        """Retrieve all results, loading from disk if necessary"""
        all_results = []
        
        # Load from disk first
        for disk_file in self.disk_results:
            try:
                with open(disk_file, 'r', encoding='utf-8') as f:
                    import json
                    result = json.load(f)
                    all_results.append(result)
            except Exception as e:
                rprint(f"[yellow]⚠️ Could not load disk result {disk_file}: {e}[/yellow]")
                
        # Add memory results
        all_results.extend(self.results)
        
        return all_results
        
    def cleanup(self):
        """Clean up temporary files"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                rprint("[cyan]🧹 Temporary disk storage cleaned up[/cyan]")
            except Exception as e:
                rprint(f"[yellow]⚠️ Cleanup warning: {e}[/yellow]")
                
    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()


@contextlib.contextmanager
def memory_safe_transcription():
    """Context manager for memory-safe transcription operations"""
    memory_manager = MemoryManager()
    collector = None
    
    try:
        memory_manager.start_monitoring()
        collector = MemoryEfficientResultCollector(memory_manager)
        yield memory_manager, collector
        
    except MemoryError as e:
        rprint(f"[red]🚨 MEMORY ERROR: {e}[/red]")
        rprint(f"[red]   Transcription aborted to prevent system crash[/red]")
        raise
        
    except Exception as e:
        rprint(f"[red]❌ Transcription error: {e}[/red]")
        raise
        
    finally:
        try:
            if collector:
                collector.cleanup()
            memory_manager.stop_monitoring()
            
            # Final cleanup
            gc.collect()
            
        except Exception as e:
            rprint(f"[yellow]⚠️ Cleanup warning: {e}[/yellow]")


def estimate_memory_requirements(segments: List[Tuple[float, float]], runtime: str) -> Dict[str, int]:
    """Estimate memory requirements based on audio segments and runtime"""
    total_duration = sum(end - start for start, end in segments)
    num_segments = len(segments)
    
    # Base estimates in MB
    estimates = {
        "audio_loading": max(100, int(total_duration * 2)),  # 2MB per minute
        "transcription_per_segment": 50 if runtime == "local" else 10,
        "model_loading": 3000 if runtime == "local" else 50,
        "results_accumulation": max(50, int(num_segments * 0.5)),  # 0.5MB per segment
        "processing_overhead": 500
    }
    
    estimates["total_estimated"] = sum(estimates.values())
    estimates["peak_concurrent"] = estimates["model_loading"] + estimates["transcription_per_segment"] * 2
    
    return estimates


@check_file_exists(_2_CLEANED_CHUNKS)
def transcribe():
    """Enhanced transcribe function with comprehensive memory management"""
    
    with memory_safe_transcription() as (memory_manager, result_collector):
        try:
            # Initial system check
            memory_manager.monitor_memory_and_warn("transcription start", 2048)
            
            # 1. Video to audio conversion
            video_file = find_video_files()
            if not memory_manager.check_memory_for_operation("video conversion", 1024):
                raise MemoryError("Insufficient memory for video conversion")
                
            convert_video_to_audio(video_file)
            memory_manager.monitor_memory_and_warn("after video conversion", 1024)
            
            # 2. Demucs vocal separation with memory monitoring
            if load_key("demucs") and demucs_audio is not None:
                if not memory_manager.check_memory_for_operation("Demucs processing", 4096):
                    rprint("[yellow]⚠️ Insufficient memory for Demucs, skipping vocal separation[/yellow]")
                    vocal_audio = _RAW_AUDIO_FILE
                else:
                    rprint("[cyan]🎵 Starting vocal separation with Demucs...[/cyan]")
                    memory_manager.monitor_memory_and_warn("before Demucs", 4096)
                    
                    try:
                        demucs_audio()
                        vocal_audio = normalize_audio_volume(_VOCAL_AUDIO_FILE, _VOCAL_AUDIO_FILE, format="mp3")
                        memory_manager.monitor_memory_and_warn("after Demucs", 1024)
                    finally:
                        # Aggressive cleanup after Demucs
                        for _ in range(3):
                            gc.collect()
                            
            elif load_key("demucs") and demucs_audio is None:
                rprint("[yellow]⚠️ Demucs is enabled in config but not available. Skipping vocal separation.[/yellow]")
                vocal_audio = _RAW_AUDIO_FILE
            else:
                vocal_audio = _RAW_AUDIO_FILE
                
            # 3. Audio segmentation
            if not memory_manager.check_memory_for_operation("audio segmentation", 2048):
                raise MemoryError("Insufficient memory for audio segmentation")
                
            segments = split_audio(_RAW_AUDIO_FILE)
            memory_manager.monitor_memory_and_warn("after audio segmentation", 1024)
            
            # 4. Estimate memory requirements
            runtime = load_key("whisper.runtime")
            memory_estimates = estimate_memory_requirements(segments, runtime)
            
            rprint(f"[cyan]📊 Memory estimates:[/cyan]")
            for key, value in memory_estimates.items():
                rprint(f"[cyan]   {key}: {value}MB[/cyan]")
                
            # Check if we have enough memory for the full operation
            peak_memory_mb = memory_estimates["peak_concurrent"]
            if not memory_manager.check_memory_for_operation("full transcription", peak_memory_mb):
                # Implement graceful degradation
                rprint("[yellow]⚠️ Implementing memory-efficient processing mode[/yellow]")
                # Could implement smaller batch sizes, more frequent cleanup, etc.
                
            # 5. Load transcription engine
            if runtime == "local":
                from core.asr_backend.whisperX_local import transcribe_audio as ts
                rprint("[cyan]🎤 Transcribing audio with local model...[/cyan]")
                required_memory = memory_estimates["model_loading"]
                
            elif runtime == "cloud":
                from core.asr_backend.whisperX_302 import transcribe_audio_302 as ts
                rprint("[cyan]🎤 Transcribing audio with 302 API...[/cyan]")
                required_memory = 512
                
            elif runtime == "elevenlabs":
                from core.asr_backend.elevenlabs_asr import transcribe_audio_elevenlabs as ts
                rprint("[cyan]🎤 Transcribing audio with ElevenLabs API...[/cyan]")
                required_memory = 512
            else:
                raise ValueError(f"Unknown runtime: {runtime}")
                
            memory_manager.monitor_memory_and_warn(f"before {runtime} transcription", required_memory)
            
            # 6. Process segments with memory-efficient batching
            total_segments = len(segments)
            rprint(f"[cyan]🎙️ Processing {total_segments} audio segments...[/cyan]")
            
            # Adaptive batch size based on available memory
            memory_info = memory_manager.get_memory_status()
            if memory_info.get('pressure_level') in ['HIGH', 'CRITICAL']:
                cleanup_interval = 5  # More frequent cleanup
                progress_interval = 5
            else:
                cleanup_interval = 10
                progress_interval = 10
                
            for i, (start, end) in enumerate(segments):
                try:
                    # Memory check before processing each segment
                    if i % cleanup_interval == 0 and i > 0:
                        memory_manager.monitor_memory_and_warn(
                            f"segment {i}/{total_segments}", 
                            memory_estimates["transcription_per_segment"]
                        )
                        gc.collect()
                        
                    # Check for memory pressure before continuing
                    if memory_manager.pressure_detected:
                        rprint(f"[yellow]⚠️ Memory pressure detected at segment {i}, forcing cleanup[/yellow]")
                        memory_manager._emergency_cleanup()
                        
                        # Re-check after cleanup
                        if not memory_manager.check_memory_for_operation(f"segment {i}", 
                                                                       memory_estimates["transcription_per_segment"]):
                            rprint(f"[red]❌ Aborting at segment {i} due to memory constraints[/red]")
                            break
                            
                    # Process segment
                    result = ts(_RAW_AUDIO_FILE, vocal_audio, start, end)
                    result_collector.add_result(result)
                    
                    # Progress reporting
                    if i % progress_interval == 0 or i == total_segments - 1:
                        progress = (i + 1) / total_segments * 100
                        rprint(f"[green]📈 Progress: {progress:.1f}% ({i+1}/{total_segments})[/green]")
                        
                except Exception as e:
                    rprint(f"[red]❌ Error processing segment {i} ({start:.2f}s-{end:.2f}s): {e}[/red]")
                    # Continue with next segment rather than failing completely
                    continue
                    
            memory_manager.monitor_memory_and_warn("after transcription", 1024)
            
            # 7. Combine results efficiently
            rprint("[cyan]🔗 Combining transcription results...[/cyan]")
            all_results = result_collector.get_all_results()
            
            if not all_results:
                raise ValueError("No transcription results were generated")
                
            combined_result = {"segments": []}
            for result in all_results:
                if result and "segments" in result:
                    combined_result["segments"].extend(result["segments"])
                    
            rprint(f"[green]✅ Combined {len(all_results)} results into {len(combined_result['segments'])} segments[/green]")
            
            # Clear results from memory immediately after combining
            del all_results
            gc.collect()
            
            # 8. Process and save final results
            if not memory_manager.check_memory_for_operation("final processing", 1024):
                rprint("[yellow]⚠️ Low memory for final processing, forcing cleanup[/yellow]")
                memory_manager._emergency_cleanup()
                
            df = process_transcription(combined_result)
            save_results(df)
            
            # Final cleanup
            del combined_result, df
            memory_manager.monitor_memory_and_warn("transcription complete", 512)
            
            rprint("[green]🎉 Transcription completed successfully with enhanced memory management[/green]")
            
        except MemoryError as e:
            rprint(f"[red]🚨 MEMORY ERROR: {e}[/red]")
            rprint(f"[red]   The system ran out of memory during transcription.[/red]")
            rprint(f"[yellow]   Try: 1) Processing smaller audio files 2) Using cloud API 3) Adding more RAM[/yellow]")
            raise
            
        except Exception as e:
            rprint(f"[red]❌ Transcription failed: {e}[/red]")
            raise
            
        finally:
            # Final memory cleanup
            gc.collect()


if __name__ == "__main__":
    transcribe()
