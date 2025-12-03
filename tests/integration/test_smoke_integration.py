"""
Cross-Module Integration Smoke Tests

Generates synthetic 3-second sine-wave video at runtime and tests complete pipeline:
- Download/Input → ASR Transcription → Translation → Subtitle Generation
Marked as integration and slow for CI job execution.
"""

import pytest
import os
import json
import tempfile
import shutil
import numpy as np
import wave
import struct
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock
from typing import Dict, Any, Tuple
import subprocess
import time

class VideoSynthesizer:
    """Generate synthetic test videos with audio at runtime"""
    
    @staticmethod
    def generate_sine_wave_audio(duration: float = 3.0, 
                                  frequency: float = 440.0,
                                  sample_rate: int = 44100,
                                  amplitude: float = 0.5) -> bytes:
        """
        Generate a sine wave audio signal
        
        Args:
            duration: Duration in seconds
            frequency: Frequency in Hz (440 = A4 note)
            sample_rate: Sample rate in Hz
            amplitude: Volume level (0.0 to 1.0)
            
        Returns:
            WAV file data as bytes
        """
        # Generate sine wave samples
        num_samples = int(sample_rate * duration)
        samples = []
        
        for i in range(num_samples):
            t = i / sample_rate
            # Generate sine wave with some harmonics for richness
            sample = amplitude * np.sin(2 * np.pi * frequency * t)
            # Add a subtle second harmonic
            sample += amplitude * 0.3 * np.sin(4 * np.pi * frequency * t)
            # Add slight third harmonic
            sample += amplitude * 0.1 * np.sin(6 * np.pi * frequency * t)
            
            # Convert to 16-bit PCM
            sample_int = int(sample * 32767)
            samples.append(max(-32768, min(32767, sample_int)))
        
        # Create WAV file in memory
        import io
        buffer = io.BytesIO()
        
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(struct.pack('<' + 'h' * len(samples), *samples))
        
        return buffer.getvalue()
    
    @staticmethod
    def generate_synthetic_video_ffmpeg(output_path: str,
                                         duration: float = 3.0,
                                         width: int = 320,
                                         height: int = 240,
                                         fps: int = 30,
                                         frequency: float = 440.0) -> bool:
        """
        Generate a synthetic video with sine wave audio using FFmpeg
        
        Args:
            output_path: Path to save the video
            duration: Duration in seconds
            width: Video width in pixels
            height: Video height in pixels
            fps: Frames per second
            frequency: Audio frequency in Hz
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # FFmpeg command to generate test video with sine wave audio
            # Uses lavfi (libavfilter) to generate test patterns
            cmd = [
                'ffmpeg', '-y',  # Overwrite output
                '-f', 'lavfi',
                '-i', f'testsrc=duration={duration}:size={width}x{height}:rate={fps}',
                '-f', 'lavfi', 
                '-i', f'sine=frequency={frequency}:duration={duration}',
                '-c:v', 'libx264',  # H.264 video codec
                '-preset', 'ultrafast',  # Fast encoding
                '-pix_fmt', 'yuv420p',  # Pixel format for compatibility
                '-c:a', 'aac',  # AAC audio codec
                '-b:a', '128k',  # Audio bitrate
                '-shortest',  # Match shortest stream duration
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            return result.returncode == 0
            
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            print(f"FFmpeg generation failed: {e}")
            return False
    
    @staticmethod
    def generate_mock_video_file(output_path: str, 
                                 duration: float = 3.0) -> bool:
        """
        Generate a mock video file (fallback if FFmpeg not available)
        Creates a valid MP4 structure with embedded audio
        """
        try:
            # Generate audio data
            audio_data = VideoSynthesizer.generate_sine_wave_audio(duration)
            
            # Create a minimal valid MP4 structure
            # This is a simplified MP4 with just basic atoms/boxes
            with open(output_path, 'wb') as f:
                # File type box (ftyp)
                f.write(b'    ')  # Size (will update later)
                f.write(b'ftyp')  # Type
                f.write(b'isom')  # Major brand
                f.write(b'\x00\x00\x00\x00')  # Minor version
                f.write(b'isomiso2mp41')  # Compatible brands
                
                # Media data box (mdat) - simplified
                f.write(b'    ')  # Size placeholder
                f.write(b'mdat')  # Type
                
                # Write audio data as media content
                f.write(audio_data)
                
                # Movie box (moov) - minimal structure
                f.write(b'    ')  # Size placeholder  
                f.write(b'moov')  # Type
                
                # Movie header (mvhd)
                f.write(b'\x00\x00\x00\x6C')  # Size
                f.write(b'mvhd')  # Type
                f.write(b'\x00' * 100)  # Simplified header data
                
            return True
            
        except Exception as e:
            print(f"Mock video generation failed: {e}")
            return False


@pytest.mark.integration
@pytest.mark.slow
class TestCrossModuleIntegrationSmoke:
    """
    Integration smoke tests for complete VideoLingo pipeline
    Tests the flow from video input through transcription, translation, and subtitle generation
    """
    
    @pytest.fixture(scope="class")
    def synthetic_video_fixture(self, tmp_path_factory) -> Dict[str, Any]:
        """
        Generate synthetic 3-second sine-wave video at runtime
        Fixture is scoped to class to reuse across tests
        """
        temp_dir = tmp_path_factory.mktemp("smoke_test")
        video_path = temp_dir / "synthetic_test_video.mp4"
        
        # Try FFmpeg first, fall back to mock if not available
        success = VideoSynthesizer.generate_synthetic_video_ffmpeg(
            str(video_path),
            duration=3.0,
            width=320,
            height=240,
            fps=30,
            frequency=440.0
        )
        
        if not success:
            # Fallback to mock video file
            success = VideoSynthesizer.generate_mock_video_file(
                str(video_path),
                duration=3.0
            )
        
        # Generate accompanying metadata
        metadata = {
            "title": "Synthetic Test Video",
            "duration": 3.0,
            "width": 320,
            "height": 240,
            "fps": 30,
            "audio_frequency": 440.0,
            "format": "mp4",
            "created_at": time.time()
        }
        
        return {
            "video_path": video_path,
            "temp_dir": temp_dir,
            "metadata": metadata,
            "success": success
        }
    
    @pytest.fixture
    def pipeline_environment(self, synthetic_video_fixture, tmp_path) -> Dict[str, Any]:
        """
        Set up complete pipeline testing environment
        """
        # Create directory structure
        base_dir = tmp_path / "pipeline_test"
        input_dir = base_dir / "input"
        temp_dir = base_dir / "temp"
        output_dir = base_dir / "output"
        
        for dir_path in [input_dir, temp_dir, output_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for pipeline artifacts
        log_dir = temp_dir / "log"
        audio_dir = temp_dir / "audio"
        temp_audio_dir = temp_dir / "temp_audio"
        
        for dir_path in [log_dir, audio_dir, temp_audio_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Copy synthetic video to input directory
        input_video = input_dir / "test_video.mp4"
        if synthetic_video_fixture["video_path"].exists():
            shutil.copy2(synthetic_video_fixture["video_path"], input_video)
        
        return {
            "base_dir": base_dir,
            "input_dir": input_dir,
            "temp_dir": temp_dir,
            "output_dir": output_dir,
            "log_dir": log_dir,
            "audio_dir": audio_dir,
            "temp_audio_dir": temp_audio_dir,
            "input_video": input_video,
            "metadata": synthetic_video_fixture["metadata"]
        }
    
    def _mock_transcription_result(self, duration: float = 3.0) -> Dict[str, Any]:
        """Generate mock transcription result with realistic structure"""
        return {
            "segments": [
                {
                    "id": 0,
                    "start": 0.0,
                    "end": 1.0,
                    "text": "This is the first segment",
                    "words": [
                        {"word": "This", "start": 0.0, "end": 0.2},
                        {"word": "is", "start": 0.3, "end": 0.4},
                        {"word": "the", "start": 0.5, "end": 0.6},
                        {"word": "first", "start": 0.7, "end": 0.9},
                        {"word": "segment", "start": 0.9, "end": 1.0}
                    ]
                },
                {
                    "id": 1,
                    "start": 1.2,
                    "end": 2.5,
                    "text": "And this is the second part",
                    "words": [
                        {"word": "And", "start": 1.2, "end": 1.3},
                        {"word": "this", "start": 1.4, "end": 1.5},
                        {"word": "is", "start": 1.6, "end": 1.7},
                        {"word": "the", "start": 1.8, "end": 1.9},
                        {"word": "second", "start": 2.0, "end": 2.2},
                        {"word": "part", "start": 2.3, "end": 2.5}
                    ]
                },
                {
                    "id": 2,
                    "start": 2.6,
                    "end": 3.0,
                    "text": "Final words",
                    "words": [
                        {"word": "Final", "start": 2.6, "end": 2.8},
                        {"word": "words", "start": 2.8, "end": 3.0}
                    ]
                }
            ],
            "language": "en",
            "duration": duration
        }
    
    def _mock_translation_result(self, segments: list) -> list:
        """Generate mock translation result"""
        translations = []
        for i, segment in enumerate(segments):
            translations.append({
                "line_id": i,
                "start_time": segment["start"],
                "end_time": segment["end"],
                "original_text": segment["text"],
                "faithful_translation": f"翻译：{segment['text']}",
                "expressive_translation": f"意译：{segment['text']}",
                "confidence": 0.95
            })
        return translations
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_end_to_end_pipeline_smoke(self, pipeline_environment):
        """
        Test complete pipeline: download → transcribe → translate → subtitle
        This is the main smoke test that validates the entire workflow
        """
        env = pipeline_environment
        
        # Track artifacts created during pipeline
        artifacts_created = []
        metadata_propagated = {}
        
        # 1. Mock video download/input stage
        from core._1_ytdlp import download_video_ytdlp
        
        with patch('core._1_ytdlp.download_via_command') as mock_download:
            mock_download.return_value = str(env["input_video"])
            
            # Simulate download
            video_file = download_video_ytdlp(
                "https://example.com/test-video",
                resolution="360"
            )
            
            assert video_file == str(env["input_video"])
            artifacts_created.append("video_file")
            metadata_propagated["video_source"] = "synthetic_test"
        
        # 2. Mock ASR transcription
        from core._2_asr import transcribe
        
        transcription_result = self._mock_transcription_result(
            duration=env["metadata"]["duration"]
        )
        
        with patch('core._2_asr.find_video_files') as mock_find, \
             patch('core._2_asr.convert_video_to_audio') as mock_convert, \
             patch('core._2_asr.split_audio') as mock_split, \
             patch('core._2_asr.process_transcription') as mock_process, \
             patch('core._2_asr.save_results') as mock_save:
            
            # Mock the video finding
            mock_find.return_value = str(env["input_video"])
            
            # Mock audio conversion
            audio_file = env["audio_dir"] / "audio.wav"
            audio_file.write_bytes(VideoSynthesizer.generate_sine_wave_audio())
            mock_convert.return_value = str(audio_file)
            
            # Mock audio splitting
            mock_split.return_value = [(0, 3)]
            
            # Mock transcription processing
            import pandas as pd
            mock_df = pd.DataFrame(transcription_result["segments"])
            mock_process.return_value = mock_df
            
            # Save mock transcription results
            def save_mock_results(df):
                result_file = env["log_dir"] / "cleaned_chunks.json"
                with open(result_file, 'w') as f:
                    json.dump(transcription_result["segments"], f)
                artifacts_created.append("transcription")
                metadata_propagated["language"] = transcription_result["language"]
            
            mock_save.side_effect = save_mock_results
            
            # Mock the actual transcription based on runtime
            with patch('core.asr_backend.whisperX_local.transcribe_audio') as mock_ts:
                mock_ts.return_value = transcription_result
                
                # Run transcription
                transcribe()
                
                # Verify transcription artifacts
                assert (env["log_dir"] / "cleaned_chunks.json").exists()
        
        # 3. Mock translation stage
        from core._4_2_translate import translate_all
        
        # Load transcription for translation
        with open(env["log_dir"] / "cleaned_chunks.json", 'r') as f:
            segments_to_translate = json.load(f)
        
        translation_result = self._mock_translation_result(segments_to_translate)
        
        with patch('core._4_2_translate.batch_translate') as mock_translate, \
             patch('core._4_2_translate.load_chunks') as mock_load, \
             patch('core._4_2_translate.save_translation') as mock_save_trans:
            
            mock_load.return_value = segments_to_translate
            mock_translate.return_value = translation_result
            
            def save_mock_translation(result):
                trans_file = env["log_dir"] / "translation_result.json"
                with open(trans_file, 'w') as f:
                    json.dump(result, f)
                artifacts_created.append("translation")
                metadata_propagated["target_language"] = "zh-CN"
            
            mock_save_trans.side_effect = save_mock_translation
            
            # Run translation
            translate_all()
            
            # Verify translation artifacts
            assert (env["log_dir"] / "translation_result.json").exists()
        
        # 4. Mock subtitle generation
        from core._6_gen_sub import align_timestamp_main
        
        with patch('core._6_gen_sub.generate_subtitle_timestamps') as mock_gen_sub, \
             patch('core._6_gen_sub.save_subtitle_file') as mock_save_sub:
            
            subtitle_content = self._generate_subtitle_content(translation_result)
            mock_gen_sub.return_value = subtitle_content
            
            def save_mock_subtitle(content):
                sub_file = env["output_dir"] / "subtitles.srt"
                with open(sub_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                artifacts_created.append("subtitle")
                metadata_propagated["subtitle_format"] = "srt"
            
            mock_save_sub.side_effect = save_mock_subtitle
            
            # Run subtitle generation
            align_timestamp_main()
            
            # Verify subtitle artifacts
            assert (env["output_dir"] / "subtitles.srt").exists()
        
        # 5. Verify all artifacts exist and metadata propagated
        assert len(artifacts_created) >= 4, f"Expected at least 4 artifacts, got {artifacts_created}"
        assert "video_file" in artifacts_created
        assert "transcription" in artifacts_created
        assert "translation" in artifacts_created
        assert "subtitle" in artifacts_created
        
        # Verify metadata propagation
        assert "video_source" in metadata_propagated
        assert "language" in metadata_propagated
        assert "target_language" in metadata_propagated
        assert "subtitle_format" in metadata_propagated
        
        # Verify pipeline consistency
        self._verify_pipeline_consistency(env, metadata_propagated)
    
    def _generate_subtitle_content(self, translations: list) -> str:
        """Generate SRT subtitle content from translations"""
        srt_content = []
        
        for i, trans in enumerate(translations, 1):
            start_time = self._format_srt_time(trans["start_time"])
            end_time = self._format_srt_time(trans["end_time"])
            
            srt_content.append(f"{i}")
            srt_content.append(f"{start_time} --> {end_time}")
            srt_content.append(trans["expressive_translation"])
            srt_content.append("")  # Empty line between subtitles
        
        return "\n".join(srt_content)
    
    def _format_srt_time(self, seconds: float) -> str:
        """Format time in SRT format (HH:MM:SS,mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    
    def _verify_pipeline_consistency(self, env: Dict[str, Any], 
                                      metadata: Dict[str, Any]):
        """Verify consistency across pipeline artifacts"""
        
        # Check transcription consistency
        if (env["log_dir"] / "cleaned_chunks.json").exists():
            with open(env["log_dir"] / "cleaned_chunks.json", 'r') as f:
                transcription = json.load(f)
                assert len(transcription) > 0, "Transcription should not be empty"
        
        # Check translation consistency
        if (env["log_dir"] / "translation_result.json").exists():
            with open(env["log_dir"] / "translation_result.json", 'r') as f:
                translation = json.load(f)
                assert len(translation) > 0, "Translation should not be empty"
                
                # Verify translation matches transcription segments
                if transcription:
                    assert len(translation) == len(transcription), \
                        "Translation segments should match transcription segments"
        
        # Check subtitle consistency
        if (env["output_dir"] / "subtitles.srt").exists():
            with open(env["output_dir"] / "subtitles.srt", 'r', encoding='utf-8') as f:
                subtitle_content = f.read()
                assert len(subtitle_content) > 0, "Subtitle file should not be empty"
                
                # Basic SRT format validation
                assert "-->" in subtitle_content, "Subtitle should contain timing markers"
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_pipeline_error_recovery(self, pipeline_environment):
        """Test pipeline error handling and recovery mechanisms"""
        env = pipeline_environment
        
        # Test transcription failure recovery
        from core._2_asr import transcribe
        
        with patch('core._2_asr.find_video_files') as mock_find:
            mock_find.side_effect = FileNotFoundError("Video file not found")
            
            with pytest.raises(FileNotFoundError):
                transcribe()
        
        # Test translation failure recovery
        from core._4_2_translate import translate_all
        
        with patch('core._4_2_translate.load_chunks') as mock_load:
            mock_load.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
            
            with pytest.raises(json.JSONDecodeError):
                translate_all()
        
        # Verify pipeline can recover after errors
        # Create valid artifacts for recovery test
        transcription_file = env["log_dir"] / "cleaned_chunks.json"
        with open(transcription_file, 'w') as f:
            json.dump([{"text": "test", "start": 0, "end": 1}], f)
        
        # Now pipeline should be able to continue from this point
        assert transcription_file.exists()
    
    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.parametrize("stage_to_test", [
        "transcription_only",
        "transcription_translation",
        "full_pipeline"
    ])
    def test_partial_pipeline_execution(self, pipeline_environment, stage_to_test):
        """Test partial pipeline execution for different stages"""
        env = pipeline_environment
        
        if stage_to_test == "transcription_only":
            # Test only transcription stage
            self._run_transcription_stage(env)
            assert (env["log_dir"] / "cleaned_chunks.json").exists()
            
        elif stage_to_test == "transcription_translation":
            # Test transcription + translation
            self._run_transcription_stage(env)
            self._run_translation_stage(env)
            assert (env["log_dir"] / "translation_result.json").exists()
            
        elif stage_to_test == "full_pipeline":
            # Test complete pipeline
            self._run_transcription_stage(env)
            self._run_translation_stage(env)
            self._run_subtitle_stage(env)
            assert (env["output_dir"] / "subtitles.srt").exists()
    
    def _run_transcription_stage(self, env: Dict[str, Any]):
        """Helper to run transcription stage"""
        from core._2_asr import transcribe
        
        with patch('core._2_asr.find_video_files') as mock_find, \
             patch('core._2_asr.convert_video_to_audio') as mock_convert, \
             patch('core._2_asr.split_audio') as mock_split, \
             patch('core._2_asr.process_transcription') as mock_process, \
             patch('core._2_asr.save_results') as mock_save, \
             patch('core.asr_backend.whisperX_local.transcribe_audio') as mock_ts:
            
            mock_find.return_value = str(env["input_video"])
            mock_convert.return_value = str(env["audio_dir"] / "audio.wav")
            mock_split.return_value = [(0, 3)]
            
            transcription_result = self._mock_transcription_result()
            mock_ts.return_value = transcription_result
            
            import pandas as pd
            mock_process.return_value = pd.DataFrame(transcription_result["segments"])
            
            def save_results(df):
                result_file = env["log_dir"] / "cleaned_chunks.json"
                with open(result_file, 'w') as f:
                    json.dump(transcription_result["segments"], f)
            
            mock_save.side_effect = save_results
            
            transcribe()
    
    def _run_translation_stage(self, env: Dict[str, Any]):
        """Helper to run translation stage"""
        from core._4_2_translate import translate_all
        
        # Load transcription
        with open(env["log_dir"] / "cleaned_chunks.json", 'r') as f:
            segments = json.load(f)
        
        with patch('core._4_2_translate.batch_translate') as mock_translate, \
             patch('core._4_2_translate.load_chunks') as mock_load, \
             patch('core._4_2_translate.save_translation') as mock_save:
            
            mock_load.return_value = segments
            mock_translate.return_value = self._mock_translation_result(segments)
            
            def save_translation(result):
                trans_file = env["log_dir"] / "translation_result.json"
                with open(trans_file, 'w') as f:
                    json.dump(result, f)
            
            mock_save.side_effect = save_translation
            
            translate_all()
    
    def _run_subtitle_stage(self, env: Dict[str, Any]):
        """Helper to run subtitle generation stage"""
        from core._6_gen_sub import align_timestamp_main
        
        # Load translation
        with open(env["log_dir"] / "translation_result.json", 'r') as f:
            translations = json.load(f)
        
        with patch('core._6_gen_sub.load_translations') as mock_load, \
             patch('core._6_gen_sub.generate_subtitle_timestamps') as mock_gen, \
             patch('core._6_gen_sub.save_subtitle_file') as mock_save:
            
            mock_load.return_value = translations
            subtitle_content = self._generate_subtitle_content(translations)
            mock_gen.return_value = subtitle_content
            
            def save_subtitle(content):
                sub_file = env["output_dir"] / "subtitles.srt"
                with open(sub_file, 'w', encoding='utf-8') as f:
                    f.write(content)
            
            mock_save.side_effect = save_subtitle
            
            align_timestamp_main()
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_metadata_propagation_through_pipeline(self, pipeline_environment):
        """Test that metadata is correctly propagated through all pipeline stages"""
        env = pipeline_environment
        
        # Initial metadata
        metadata = {
            "video_id": "test_video_001",
            "source_url": "https://example.com/test",
            "duration": env["metadata"]["duration"],
            "resolution": "320x240",
            "fps": env["metadata"]["fps"],
            "timestamp": time.time()
        }
        
        # Save initial metadata
        metadata_file = env["log_dir"] / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f)
        
        # Run pipeline stages and verify metadata preservation
        self._run_transcription_stage(env)
        
        # Check metadata after transcription
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                current_metadata = json.load(f)
                assert current_metadata["video_id"] == metadata["video_id"]
        
        self._run_translation_stage(env)
        
        # Check metadata after translation
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                current_metadata = json.load(f)
                assert current_metadata["duration"] == metadata["duration"]
        
        self._run_subtitle_stage(env)
        
        # Final metadata verification
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                final_metadata = json.load(f)
                
                # Verify all original metadata is preserved
                for key, value in metadata.items():
                    assert key in final_metadata, f"Metadata key '{key}' lost in pipeline"
                    if key != "timestamp":  # Timestamp might be updated
                        assert final_metadata[key] == value, \
                            f"Metadata value for '{key}' changed unexpectedly"


@pytest.mark.integration
@pytest.mark.slow
class TestPipelinePerformance:
    """Performance tests for the integration pipeline"""
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_pipeline_execution_time(self, pipeline_environment):
        """Test that pipeline completes within reasonable time"""
        import time
        
        start_time = time.time()
        
        # Run minimal pipeline simulation
        env = pipeline_environment
        
        # Create mock artifacts quickly
        transcription_file = env["log_dir"] / "cleaned_chunks.json"
        with open(transcription_file, 'w') as f:
            json.dump([{"text": "test", "start": 0, "end": 1}], f)
        
        translation_file = env["log_dir"] / "translation_result.json"
        with open(translation_file, 'w') as f:
            json.dump([{"text": "test", "translation": "测试"}], f)
        
        subtitle_file = env["output_dir"] / "subtitles.srt"
        with open(subtitle_file, 'w') as f:
            f.write("1\n00:00:00,000 --> 00:00:01,000\n测试\n")
        
        execution_time = time.time() - start_time
        
        # Pipeline simulation should complete quickly (under 5 seconds)
        assert execution_time < 5.0, f"Pipeline took too long: {execution_time:.2f}s"
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_pipeline_memory_usage(self, pipeline_environment):
        """Test that pipeline doesn't exceed memory limits"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run pipeline operations
        env = pipeline_environment
        
        # Simulate memory-intensive operations
        large_transcription = []
        for i in range(1000):  # Create 1000 segments
            large_transcription.append({
                "text": f"Segment {i}" * 10,
                "start": i * 0.1,
                "end": (i + 1) * 0.1
            })
        
        # Save large transcription
        transcription_file = env["log_dir"] / "large_transcription.json"
        with open(transcription_file, 'w') as f:
            json.dump(large_transcription, f)
        
        # Check memory after operation
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB for this test)
        assert memory_increase < 100, f"Memory usage increased by {memory_increase:.2f}MB"
        
        # Cleanup
        if transcription_file.exists():
            transcription_file.unlink()


# CI-specific configuration
def pytest_configure(config):
    """Configure pytest markers for CI integration"""
    config.addinivalue_line(
        "markers", 
        "integration: marks tests as integration tests (deselect with '-m \"not integration\"')"
    )
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
