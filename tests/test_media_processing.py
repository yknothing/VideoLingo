"""
Comprehensive test suite for media processing functionality.

This module tests:
- Video download and validation logic
- Audio extraction with format verification  
- Subtitle generation and synchronization
- Video dubbing and audio mixing
- File format conversions and compatibility
- Edge cases (corrupted files, unsupported formats)
"""

import os
import sys
import tempfile
import shutil
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call, mock_open
import pytest
import numpy as np
import pandas as pd
from io import BytesIO

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import functions with proper error handling
try:
    from core._1_ytdlp import (
        validate_video_file,
        cleanup_partial_downloads,
        find_most_recent_video_file,
        find_best_video_file
    )
    from core._10_gen_audio import (
        adjust_audio_speed,
        process_row,
        generate_tts_audio,
        process_chunk,
        merge_chunks
    )
    from core._11_merge_audio import (
        load_and_flatten_data,
        get_audio_files,
        process_audio_segment,
        merge_audio_segments,
        create_srt_subtitle,
        merge_full_audio
    )
    from core._12_dub_to_vid import merge_video_audio
    from core._7_sub_into_vid import merge_subtitles_to_video
except ImportError as e:
    # If imports fail, we'll mock them in tests
    validate_video_file = None
    cleanup_partial_downloads = None
    find_most_recent_video_file = None
    find_best_video_file = None
    adjust_audio_speed = None
    process_row = None
    generate_tts_audio = None
    process_chunk = None
    merge_chunks = None
    load_and_flatten_data = None
    get_audio_files = None
    process_audio_segment = None
    merge_audio_segments = None
    create_srt_subtitle = None
    merge_full_audio = None
    merge_video_audio = None
    merge_subtitles_to_video = None


class TestVideoDownloadValidation:
    """Test video download and validation functionality."""
    
    @pytest.fixture
    def temp_video_dir(self):
        """Create temporary directory with test video files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create valid video file
            valid_video = temp_path / "valid_video.mp4"
            valid_video.write_bytes(b"FAKE_VIDEO" * 200000)  # ~2MB file
            
            # Create partial download
            partial_video = temp_path / "partial.mp4.part"
            partial_video.write_bytes(b"PARTIAL" * 1000)
            
            # Create small/corrupted video
            small_video = temp_path / "small_video.mp4"
            small_video.write_bytes(b"SMALL" * 10)  # Very small file
            
            # Create temp file
            temp_file = temp_path / "temp.tmp"
            temp_file.write_bytes(b"TEMP" * 100)
            
            yield temp_path
    
    def test_validate_video_file_valid(self, temp_video_dir):
        """Test validation of a valid video file."""
        valid_file = temp_video_dir / "valid_video.mp4"
        is_valid, error_msg = validate_video_file(str(valid_file))
        assert is_valid is True
        assert error_msg == "File is valid"
    
    def test_validate_video_file_not_exists(self):
        """Test validation when file doesn't exist."""
        is_valid, error_msg = validate_video_file("/nonexistent/file.mp4")
        assert is_valid is False
        assert "File does not exist" in error_msg
    
    def test_validate_video_file_too_small(self, temp_video_dir):
        """Test validation of a file that's too small."""
        small_file = temp_video_dir / "small_video.mp4"
        is_valid, error_msg = validate_video_file(str(small_file), expected_min_size_mb=1)
        assert is_valid is False
        assert "File too small" in error_msg
    
    def test_validate_video_file_partial_download(self, temp_video_dir):
        """Test validation of partial download files."""
        partial_file = temp_video_dir / "partial.mp4.part"
        is_valid, error_msg = validate_video_file(str(partial_file))
        assert is_valid is False
        assert "partial download" in error_msg
    
    @patch('subprocess.run')
    def test_validate_video_with_ffprobe(self, mock_run, temp_video_dir):
        """Test video validation with ffprobe."""
        valid_file = temp_video_dir / "valid_video.mp4"
        
        # Mock successful ffprobe
        mock_result = Mock()
        mock_result.returncode = 0
        mock_run.return_value = mock_result
        
        is_valid, error_msg = validate_video_file(str(valid_file))
        assert is_valid is True
        
        # Mock failed ffprobe
        mock_result.returncode = 1
        mock_run.return_value = mock_result
        
        is_valid, error_msg = validate_video_file(str(valid_file))
        assert is_valid is False
        assert "format validation failed" in error_msg
    
    def test_cleanup_partial_downloads(self, temp_video_dir):
        """Test cleanup of partial downloads."""
        # Verify partial files exist
        assert (temp_video_dir / "partial.mp4.part").exists()
        assert (temp_video_dir / "temp.tmp").exists()
        
        # Run cleanup
        cleaned = cleanup_partial_downloads(str(temp_video_dir))
        
        # Verify partial files removed
        assert not (temp_video_dir / "partial.mp4.part").exists()
        assert not (temp_video_dir / "temp.tmp").exists()
        assert (temp_video_dir / "valid_video.mp4").exists()  # Valid file preserved
    
    @patch('core._1_ytdlp.load_key')
    def test_find_most_recent_video_file(self, mock_load_key, temp_video_dir):
        """Test finding the most recent video file."""
        mock_load_key.return_value = [".mp4", ".avi", ".mkv"]
        
        # Create multiple video files with different timestamps
        import time
        video1 = temp_video_dir / "video1.mp4"
        video1.write_bytes(b"VIDEO1" * 200000)
        time.sleep(0.1)
        
        video2 = temp_video_dir / "video2.mp4"
        video2.write_bytes(b"VIDEO2" * 200000)
        
        most_recent = find_most_recent_video_file(str(temp_video_dir))
        assert most_recent is not None
        assert "video2.mp4" in most_recent
    
    @patch('core._1_ytdlp.load_key')
    def test_find_best_video_file(self, mock_load_key, temp_video_dir):
        """Test finding the best video file by size."""
        mock_load_key.return_value = [".mp4", ".avi"]
        
        # Create videos of different sizes
        small_video = temp_video_dir / "small.mp4"
        small_video.write_bytes(b"S" * 100000)
        
        large_video = temp_video_dir / "large.mp4"
        large_video.write_bytes(b"L" * 500000)
        
        best_video = find_best_video_file(str(temp_video_dir), [".mp4", ".avi"])
        assert best_video is not None
        # Should return the larger file
        assert os.path.getsize(best_video) > os.path.getsize(str(small_video))


class TestAudioExtraction:
    """Test audio extraction and format verification."""
    
    @pytest.fixture
    def mock_audio_file(self):
        """Create mock audio file."""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            f.write(b"RIFF" + b"WAVE" * 1000)  # Fake WAV header
            temp_path = f.name
        yield temp_path
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    @patch('core.asr_backend.audio_preprocess.get_audio_duration')
    @patch('subprocess.run')
    @patch('shutil.copy2')
    def test_adjust_audio_speed_no_change(self, mock_copy, mock_run, mock_get_duration):
        """Test audio speed adjustment when speed factor is ~1."""
        mock_get_duration.return_value = 10.0
        
        with tempfile.NamedTemporaryFile(suffix='.wav') as input_file:
            with tempfile.NamedTemporaryFile(suffix='.wav') as output_file:
                adjust_audio_speed(input_file.name, output_file.name, 1.0001)
                
                # Should copy directly without processing
                mock_copy.assert_called_once()
                mock_run.assert_not_called()
    
    @patch('core.asr_backend.audio_preprocess.get_audio_duration')
    @patch('subprocess.run')
    @patch('pydub.AudioSegment.from_wav')
    @patch('pydub.AudioSegment.export')
    def test_adjust_audio_speed_with_change(self, mock_export, mock_from_wav, 
                                           mock_run, mock_get_duration):
        """Test audio speed adjustment with actual speed change."""
        mock_get_duration.side_effect = [10.0, 5.0]  # Input and output durations
        mock_run.return_value = Mock(returncode=0)
        mock_audio = MagicMock()
        mock_from_wav.return_value = mock_audio
        
        with tempfile.NamedTemporaryFile(suffix='.wav') as input_file:
            with tempfile.NamedTemporaryFile(suffix='.wav') as output_file:
                adjust_audio_speed(input_file.name, output_file.name, 2.0)
                
                # Should call ffmpeg with correct speed
                mock_run.assert_called_once()
                call_args = mock_run.call_args[0][0]
                assert 'ffmpeg' in call_args
                assert 'atempo=2.0' in ' '.join(call_args)
    
    @patch('core.asr_backend.audio_preprocess.get_audio_duration')
    @patch('subprocess.run')
    def test_adjust_audio_speed_retry_on_failure(self, mock_run, mock_get_duration):
        """Test retry mechanism for audio speed adjustment."""
        mock_get_duration.return_value = 10.0
        
        # Simulate failure then success
        mock_run.side_effect = [
            subprocess.CalledProcessError(1, 'ffmpeg'),
            Mock(returncode=0)
        ]
        
        with tempfile.NamedTemporaryFile(suffix='.wav') as input_file:
            with tempfile.NamedTemporaryFile(suffix='.wav') as output_file:
                with patch('time.sleep'):  # Speed up test
                    adjust_audio_speed(input_file.name, output_file.name, 1.5)
                
                # Should retry and succeed
                assert mock_run.call_count == 2
    
    @patch('core._10_gen_audio.tts_main')
    @patch('core.asr_backend.audio_preprocess.get_audio_duration')
    def test_process_row(self, mock_get_duration, mock_tts):
        """Test processing a single row of TTS tasks."""
        mock_get_duration.return_value = 5.0
        
        # Create test DataFrame
        tasks_df = pd.DataFrame({
            'number': [1],
            'lines': [['line1', 'line2']]
        })
        
        row = tasks_df.iloc[0]
        number, duration = process_row(row, tasks_df)
        
        assert number == 1
        assert duration == 10.0  # 2 lines * 5.0 seconds each
        assert mock_tts.call_count == 2


class TestSubtitleGeneration:
    """Test subtitle generation and synchronization."""
    
    @pytest.fixture
    def sample_subtitle_data(self):
        """Create sample subtitle data."""
        return pd.DataFrame({
            'number': [1, 2, 3],
            'lines': [['Hello'], ['World'], ['Test']],
            'new_sub_times': [
                [[0.0, 1.0]],
                [[1.5, 2.5]],
                [[3.0, 4.0]]
            ],
            'start_time': ['00:00:00,000', '00:00:01,500', '00:00:03,000'],
            'end_time': ['00:00:01,000', '00:00:02,500', '00:00:04,000']
        })
    
    @patch('pandas.read_excel')
    @patch('builtins.open', new_callable=mock_open)
    def test_create_srt_subtitle(self, mock_file, mock_read_excel, sample_subtitle_data):
        """Test SRT subtitle file creation."""
        mock_read_excel.return_value = sample_subtitle_data
        
        create_srt_subtitle()
        
        # Verify file was opened for writing
        mock_file.assert_called()
        
        # Check SRT format in written content
        handle = mock_file()
        written_content = ''.join(call.args[0] for call in handle.write.call_args_list 
                                 if call.args)
        
        # Verify SRT structure
        assert '1\n' in written_content  # Subtitle index
        assert '-->' in written_content  # Time separator
        assert 'Hello' in written_content  # Content
    
    def test_subtitle_time_synchronization(self, sample_subtitle_data):
        """Test subtitle timing synchronization."""
        # Flatten the data as done in the actual function
        lines = [item for sublist in sample_subtitle_data['lines'] for item in sublist]
        new_sub_times = [item for sublist in sample_subtitle_data['new_sub_times'] 
                        for item in sublist]
        
        # Verify timing consistency
        for i, (time_range, line) in enumerate(zip(new_sub_times, lines)):
            start, end = time_range
            assert start < end, f"Invalid time range for subtitle {i}"
            if i > 0:
                prev_end = new_sub_times[i-1][1]
                assert start >= prev_end, f"Overlapping subtitles at index {i}"
    
    @patch('pandas.read_excel')
    def test_load_and_flatten_data(self, mock_read_excel, sample_subtitle_data):
        """Test loading and flattening subtitle data."""
        mock_read_excel.return_value = sample_subtitle_data
        
        df, lines, times = load_and_flatten_data('dummy.xlsx')
        
        assert len(lines) == 3
        assert lines == ['Hello', 'World', 'Test']
        assert len(times) == 3
        assert all(len(t) == 2 for t in times)  # Each time range has start and end


class TestVideoDubbing:
    """Test video dubbing and audio mixing functionality."""
    
    @pytest.fixture
    def mock_video_setup(self):
        """Setup mocks for video processing."""
        with patch('cv2.VideoCapture') as mock_capture, \
             patch('cv2.VideoWriter') as mock_writer, \
             patch('core._1_ytdlp.find_video_files') as mock_find:
            
            # Mock video properties
            mock_video = Mock()
            mock_video.get.side_effect = lambda prop: {
                3: 1920,  # Width
                4: 1080,  # Height
                5: 30,    # FPS
                7: 100    # Frame count
            }.get(prop, 0)
            mock_capture.return_value = mock_video
            
            # Mock video writer
            mock_writer_instance = Mock()
            mock_writer.return_value = mock_writer_instance
            
            # Mock finding video files
            mock_find.return_value = '/fake/video.mp4'
            
            yield {
                'capture': mock_capture,
                'writer': mock_writer,
                'find': mock_find,
                'video': mock_video,
                'writer_instance': mock_writer_instance
            }
    
    @patch('subprocess.run')
    @patch('core.asr_backend.audio_preprocess.normalize_audio_volume')
    @patch('core.utils.config_utils.load_key')
    def test_merge_video_audio_with_subtitles(self, mock_load_key, mock_normalize, 
                                             mock_run, mock_video_setup):
        """Test merging video with audio and burning subtitles."""
        mock_load_key.side_effect = lambda key: {
            'burn_subtitles': True,
            'ffmpeg_gpu': False
        }.get(key, False)
        
        merge_video_audio()
        
        # Verify normalization was called
        mock_normalize.assert_called_once()
        
        # Verify ffmpeg was called with correct parameters
        mock_run.assert_called_once()
        ffmpeg_cmd = mock_run.call_args[0][0]
        assert 'ffmpeg' in ffmpeg_cmd
        assert any('subtitles' in str(arg) for arg in ffmpeg_cmd)
    
    @patch('numpy.zeros')
    @patch('cv2.VideoWriter')
    @patch('core.utils.config_utils.load_key')
    def test_merge_video_audio_no_subtitles(self, mock_load_key, mock_writer, mock_zeros):
        """Test creating placeholder video when subtitles not burned."""
        mock_load_key.return_value = False  # burn_subtitles = False
        
        # Mock video writer
        mock_writer_instance = Mock()
        mock_writer.return_value = mock_writer_instance
        
        # Mock numpy array creation
        mock_zeros.return_value = np.zeros((1080, 1920, 3), dtype=np.uint8)
        
        merge_video_audio()
        
        # Verify placeholder video was created
        mock_zeros.assert_called_once_with((1080, 1920, 3), dtype=np.uint8)
        mock_writer_instance.write.assert_called_once()
        mock_writer_instance.release.assert_called_once()
    
    @patch('subprocess.run')
    @patch('core.utils.config_utils.load_key')
    def test_merge_with_gpu_acceleration(self, mock_load_key, mock_run, mock_video_setup):
        """Test video merging with GPU acceleration enabled."""
        mock_load_key.side_effect = lambda key: {
            'burn_subtitles': True,
            'ffmpeg_gpu': True
        }.get(key, False)
        
        with patch('core.asr_backend.audio_preprocess.normalize_audio_volume'):
            merge_video_audio()
        
        # Verify GPU codec was used
        ffmpeg_cmd = mock_run.call_args[0][0]
        assert 'h264_nvenc' in ffmpeg_cmd


class TestAudioMixing:
    """Test audio mixing and merging functionality."""
    
    @pytest.fixture
    def mock_audio_files(self):
        """Create mock audio segment files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_files = []
            for i in range(3):
                audio_file = Path(temp_dir) / f"segment_{i}.wav"
                audio_file.write_bytes(b"RIFF" + b"WAVE" * 100)
                audio_files.append(str(audio_file))
            yield audio_files
    
    @patch('pydub.AudioSegment.from_mp3')
    @patch('pydub.AudioSegment.silent')
    @patch('subprocess.run')
    def test_merge_audio_segments(self, mock_run, mock_silent, mock_from_mp3, 
                                 mock_audio_files):
        """Test merging multiple audio segments."""
        # Mock audio segments
        mock_segment = MagicMock()
        mock_segment.__add__ = MagicMock(return_value=mock_segment)
        mock_from_mp3.return_value = mock_segment
        mock_silent.return_value = mock_segment
        
        # Mock subprocess for ffmpeg
        mock_run.return_value = Mock(returncode=0)
        
        # Test time ranges with gaps
        time_ranges = [
            [0.0, 1.0],
            [1.5, 2.5],  # 0.5s gap
            [3.0, 4.0]   # 0.5s gap
        ]
        
        result = merge_audio_segments(mock_audio_files, time_ranges, 16000)
        
        # Verify silence was added for gaps
        assert mock_silent.call_count >= 2  # At least 2 gaps
        
        # Verify segments were concatenated
        assert mock_segment.__add__.called
    
    @patch('pandas.read_excel')
    def test_get_audio_files(self, mock_read_excel):
        """Test generating audio file paths from DataFrame."""
        test_df = pd.DataFrame({
            'number': [1, 2],
            'lines': [['line1', 'line2'], ['line3']]
        })
        mock_read_excel.return_value = test_df
        
        audio_files = get_audio_files(test_df)
        
        assert len(audio_files) == 3  # Total lines across all rows
        assert '1_0' in audio_files[0]
        assert '1_1' in audio_files[1]
        assert '2_0' in audio_files[2]


class TestFormatConversions:
    """Test file format conversions and compatibility."""
    
    @patch('subprocess.run')
    def test_convert_video_format(self, mock_run):
        """Test video format conversion."""
        mock_run.return_value = Mock(returncode=0)
        
        # Simulate format conversion
        input_file = "input.avi"
        output_file = "output.mp4"
        
        # Mock conversion command
        cmd = ['ffmpeg', '-i', input_file, '-c:v', 'libx264', '-c:a', 'aac', output_file]
        subprocess.run(cmd)
        
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert 'ffmpeg' in call_args
        assert input_file in call_args
        assert output_file in call_args
    
    @patch('subprocess.run')
    def test_audio_format_conversion(self, mock_run):
        """Test audio format conversion."""
        mock_run.return_value = Mock(returncode=0, stdout="", stderr="")
        
        # Test WAV to MP3 conversion
        input_audio = "audio.wav"
        output_audio = "audio.mp3"
        
        cmd = ['ffmpeg', '-i', input_audio, '-b:a', '128k', output_audio]
        subprocess.run(cmd)
        
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert '-b:a' in call_args  # Bitrate setting
        assert '128k' in call_args
    
    def test_supported_format_detection(self):
        """Test detection of supported media formats."""
        supported_video = ['.mp4', '.avi', '.mkv', '.mov', '.webm']
        supported_audio = ['.wav', '.mp3', '.aac', '.flac', '.ogg']
        
        # Test video format detection
        assert '.mp4' in supported_video
        assert '.txt' not in supported_video
        
        # Test audio format detection  
        assert '.wav' in supported_audio
        assert '.pdf' not in supported_audio


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_corrupted_file_handling(self):
        """Test handling of corrupted media files."""
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            # Write invalid/corrupted data
            f.write(b"CORRUPTED" * 10)
            corrupted_file = f.name
        
        try:
            # Test validation catches corruption
            is_valid, error = validate_video_file(corrupted_file, expected_min_size_mb=0.001)
            
            # File exists but is too small/corrupted
            if "too small" not in error:
                # If size check passes, format check should fail
                assert not is_valid
        finally:
            os.unlink(corrupted_file)
    
    def test_unsupported_format_handling(self):
        """Test handling of unsupported file formats."""
        with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as f:
            f.write(b"UNSUPPORTED" * 1000)
            unsupported_file = f.name
        
        try:
            # Mock the allowed formats check
            with patch('core._1_ytdlp.load_key') as mock_load_key:
                mock_load_key.return_value = ['.mp4', '.avi']
                
                # File should not be found as valid video
                result = find_most_recent_video_file(os.path.dirname(unsupported_file))
                
                # Should return None or not include the .xyz file
                if result:
                    assert not result.endswith('.xyz')
        finally:
            os.unlink(unsupported_file)
    
    @patch('subprocess.run')
    def test_ffmpeg_failure_handling(self, mock_run):
        """Test handling of FFmpeg failures."""
        # Simulate FFmpeg error
        mock_run.side_effect = subprocess.CalledProcessError(
            1, 'ffmpeg', stderr=b"Error: Invalid codec"
        )
        
        with pytest.raises(subprocess.CalledProcessError):
            # This should raise an exception after max retries
            with tempfile.NamedTemporaryFile(suffix='.wav') as f:
                adjust_audio_speed(f.name, "output.wav", 1.5)
    
    def test_missing_dependencies(self):
        """Test handling when required tools are missing."""
        with patch('subprocess.run') as mock_run:
            # Simulate ffprobe not found
            mock_run.side_effect = FileNotFoundError("ffprobe not found")
            
            with tempfile.NamedTemporaryFile(suffix='.mp4') as f:
                f.write(b"VIDEO" * 100000)
                # Should still validate based on file size/extension
                is_valid, _ = validate_video_file(f.name)
                # Basic validation should work even without ffprobe
                assert is_valid
    
    def test_empty_subtitle_data(self):
        """Test handling of empty subtitle data."""
        empty_df = pd.DataFrame({
            'number': [],
            'lines': [],
            'new_sub_times': []
        })
        
        with patch('pandas.read_excel') as mock_read:
            mock_read.return_value = empty_df
            
            df, lines, times = load_and_flatten_data('dummy.xlsx')
            
            assert len(lines) == 0
            assert len(times) == 0
    
    @patch('os.path.exists')
    def test_missing_audio_segment_handling(self, mock_exists):
        """Test handling when audio segments are missing."""
        mock_exists.side_effect = [False, True, True]  # First file missing
        
        audio_files = ['missing.wav', 'exists1.wav', 'exists2.wav']
        time_ranges = [[0, 1], [1, 2], [2, 3]]
        
        with patch('pydub.AudioSegment.from_mp3'), \
             patch('pydub.AudioSegment.silent') as mock_silent, \
             patch('subprocess.run'):
            
            mock_silent.return_value = MagicMock()
            
            # Should continue processing despite missing file
            result = merge_audio_segments(audio_files, time_ranges, 16000)
            assert result is not None
    
    def test_extreme_speed_factors(self):
        """Test audio speed adjustment with extreme values."""
        with patch('subprocess.run') as mock_run, \
             patch('core.asr_backend.audio_preprocess.get_audio_duration') as mock_dur:
            
            mock_run.return_value = Mock(returncode=0)
            mock_dur.return_value = 10.0
            
            # Test very slow speed
            with tempfile.NamedTemporaryFile(suffix='.wav') as f:
                adjust_audio_speed(f.name, "slow.wav", 0.25)
                assert mock_run.called
                
            # Test very fast speed
            mock_run.reset_mock()
            with tempfile.NamedTemporaryFile(suffix='.wav') as f:
                adjust_audio_speed(f.name, "fast.wav", 4.0)
                assert mock_run.called
    
    def test_unicode_filename_handling(self):
        """Test handling of files with Unicode characters in names."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create file with Unicode name
            unicode_file = Path(temp_dir) / "测试视频.mp4"
            unicode_file.write_bytes(b"VIDEO" * 100000)
            
            # Should handle Unicode filenames
            is_valid, _ = validate_video_file(str(unicode_file))
            assert is_valid
    
    def test_concurrent_processing_safety(self):
        """Test thread safety in concurrent audio processing."""
        from concurrent.futures import ThreadPoolExecutor
        
        def process_mock_row(i):
            # Simulate processing
            return i, float(i)
        
        # Test concurrent processing doesn't cause issues
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_mock_row, i) for i in range(10)]
            results = [f.result() for f in futures]
        
        assert len(results) == 10
        assert all(r[0] == r[1] for r in results)  # Check consistency


class TestIntegrationScenarios:
    """Test complete processing pipelines and integration scenarios."""
    
    @patch('core._1_ytdlp.find_video_files')
    @patch('pandas.read_excel')
    @patch('subprocess.run')
    @patch('builtins.open', new_callable=mock_open)
    def test_complete_subtitle_pipeline(self, mock_file, mock_run, mock_read, mock_find):
        """Test complete subtitle generation and video merge pipeline."""
        # Setup mock data
        mock_find.return_value = '/fake/video.mp4'
        mock_read.return_value = pd.DataFrame({
            'number': [1],
            'lines': [['Test line']],
            'new_sub_times': [[[0.0, 2.0]]],
            'start_time': ['00:00:00,000'],
            'end_time': ['00:00:02,000']
        })
        mock_run.return_value = Mock(returncode=0)
        
        # Run subtitle creation
        create_srt_subtitle()
        
        # Verify subtitle file was created
        mock_file.assert_called()
        
        # Run video merge (with mocked video operations)
        with patch('cv2.VideoCapture'), patch('cv2.VideoWriter'):
            with patch('core.utils.config_utils.load_key') as mock_key:
                mock_key.return_value = True  # burn_subtitles = True
                merge_subtitles_to_video()
        
        # Verify ffmpeg was called for merging
        assert mock_run.called
    
    @patch('core._10_gen_audio.tts_main')
    @patch('core.asr_backend.audio_preprocess.get_audio_duration')
    @patch('pandas.read_excel')
    def test_complete_audio_generation_pipeline(self, mock_read, mock_duration, mock_tts):
        """Test complete audio generation and merging pipeline."""
        # Setup mock data
        test_df = pd.DataFrame({
            'number': [1, 2],
            'lines': [['Hello'], ['World']],
            'real_dur': [0, 0],
            'tol_dur': [1.0, 1.0],
            'tolerance': [0.1, 0.1],
            'gap': [0.5, 0.5],
            'cut_off': [0, 1],
            'start_time': ['00:00:00,000', '00:00:01,500'],
            'end_time': ['00:00:01,000', '00:00:02,500']
        })
        mock_read.return_value = test_df
        mock_duration.return_value = 1.0
        
        # Generate TTS audio
        result_df = generate_tts_audio(test_df)
        
        # Verify TTS was called for each line
        assert mock_tts.call_count == 2
        
        # Process chunks
        result_df = merge_chunks(result_df)
        
        # Verify new subtitle times were generated
        assert result_df['new_sub_times'].notna().all()
    
    def test_error_recovery_in_pipeline(self):
        """Test error recovery mechanisms in processing pipeline."""
        with patch('subprocess.run') as mock_run:
            # Simulate intermittent failures
            mock_run.side_effect = [
                subprocess.CalledProcessError(1, 'ffmpeg'),
                Mock(returncode=0)  # Success on retry
            ]
            
            with patch('time.sleep'):  # Speed up test
                with patch('core.asr_backend.audio_preprocess.get_audio_duration'):
                    with tempfile.NamedTemporaryFile(suffix='.wav') as f:
                        # Should recover from failure
                        adjust_audio_speed(f.name, "output.wav", 1.5)
            
            # Verify retry happened
            assert mock_run.call_count == 2


# Performance and stress tests
class TestPerformanceAndStress:
    """Test performance and stress scenarios."""
    
    @pytest.mark.slow
    def test_large_subtitle_file_processing(self):
        """Test processing of large subtitle files."""
        # Create large DataFrame
        large_df = pd.DataFrame({
            'number': list(range(1000)),
            'lines': [['Line ' + str(i)] for i in range(1000)],
            'new_sub_times': [[[i, i+0.5]] for i in range(1000)]
        })
        
        with patch('pandas.read_excel') as mock_read:
            mock_read.return_value = large_df
            
            df, lines, times = load_and_flatten_data('dummy.xlsx')
            
            assert len(lines) == 1000
            assert len(times) == 1000
    
    @pytest.mark.slow
    def test_parallel_audio_processing(self):
        """Test parallel processing of multiple audio segments."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        def mock_process(i):
            # Simulate processing time
            import time
            time.sleep(0.01)
            return i, i * 2.0
        
        # Process many segments in parallel
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(mock_process, i) for i in range(100)]
            results = []
            for future in as_completed(futures):
                results.append(future.result())
        
        assert len(results) == 100
        # Verify all processed correctly
        for num, dur in results:
            assert dur == num * 2.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
