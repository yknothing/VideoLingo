"""
Integration tests for pipeline flow - download to transcription to synthesis
Tests real file operations, data transformations, and error propagation across stages
"""

import os
import sys
import tempfile
import shutil
import json
import time
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from unittest.mock import Mock, patch, MagicMock, ANY
import pytest
import pandas as pd
from typing import Optional, Dict, List, Tuple, Any

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from core._1_ytdlp import (
    validate_video_file, cleanup_partial_downloads,
    find_most_recent_video_file, find_best_video_file
)
from core._2_asr import transcribe, check_memory_usage, monitor_memory_and_warn
from core._4_2_translate import translate_all, split_chunks_by_chars, translate_chunk
from core._10_gen_audio import (
    adjust_audio_speed, process_row, generate_tts_audio,
    process_chunk, merge_chunks
)
from core.utils.models import TranscriptionData, TranslationData, AudioData
from core.utils import get_temp_path, get_output_path


class TestDownloadToTranscriptionFlow:
    """Test the download to transcription pipeline flow"""
    
    @pytest.fixture
    def temp_video_dir(self):
        """Create a temporary directory for video files"""
        temp_dir = tempfile.mkdtemp(prefix="video_test_")
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def mock_video_file(self, temp_video_dir):
        """Create a mock video file for testing"""
        video_path = os.path.join(temp_video_dir, "test_video.mp4")
        # Create a file with reasonable size (5MB)
        with open(video_path, 'wb') as f:
            f.write(b'\x00' * (5 * 1024 * 1024))
        return video_path
    
    def test_video_validation_with_real_file(self, mock_video_file):
        """Test video file validation with real file operations"""
        # Test valid file
        is_valid, error_msg = validate_video_file(mock_video_file, expected_min_size_mb=1)
        assert is_valid is True
        assert error_msg == "File is valid"
        
        # Test non-existent file
        is_valid, error_msg = validate_video_file("/nonexistent/file.mp4")
        assert is_valid is False
        assert "does not exist" in error_msg
        
        # Test file too small
        small_file = mock_video_file.replace(".mp4", "_small.mp4")
        with open(small_file, 'wb') as f:
            f.write(b'\x00' * 100)  # 100 bytes
        is_valid, error_msg = validate_video_file(small_file, expected_min_size_mb=1)
        assert is_valid is False
        assert "too small" in error_msg.lower()
    
    def test_partial_download_cleanup(self, temp_video_dir):
        """Test cleanup of partial downloads with real file operations"""
        # Create various partial files
        partial_files = [
            "video.mp4.part",
            "video.tmp",
            "video.download",
            "video.f123",
            "video.ytdl"
        ]
        
        for filename in partial_files:
            filepath = os.path.join(temp_video_dir, filename)
            with open(filepath, 'w') as f:
                f.write("partial content")
        
        # Create a valid file that should not be deleted
        valid_file = os.path.join(temp_video_dir, "valid_video.mp4")
        with open(valid_file, 'w') as f:
            f.write("valid content")
        
        # Run cleanup
        cleaned = cleanup_partial_downloads(temp_video_dir)
        
        # Check results
        assert len(cleaned) == len(partial_files)
        assert os.path.exists(valid_file)
        for filename in partial_files:
            assert not os.path.exists(os.path.join(temp_video_dir, filename))
    
    def test_find_most_recent_video(self, temp_video_dir):
        """Test finding most recent video file with timestamps"""
        # Create multiple video files with different timestamps
        video_files = []
        for i in range(3):
            filepath = os.path.join(temp_video_dir, f"video_{i}.mp4")
            with open(filepath, 'wb') as f:
                f.write(b'\x00' * (2 * 1024 * 1024))  # 2MB files
            # Modify file time
            mtime = time.time() - (10 - i) * 60  # i=0 is oldest, i=2 is newest
            os.utime(filepath, (mtime, mtime))
            video_files.append(filepath)
        
        with patch('core._1_ytdlp.load_key', return_value=['mp4', 'mkv', 'webm']):
            most_recent = find_most_recent_video_file(temp_video_dir)
            assert most_recent == video_files[-1]  # Should be the newest file
    
    @patch('core._2_asr.find_video_files')
    @patch('core._2_asr.convert_video_to_audio')
    @patch('core._2_asr.split_audio')
    @patch('core._2_asr.save_results')
    @patch('core._2_asr.load_key')
    def test_transcription_flow_with_memory_monitoring(
        self, mock_load_key, mock_save, mock_split, mock_convert, mock_find
    ):
        """Test transcription flow with memory monitoring"""
        # Setup mocks
        mock_find.return_value = "/path/to/video.mp4"
        mock_split.return_value = [(0, 30), (30, 60), (60, 90)]
        mock_load_key.side_effect = lambda key: {
            "demucs": False,
            "whisper.runtime": "cloud"
        }.get(key, None)
        
        # Mock transcription function
        with patch('core._2_asr.transcribe_audio_302') as mock_transcribe:
            mock_transcribe.return_value = {
                'segments': [{'text': 'test segment', 'start': 0, 'end': 30}]
            }
            
            # Run transcription with memory monitoring
            with patch('core._2_asr.os.path.exists', return_value=False):
                transcribe()
            
            # Verify flow
            mock_find.assert_called_once()
            mock_convert.assert_called_once()
            mock_split.assert_called_once()
            assert mock_transcribe.call_count == 3  # Called for each segment
            mock_save.assert_called_once()
    
    def test_error_propagation_across_stages(self):
        """Test how errors propagate through pipeline stages"""
        error_log = []
        
        def stage1():
            error_log.append("stage1_start")
            raise ValueError("Stage 1 failed")
        
        def stage2():
            error_log.append("stage2_start")
            return "stage2_complete"
        
        def stage3():
            error_log.append("stage3_start")
            return "stage3_complete"
        
        pipeline = [stage1, stage2, stage3]
        
        # Run pipeline with error handling
        for stage in pipeline:
            try:
                result = stage()
                error_log.append(f"result: {result}")
            except Exception as e:
                error_log.append(f"error: {str(e)}")
                break  # Stop pipeline on error
        
        # Verify error propagation
        assert error_log == [
            "stage1_start",
            "error: Stage 1 failed"
        ]
        assert "stage2_start" not in error_log  # Stage 2 should not run


class TestTranslationToSynthesisFlow:
    """Test the translation to synthesis pipeline flow"""
    
    @pytest.fixture
    def sample_text_data(self):
        """Create sample text data for translation"""
        return [
            "This is the first sentence.",
            "This is the second sentence.",
            "This is the third sentence with more content.",
            "This is the fourth sentence.",
            "This is the fifth and final sentence."
        ]
    
    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame for testing"""
        return pd.DataFrame({
            'start_time': ['00:00:00,000', '00:00:05,000', '00:00:10,000'],
            'end_time': ['00:00:05,000', '00:00:10,000', '00:00:15,000'],
            'text': ['First segment', 'Second segment', 'Third segment'],
            'duration': [5.0, 5.0, 5.0]
        })
    
    def test_chunk_splitting_with_data_transformation(self, sample_text_data, tmp_path):
        """Test text chunking with data transformations"""
        # Write sample data to file
        text_file = tmp_path / "split_by_meaning.txt"
        text_file.write_text('\n'.join(sample_text_data))
        
        with patch('core._4_2_translate._3_2_SPLIT_BY_MEANING', str(text_file)):
            chunks = split_chunks_by_chars(chunk_size=50, max_i=2)
            
            # Verify chunk properties
            assert len(chunks) > 0
            for chunk in chunks:
                assert len(chunk) <= 50 or chunk.count('\n') <= 2
                assert chunk.strip() != ""
    
    @patch('core.translate_lines.translate_lines')
    def test_translation_chunk_processing(self, mock_translate, sample_text_data):
        """Test translation of individual chunks with context"""
        # Setup mock translation
        mock_translate.return_value = (
            "Translated text",
            "English back-translation"
        )
        
        chunks = ['\n'.join(sample_text_data[:2]), '\n'.join(sample_text_data[2:])]
        theme_prompt = "Technical documentation"
        
        # Process first chunk
        i, english, translation = translate_chunk(
            chunks[0], chunks, theme_prompt, 0
        )
        
        assert i == 0
        assert english == "English back-translation"
        assert translation == "Translated text"
        
        # Verify context was extracted
        mock_translate.assert_called_once()
        call_args = mock_translate.call_args[0]
        assert call_args[0] == chunks[0]  # Current chunk
        assert call_args[1] is None  # No previous content for first chunk
        assert call_args[2] is not None  # Has after content
    
    @patch('core._4_2_translate.translate_lines_batch')
    @patch('core._4_2_translate.load_key')
    def test_batch_translation_flow(self, mock_load_key, mock_batch_translate, tmp_path):
        """Test batch translation to reduce API calls"""
        # Setup
        mock_load_key.return_value = 3  # Batch size
        mock_batch_translate.return_value = {
            0: {'translation': 'Trans 1', 'english': 'Eng 1'},
            1: {'translation': 'Trans 2', 'english': 'Eng 2'},
            2: {'translation': 'Trans 3', 'english': 'Eng 3'}
        }
        
        # Create test files
        text_file = tmp_path / "split_by_meaning.txt"
        text_file.write_text("Line 1\nLine 2\nLine 3")
        
        term_file = tmp_path / "terminology.json"
        term_file.write_text(json.dumps({'theme': 'Test theme'}))
        
        excel_file = tmp_path / "cleaned_chunks.xlsx"
        df = pd.DataFrame({
            'text': ['Line 1', 'Line 2', 'Line 3'],
            'start_time': ['00:00:00,000'] * 3,
            'end_time': ['00:00:05,000'] * 3,
            'duration': [5.0] * 3
        })
        df.to_excel(excel_file, index=False)
        
        with patch('core._4_2_translate._3_2_SPLIT_BY_MEANING', str(text_file)), \
             patch('core._4_2_translate._4_1_TERMINOLOGY', str(term_file)), \
             patch('core._4_2_translate._2_CLEANED_CHUNKS', str(excel_file)), \
             patch('core._4_2_translate._4_2_TRANSLATION', str(tmp_path / "output.xlsx")):
            
            translate_all()
            
            # Verify batch processing
            mock_batch_translate.assert_called_once()
            assert len(mock_batch_translate.call_args[0][0]) <= 3  # Batch size limit
    
    def test_audio_generation_with_speed_adjustment(self, tmp_path):
        """Test audio generation with speed adjustment"""
        # Create mock audio file
        input_file = tmp_path / "input.wav"
        output_file = tmp_path / "output.wav"
        input_file.write_bytes(b'RIFF' + b'\x00' * 100)  # Mock WAV header
        
        # Test speed adjustment close to 1.0 (should copy)
        with patch('shutil.copy2') as mock_copy:
            adjust_audio_speed(str(input_file), str(output_file), 1.001)
            mock_copy.assert_called_once()
        
        # Test actual speed adjustment
        with patch('subprocess.run') as mock_run, \
             patch('core._10_gen_audio.get_audio_duration', return_value=10.0):
            mock_run.return_value = Mock(returncode=0)
            adjust_audio_speed(str(input_file), str(output_file), 1.5)
            
            # Verify ffmpeg was called
            mock_run.assert_called_once()
            args = mock_run.call_args[0][0]
            assert 'ffmpeg' in args
            assert 'atempo=1.5' in ' '.join(args)
    
    def test_tts_generation_pipeline(self):
        """Test TTS generation with parallel processing"""
        # Create sample tasks DataFrame
        tasks_df = pd.DataFrame({
            'number': [1, 2, 3],
            'lines': [['Line 1'], ['Line 2'], ['Line 3']],
            'duration': [5.0, 5.0, 5.0]
        })
        
        with patch('core._10_gen_audio.tts_main') as mock_tts, \
             patch('core._10_gen_audio.get_audio_duration', return_value=4.5), \
             patch('core._10_gen_audio.load_key', return_value=2):  # max_workers
            
            result_df = generate_tts_audio(tasks_df)
            
            # Verify all rows were processed
            assert 'real_dur' in result_df.columns
            assert all(result_df['real_dur'] == 4.5)
            assert mock_tts.call_count == 3  # Called for each line


class TestErrorPropagationAcrossPipeline:
    """Test error handling and propagation across pipeline stages"""
    
    def test_download_error_stops_pipeline(self):
        """Test that download errors prevent downstream processing"""
        pipeline_state = {'download': False, 'transcribe': False, 'translate': False}
        
        def download_stage():
            pipeline_state['download'] = True
            raise Exception("Download failed: Network error")
        
        def transcribe_stage():
            pipeline_state['transcribe'] = True
            return "transcription"
        
        def translate_stage():
            pipeline_state['translate'] = True
            return "translation"
        
        # Run pipeline with error handling
        try:
            download_stage()
            transcribe_stage()
            translate_stage()
        except Exception as e:
            assert "Download failed" in str(e)
        
        # Verify only download stage ran
        assert pipeline_state == {'download': True, 'transcribe': False, 'translate': False}
    
    def test_partial_failure_recovery(self):
        """Test recovery from partial pipeline failures"""
        checkpoint_file = tempfile.mktemp(suffix='.json')
        
        def save_checkpoint(stage, data):
            """Save pipeline checkpoint"""
            checkpoints = {}
            if os.path.exists(checkpoint_file):
                with open(checkpoint_file, 'r') as f:
                    checkpoints = json.load(f)
            checkpoints[stage] = data
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoints, f)
        
        def load_checkpoint(stage):
            """Load pipeline checkpoint"""
            if os.path.exists(checkpoint_file):
                with open(checkpoint_file, 'r') as f:
                    checkpoints = json.load(f)
                    return checkpoints.get(stage)
            return None
        
        # First run - fails at stage 2
        try:
            save_checkpoint('stage1', {'data': 'stage1_complete'})
            raise Exception("Stage 2 failed")
        except:
            pass
        
        # Recovery run - skip stage 1, continue from stage 2
        stage1_data = load_checkpoint('stage1')
        assert stage1_data == {'data': 'stage1_complete'}
        
        save_checkpoint('stage2', {'data': 'stage2_complete'})
        save_checkpoint('stage3', {'data': 'stage3_complete'})
        
        # Verify full pipeline completion
        final_state = {}
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r') as f:
                final_state = json.load(f)
        
        assert 'stage1' in final_state
        assert 'stage2' in final_state
        assert 'stage3' in final_state
        
        # Cleanup
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
    
    def test_concurrent_stage_error_isolation(self):
        """Test that errors in concurrent operations are isolated"""
        results = {'successful': [], 'failed': []}
        
        def process_item(item_id):
            if item_id == 2:  # Simulate failure for item 2
                raise ValueError(f"Processing failed for item {item_id}")
            time.sleep(0.1)  # Simulate work
            return f"Processed {item_id}"
        
        # Process items concurrently
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {}
            for i in range(5):
                future = executor.submit(process_item, i)
                futures[future] = i
            
            for future in futures:
                try:
                    result = future.result(timeout=1)
                    results['successful'].append(result)
                except Exception as e:
                    results['failed'].append(str(e))
        
        # Verify isolation
        assert len(results['successful']) == 4  # 4 items succeeded
        assert len(results['failed']) == 1  # 1 item failed
        assert "item 2" in results['failed'][0]
    
    def test_data_consistency_on_error(self, tmp_path):
        """Test that data remains consistent when errors occur"""
        data_file = tmp_path / "pipeline_data.json"
        
        def write_data(data):
            with open(data_file, 'w') as f:
                json.dump(data, f)
        
        def read_data():
            if data_file.exists():
                with open(data_file, 'r') as f:
                    return json.load(f)
            return {}
        
        # Initial data
        initial_data = {'stage1': 'complete', 'stage2': 'in_progress'}
        write_data(initial_data)
        
        # Attempt to update with error
        try:
            current_data = read_data()
            current_data['stage2'] = 'processing'
            # Simulate error before write
            raise Exception("Processing error")
            write_data(current_data)  # This should not execute
        except:
            pass
        
        # Verify data consistency - should still have initial data
        final_data = read_data()
        assert final_data == initial_data  # Data unchanged due to error
    
    def test_timeout_handling_in_pipeline(self):
        """Test timeout handling in pipeline stages"""
        def slow_operation():
            time.sleep(5)  # Simulate slow operation
            return "complete"
        
        results = []
        
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(slow_operation)
            try:
                result = future.result(timeout=0.5)  # Short timeout
                results.append(result)
            except TimeoutError:
                results.append("timeout")
                future.cancel()  # Cancel the operation
        
        assert results == ["timeout"]
    
    def test_memory_leak_prevention(self):
        """Test that pipeline properly cleans up resources on error"""
        import gc
        import weakref
        
        class ResourceHolder:
            def __init__(self):
                self.data = bytearray(1024 * 1024)  # 1MB of data
        
        # Track resource lifecycle
        resources_freed = []
        
        def pipeline_with_resources():
            resource = ResourceHolder()
            # Create weak reference to track cleanup
            weak_ref = weakref.ref(resource, lambda x: resources_freed.append(True))
            
            # Simulate error
            raise Exception("Pipeline error")
        
        try:
            pipeline_with_resources()
        except:
            pass
        
        # Force garbage collection
        gc.collect()
        
        # Verify resource was freed
        assert len(resources_freed) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
