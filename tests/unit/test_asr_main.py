"""
Comprehensive test suite for ASR main orchestration module (_2_asr.py).
Tests the main transcription flow, memory management, and ASR provider coordination.
"""

import pytest
import os
import gc
import tempfile
from unittest.mock import Mock, patch, MagicMock, call
from pathlib import Path
import pandas as pd
import psutil

# Test imports for ASR main module
try:
    from core._2_asr import (
        check_memory_usage,
        monitor_memory_and_warn,
        transcribe
    )
    from core.utils.models import _2_CLEANED_CHUNKS, _RAW_AUDIO_FILE, _VOCAL_AUDIO_FILE, _BACKGROUND_AUDIO_FILE, _AUDIO_DIR
except ImportError as e:
    pytest.skip(f"ASR main module not available: {e}", allow_module_level=True)


class TestMemoryManagement:
    """Test memory monitoring and management functionality."""
    
    @patch('core._2_asr.psutil.virtual_memory')
    def test_check_memory_usage_success(self, mock_virtual_memory):
        """Test successful memory usage check."""
        mock_memory = Mock()
        mock_memory.available = 4 * 1024 * 1024 * 1024  # 4GB
        mock_memory.percent = 75.0
        mock_virtual_memory.return_value = mock_memory
        
        result = check_memory_usage()
        
        assert result['available_mb'] == 4096.0
        assert result['used_percent'] == 75.0
        assert result['available_percent'] == 25.0
    
    @patch('core._2_asr.psutil.virtual_memory')
    def test_check_memory_usage_exception(self, mock_virtual_memory):
        """Test memory usage check with exception."""
        mock_virtual_memory.side_effect = Exception("Memory check failed")
        
        result = check_memory_usage()
        
        assert result['available_mb'] == -1
        assert result['used_percent'] == -1
        assert result['available_percent'] == -1
    
    @patch('core._2_asr.check_memory_usage')
    @patch('core._2_asr.rprint')
    def test_monitor_memory_and_warn_low_memory(self, mock_rprint, mock_check_memory):
        """Test memory warning for low available memory."""
        mock_check_memory.return_value = {
            'available_mb': 1024.0,  # Less than 2048 required
            'used_percent': 60.0,
            'available_percent': 40.0
        }
        
        monitor_memory_and_warn("test stage", min_required_mb=2048)
        
        # Should warn about low memory
        warning_calls = [call for call in mock_rprint.call_args_list if '[red]Warning: Low memory' in str(call)]
        assert len(warning_calls) > 0
    
    @patch('core._2_asr.check_memory_usage')
    @patch('core._2_asr.rprint')
    def test_monitor_memory_and_warn_high_usage(self, mock_rprint, mock_check_memory):
        """Test memory warning for high usage percentage."""
        mock_check_memory.return_value = {
            'available_mb': 3072.0,  # Sufficient
            'used_percent': 90.0,    # High usage > 85%
            'available_percent': 10.0
        }
        
        monitor_memory_and_warn("test stage", min_required_mb=2048)
        
        # Should warn about high memory usage
        warning_calls = [call for call in mock_rprint.call_args_list if '[yellow]High memory usage' in str(call)]
        assert len(warning_calls) > 0
    
    @patch('core._2_asr.check_memory_usage')
    @patch('core._2_asr.rprint')
    def test_monitor_memory_and_warn_good_status(self, mock_rprint, mock_check_memory):
        """Test memory status with good conditions."""
        mock_check_memory.return_value = {
            'available_mb': 3072.0,  # Sufficient
            'used_percent': 60.0,    # Normal usage < 85%
            'available_percent': 40.0
        }
        
        monitor_memory_and_warn("test stage", min_required_mb=2048)
        
        # Should show green status
        status_calls = [call for call in mock_rprint.call_args_list if '[green]Memory status' in str(call)]
        assert len(status_calls) > 0
    
    @patch('core._2_asr.check_memory_usage')
    def test_monitor_memory_and_warn_invalid_memory(self, mock_check_memory):
        """Test memory monitoring with invalid memory info."""
        mock_check_memory.return_value = {
            'available_mb': -1,
            'used_percent': -1,
            'available_percent': -1
        }
        
        # Should not raise exception with invalid memory info
        monitor_memory_and_warn("test stage")


class TestTranscriptionWorkflow:
    """Test the main transcription workflow and orchestration."""
    
    @pytest.fixture
    def mock_dependencies(self):
        """Mock all external dependencies for transcription workflow."""
        with patch('core._2_asr.monitor_memory_and_warn') as mock_monitor, \
             patch('core._2_asr.find_video_files') as mock_find_video, \
             patch('core._2_asr.convert_video_to_audio') as mock_convert, \
             patch('core._2_asr.load_key') as mock_load_key, \
             patch('core._2_asr.demucs_audio') as mock_demucs, \
             patch('core._2_asr.normalize_audio_volume') as mock_normalize, \
             patch('core._2_asr.split_audio') as mock_split, \
             patch('core._2_asr.process_transcription') as mock_process, \
             patch('core._2_asr.save_results') as mock_save, \
             patch('core._2_asr.gc.collect') as mock_gc, \
             patch('core._2_asr.check_file_exists') as mock_check_file:
            
            # Configure mocks
            mock_find_video.return_value = '/tmp/video.mp4'
            mock_load_key.side_effect = lambda key: {
                'demucs': False,
                'whisper.runtime': 'local'
            }.get(key, 'default_value')
            mock_split.return_value = [(0.0, 30.0), (30.0, 60.0)]
            mock_process.return_value = pd.DataFrame({
                'text': ['Hello', 'World'],
                'start': [0.0, 1.0],
                'end': [1.0, 2.0]
            })
            
            # Make the decorator pass through
            mock_check_file.return_value = lambda func: func
            
            yield {
                'monitor': mock_monitor,
                'find_video': mock_find_video,
                'convert': mock_convert,
                'load_key': mock_load_key,
                'demucs': mock_demucs,
                'normalize': mock_normalize,
                'split': mock_split,
                'process': mock_process,
                'save': mock_save,
                'gc': mock_gc
            }
    
    def test_transcribe_workflow_local_whisper(self, mock_dependencies):
        """Test complete transcription workflow with local WhisperX."""
        mock_dependencies['load_key'].side_effect = lambda key: {
            'demucs': False,
            'whisper.runtime': 'local'
        }.get(key, 'default_value')
        
        # Mock the local transcription function
        with patch('core.asr_backend.whisperX_local.transcribe_audio') as mock_transcribe:
            mock_transcribe.return_value = {
                'segments': [
                    {'start': 0.0, 'end': 1.0, 'words': [{'word': 'Hello', 'start': 0.0, 'end': 1.0}]}
                ]
            }
            
            transcribe()
            
            # Verify workflow steps
            mock_dependencies['find_video'].assert_called_once()
            mock_dependencies['convert'].assert_called_once()
            mock_dependencies['split'].assert_called_once()
            mock_transcribe.assert_called()
            mock_dependencies['process'].assert_called_once()
            mock_dependencies['save'].assert_called_once()
    
    def test_transcribe_workflow_cloud_whisper(self, mock_dependencies):
        """Test transcription workflow with 302.ai cloud API."""
        mock_dependencies['load_key'].side_effect = lambda key: {
            'demucs': False,
            'whisper.runtime': 'cloud'
        }.get(key, 'default_value')
        
        # Mock the cloud transcription function
        with patch('core.asr_backend.whisperX_302.transcribe_audio_302') as mock_transcribe:
            mock_transcribe.return_value = {
                'segments': [
                    {'start': 0.0, 'end': 1.0, 'words': [{'word': 'Hello', 'start': 0.0, 'end': 1.0}]}
                ]
            }
            
            transcribe()
            
            # Verify cloud API was used
            mock_transcribe.assert_called()
    
    def test_transcribe_workflow_elevenlabs(self, mock_dependencies):
        """Test transcription workflow with ElevenLabs API."""
        mock_dependencies['load_key'].side_effect = lambda key: {
            'demucs': False,
            'whisper.runtime': 'elevenlabs'
        }.get(key, 'default_value')
        
        # Mock the ElevenLabs transcription function
        with patch('core.asr_backend.elevenlabs_asr.transcribe_audio_elevenlabs') as mock_transcribe:
            mock_transcribe.return_value = {
                'segments': [
                    {'start': 0.0, 'end': 1.0, 'words': [{'word': 'Hello', 'start': 0.0, 'end': 1.0}]}
                ]
            }
            
            transcribe()
            
            # Verify ElevenLabs API was used
            mock_transcribe.assert_called()
    
    def test_transcribe_with_demucs_enabled(self, mock_dependencies):
        """Test transcription workflow with Demucs vocal separation enabled."""
        mock_dependencies['load_key'].side_effect = lambda key: {
            'demucs': True,
            'whisper.runtime': 'local'
        }.get(key, 'default_value')
        
        # Mock that demucs_audio is available
        with patch('core._2_asr.demucs_audio', Mock()) as mock_demucs_func:
            with patch('core.asr_backend.whisperX_local.transcribe_audio') as mock_transcribe:
                mock_transcribe.return_value = {'segments': []}
                
                transcribe()
                
                # Verify Demucs was called
                mock_demucs_func.assert_called_once()
                mock_dependencies['normalize'].assert_called_once()
    
    def test_transcribe_with_demucs_unavailable(self, mock_dependencies):
        """Test transcription when Demucs is enabled but not available."""
        mock_dependencies['load_key'].side_effect = lambda key: {
            'demucs': True,
            'whisper.runtime': 'local'
        }.get(key, 'default_value')
        
        # Mock that demucs_audio is None (not available)
        with patch('core._2_asr.demucs_audio', None):
            with patch('core.asr_backend.whisperX_local.transcribe_audio') as mock_transcribe:
                with patch('core._2_asr.rprint') as mock_rprint:
                    mock_transcribe.return_value = {'segments': []}
                    
                    transcribe()
                    
                    # Verify warning was printed
                    warning_calls = [call for call in mock_rprint.call_args_list 
                                   if 'Demucs is enabled in config but not available' in str(call)]
                    assert len(warning_calls) > 0
    
    def test_transcribe_memory_monitoring_calls(self, mock_dependencies):
        """Test that memory monitoring is called at appropriate stages."""
        with patch('core.asr_backend.whisperX_local.transcribe_audio') as mock_transcribe:
            mock_transcribe.return_value = {'segments': []}
            
            transcribe()
            
            # Verify memory monitoring was called at various stages
            monitor_calls = mock_dependencies['monitor'].call_args_list
            call_stages = [call[0][0] for call in monitor_calls]
            
            assert "transcription start" in call_stages
            assert "after video conversion" in call_stages
            assert "after audio segmentation" in call_stages
            assert "transcription complete" in call_stages
    
    def test_transcribe_memory_cleanup(self, mock_dependencies):
        """Test that memory cleanup (gc.collect) is called appropriately."""
        mock_dependencies['split'].return_value = [(i*30, (i+1)*30) for i in range(15)]  # 15 segments
        
        with patch('core.asr_backend.whisperX_local.transcribe_audio') as mock_transcribe:
            mock_transcribe.return_value = {'segments': []}
            
            transcribe()
            
            # Verify gc.collect was called
            assert mock_dependencies['gc'].call_count >= 1
    
    def test_transcribe_segmented_processing(self, mock_dependencies):
        """Test that audio segments are processed individually."""
        # Create multiple segments
        mock_dependencies['split'].return_value = [(0.0, 30.0), (30.0, 60.0), (60.0, 90.0)]
        
        with patch('core.asr_backend.whisperX_local.transcribe_audio') as mock_transcribe:
            mock_transcribe.return_value = {
                'segments': [{'start': 0.0, 'end': 1.0, 'words': []}]
            }
            
            transcribe()
            
            # Verify transcription was called for each segment
            assert mock_transcribe.call_count == 3
    
    def test_transcribe_results_combination(self, mock_dependencies):
        """Test that results from multiple segments are combined properly."""
        mock_dependencies['split'].return_value = [(0.0, 30.0), (30.0, 60.0)]
        
        with patch('core.asr_backend.whisperX_local.transcribe_audio') as mock_transcribe:
            # Return different results for each segment
            mock_transcribe.side_effect = [
                {'segments': [{'start': 0.0, 'end': 1.0, 'words': [{'word': 'First'}]}]},
                {'segments': [{'start': 30.0, 'end': 31.0, 'words': [{'word': 'Second'}]}]}
            ]
            
            transcribe()
            
            # Verify process_transcription was called with combined results
            call_args = mock_dependencies['process'].call_args[0][0]
            assert len(call_args['segments']) == 2
    
    def test_transcribe_error_handling(self, mock_dependencies):
        """Test error handling in transcription workflow."""
        # Make convert_video_to_audio raise an exception
        mock_dependencies['convert'].side_effect = Exception("Video conversion failed")
        
        with pytest.raises(Exception, match="Video conversion failed"):
            transcribe()


class TestASRProviderSelection:
    """Test ASR provider selection logic."""
    
    @pytest.fixture
    def base_mocks(self):
        """Base mocks for provider selection tests."""
        with patch('core._2_asr.monitor_memory_and_warn'), \
             patch('core._2_asr.find_video_files', return_value='/tmp/video.mp4'), \
             patch('core._2_asr.convert_video_to_audio'), \
             patch('core._2_asr.split_audio', return_value=[(0.0, 30.0)]), \
             patch('core._2_asr.process_transcription', return_value=pd.DataFrame()), \
             patch('core._2_asr.save_results'), \
             patch('core._2_asr.check_file_exists', return_value=lambda func: func):
            yield
    
    @patch('core._2_asr.load_key')
    def test_local_provider_import_and_usage(self, mock_load_key, base_mocks):
        """Test that local WhisperX provider is imported and used correctly."""
        mock_load_key.side_effect = lambda key: {
            'demucs': False,
            'whisper.runtime': 'local'
        }.get(key, 'default_value')
        
        with patch('core.asr_backend.whisperX_local.transcribe_audio') as mock_transcribe:
            mock_transcribe.return_value = {'segments': []}
            
            transcribe()
            
            mock_transcribe.assert_called()
            call_args = mock_transcribe.call_args[0]
            assert len(call_args) == 4  # raw_audio, vocal_audio, start, end
    
    @patch('core._2_asr.load_key')
    def test_cloud_provider_import_and_usage(self, mock_load_key, base_mocks):
        """Test that 302.ai cloud provider is imported and used correctly."""
        mock_load_key.side_effect = lambda key: {
            'demucs': False,
            'whisper.runtime': 'cloud'
        }.get(key, 'default_value')
        
        with patch('core.asr_backend.whisperX_302.transcribe_audio_302') as mock_transcribe:
            mock_transcribe.return_value = {'segments': []}
            
            transcribe()
            
            mock_transcribe.assert_called()
    
    @patch('core._2_asr.load_key')
    def test_elevenlabs_provider_import_and_usage(self, mock_load_key, base_mocks):
        """Test that ElevenLabs provider is imported and used correctly."""
        mock_load_key.side_effect = lambda key: {
            'demucs': False,
            'whisper.runtime': 'elevenlabs'
        }.get(key, 'default_value')
        
        with patch('core.asr_backend.elevenlabs_asr.transcribe_audio_elevenlabs') as mock_transcribe:
            mock_transcribe.return_value = {'segments': []}
            
            transcribe()
            
            mock_transcribe.assert_called()


class TestTranscriptionMemoryOptimization:
    """Test memory optimization features in transcription."""
    
    @patch('core._2_asr.load_key')
    def test_memory_monitoring_frequencies(self, mock_load_key):
        """Test that memory monitoring occurs at specified intervals."""
        mock_load_key.side_effect = lambda key: {
            'demucs': False,
            'whisper.runtime': 'local'
        }.get(key, 'default_value')
        
        # Create many segments to trigger interval-based memory monitoring
        segments = [(i*10, (i+1)*10) for i in range(25)]  # 25 segments
        
        with patch('core._2_asr.monitor_memory_and_warn') as mock_monitor, \
             patch('core._2_asr.find_video_files', return_value='/tmp/video.mp4'), \
             patch('core._2_asr.convert_video_to_audio'), \
             patch('core._2_asr.split_audio', return_value=segments), \
             patch('core._2_asr.process_transcription', return_value=pd.DataFrame()), \
             patch('core._2_asr.save_results'), \
             patch('core._2_asr.gc.collect') as mock_gc, \
             patch('core._2_asr.check_file_exists', return_value=lambda func: func), \
             patch('core.asr_backend.whisperX_local.transcribe_audio', return_value={'segments': []}):
            
            transcribe()
            
            # Check that memory monitoring was called for segment intervals
            # Should be called for segments 0, 10, 20 (every 10th segment)
            segment_monitor_calls = [call for call in mock_monitor.call_args_list 
                                   if 'transcription segment' in str(call)]
            assert len(segment_monitor_calls) >= 2  # At least 2 interval checks
            
            # Verify gc.collect was called during processing
            assert mock_gc.call_count >= 2  # Called during intervals + final cleanup
    
    def test_demucs_memory_requirements(self):
        """Test that Demucs operations use appropriate memory settings."""
        with patch('core._2_asr.load_key', return_value=True), \
             patch('core._2_asr.demucs_audio', Mock()) as mock_demucs, \
             patch('core._2_asr.monitor_memory_and_warn') as mock_monitor, \
             patch('core._2_asr.find_video_files', return_value='/tmp/video.mp4'), \
             patch('core._2_asr.convert_video_to_audio'), \
             patch('core._2_asr.normalize_audio_volume'), \
             patch('core._2_asr.split_audio', return_value=[(0.0, 30.0)]), \
             patch('core._2_asr.process_transcription', return_value=pd.DataFrame()), \
             patch('core._2_asr.save_results'), \
             patch('core._2_asr.gc.collect'), \
             patch('core._2_asr.check_file_exists', return_value=lambda func: func), \
             patch('core.asr_backend.whisperX_local.transcribe_audio', return_value={'segments': []}):
            
            transcribe()
            
            # Verify that Demucs was called with higher memory requirement
            demucs_calls = [call for call in mock_monitor.call_args_list 
                           if 'before Demucs' in str(call)]
            assert len(demucs_calls) > 0
            
            # Check that the memory requirement for Demucs was higher (4096MB)
            demucs_call = demucs_calls[0]
            assert demucs_call[0][1] == 4096  # min_required_mb parameter


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
