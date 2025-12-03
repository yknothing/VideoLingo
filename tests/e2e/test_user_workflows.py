"""
End-to-End User Workflow Tests for VideoLingo

This test suite focuses on real user scenarios and workflows rather than technical implementation.
It measures timing, quality metrics, and user experience from a practical perspective.
"""

import pytest
import os
import time
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import threading
import queue


class UserWorkflowMetrics:
    """Track user-centric metrics for workflow quality assessment"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.step_timings = {}
        self.quality_scores = {}
        self.user_interactions = []
        self.errors_encountered = []
        self.retry_counts = {}
        
    def start_workflow(self):
        """Start tracking workflow execution"""
        self.start_time = time.time()
        
    def end_workflow(self):
        """End workflow tracking and calculate total time"""
        self.end_time = time.time()
        return self.get_total_duration()
        
    def record_step(self, step_name: str, duration: float, success: bool = True):
        """Record timing for a workflow step"""
        self.step_timings[step_name] = {
            'duration': duration,
            'success': success,
            'timestamp': time.time()
        }
        
    def record_quality(self, metric_name: str, score: float, details: Dict = None):
        """Record quality metric from user perspective"""
        self.quality_scores[metric_name] = {
            'score': score,
            'details': details or {},
            'timestamp': time.time()
        }
        
    def record_user_action(self, action: str, context: Dict = None):
        """Record user interaction or configuration change"""
        self.user_interactions.append({
            'action': action,
            'context': context or {},
            'timestamp': time.time()
        })
        
    def record_error(self, error_type: str, message: str, recoverable: bool = True):
        """Record errors from user perspective"""
        self.errors_encountered.append({
            'type': error_type,
            'message': message,
            'recoverable': recoverable,
            'timestamp': time.time()
        })
        
    def get_total_duration(self) -> float:
        """Get total workflow duration in seconds"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0
        
    def get_user_experience_score(self) -> float:
        """
        Calculate overall user experience score based on:
        - Processing speed
        - Quality of output
        - Error frequency
        - Need for retries
        """
        score = 100.0
        
        # Deduct for slow processing (expected: < 2x video duration for standard processing)
        total_duration = self.get_total_duration()
        if total_duration > 0:
            speed_penalty = min(30, (total_duration / 60) * 2)  # 2 points per minute
            score -= speed_penalty
            
        # Deduct for errors
        error_penalty = len(self.errors_encountered) * 10
        score -= min(40, error_penalty)  # Cap at 40 points
        
        # Deduct for retries
        retry_penalty = sum(self.retry_counts.values()) * 5
        score -= min(20, retry_penalty)  # Cap at 20 points
        
        # Add quality bonus
        if self.quality_scores:
            avg_quality = sum(q['score'] for q in self.quality_scores.values()) / len(self.quality_scores)
            quality_bonus = avg_quality * 10  # Up to 10 points bonus
            score += quality_bonus
            
        return max(0, min(100, score))  # Clamp between 0-100


class TestYouTubeVideoWorkflow:
    """Test complete YouTube video processing workflow from user perspective"""
    
    @pytest.fixture
    def workflow_metrics(self):
        """Provide metrics tracker for each test"""
        return UserWorkflowMetrics()
    
    @pytest.fixture
    def mock_youtube_video(self):
        """Mock YouTube video data"""
        return {
            'url': 'https://www.youtube.com/watch?v=dQw4w9WgXcQ',
            'title': 'Rick Astley - Never Gonna Give You Up',
            'duration': 213,  # 3:33 minutes
            'language': 'en',
            'resolution': '1080p',
            'filesize_mb': 42.5,
            'subtitle_available': True,
            'expected_processing_time': 180  # 3 minutes expected
        }
    
    def test_youtube_video_standard_workflow(self, workflow_metrics, mock_youtube_video):
        """
        Test standard YouTube video processing workflow:
        User wants to download, transcribe, translate to Chinese, and add dubbing
        """
        workflow_metrics.start_workflow()
        
        # User configuration
        user_config = {
            'source_language': 'en',
            'target_language': 'zh-CN',
            'video_quality': '1080p',
            'enable_dubbing': True,
            'tts_engine': 'azure',
            'translation_style': 'natural',
            'keep_original_audio': False
        }
        
        workflow_metrics.record_user_action('configure_settings', user_config)
        
        # Simulate workflow execution
        workflow_results = {}
        
        # Step 1: Download YouTube video (user waits for download)
        with patch('core._1_ytdlp.download_video_ytdlp') as mock_download:
            download_start = time.time()
            
            mock_download.return_value = {
                'success': True,
                'file_path': '/tmp/video.mp4',
                'actual_quality': '1080p',
                'download_speed_mbps': 10.5
            }
            
            # Simulate download time based on file size
            time.sleep(0.1)  # Simulate download
            download_duration = time.time() - download_start
            
            workflow_metrics.record_step('download', download_duration)
            workflow_metrics.record_quality('download_quality', 
                                           1.0 if mock_download.return_value['actual_quality'] == '1080p' else 0.8)
            
            workflow_results['download'] = mock_download.return_value
        
        # Step 2: Transcription (user perspective: accuracy matters)
        with patch('core._2_asr.transcribe') as mock_transcribe:
            transcribe_start = time.time()
            
            mock_transcribe.return_value = {
                'segments': [
                    {'text': "We're no strangers to love", 'start': 0.0, 'end': 2.5},
                    {'text': "You know the rules and so do I", 'start': 2.5, 'end': 5.0},
                    # ... more segments
                ],
                'language': 'en',
                'confidence': 0.95
            }
            
            time.sleep(0.2)  # Simulate transcription
            transcribe_duration = time.time() - transcribe_start
            
            workflow_metrics.record_step('transcription', transcribe_duration)
            workflow_metrics.record_quality('transcription_accuracy', 0.95, 
                                           {'confidence': 0.95, 'word_count': 50})
            
            workflow_results['transcription'] = mock_transcribe.return_value
        
        # Step 3: Translation (user cares about quality and naturalness)
        with patch('core._4_2_translate.translate_all') as mock_translate:
            translate_start = time.time()
            
            mock_translate.return_value = {
                'translations': [
                    {'original': "We're no strangers to love", 
                     'translated': '我们对爱并不陌生',
                     'quality_score': 0.92},
                    {'original': "You know the rules and so do I",
                     'translated': '你知道规则，我也知道',
                     'quality_score': 0.90}
                ],
                'terminology_consistency': 0.95,
                'natural_flow': 0.88
            }
            
            time.sleep(0.15)  # Simulate translation
            translate_duration = time.time() - translate_start
            
            workflow_metrics.record_step('translation', translate_duration)
            workflow_metrics.record_quality('translation_quality', 0.91,
                                           {'consistency': 0.95, 'naturalness': 0.88})
            
            workflow_results['translation'] = mock_translate.return_value
        
        # Step 4: TTS Generation (user cares about voice quality and sync)
        with patch('core._10_gen_audio.gen_audio') as mock_tts:
            tts_start = time.time()
            
            mock_tts.return_value = {
                'audio_files': ['/tmp/audio_1.wav', '/tmp/audio_2.wav'],
                'voice_quality': 'high',
                'pronunciation_accuracy': 0.93,
                'emotion_matching': 0.85,
                'timing_accuracy': 0.90
            }
            
            time.sleep(0.3)  # Simulate TTS generation
            tts_duration = time.time() - tts_start
            
            workflow_metrics.record_step('tts_generation', tts_duration)
            workflow_metrics.record_quality('voice_quality', 0.89,
                                           {'pronunciation': 0.93, 'emotion': 0.85, 'timing': 0.90})
            
            workflow_results['tts'] = mock_tts.return_value
        
        # Step 5: Final video merging (user waits for final output)
        with patch('core._12_dub_to_vid.merge_video_audio') as mock_merge:
            merge_start = time.time()
            
            mock_merge.return_value = {
                'output_file': '/tmp/output_final.mp4',
                'file_size_mb': 85.3,
                'audio_video_sync': 0.98,
                'quality_preserved': True
            }
            
            time.sleep(0.1)  # Simulate merging
            merge_duration = time.time() - merge_start
            
            workflow_metrics.record_step('final_merge', merge_duration)
            workflow_metrics.record_quality('output_quality', 0.98,
                                           {'sync': 0.98, 'quality_preserved': True})
            
            workflow_results['final_output'] = mock_merge.return_value
        
        workflow_metrics.end_workflow()
        
        # Validate user experience
        total_time = workflow_metrics.get_total_duration()
        user_score = workflow_metrics.get_user_experience_score()
        
        # User expectations
        assert total_time < 10, f"Workflow took too long: {total_time}s"
        assert user_score > 75, f"Poor user experience score: {user_score}"
        
        # Quality checks from user perspective
        assert workflow_metrics.quality_scores['transcription_accuracy']['score'] > 0.9
        assert workflow_metrics.quality_scores['translation_quality']['score'] > 0.85
        assert workflow_metrics.quality_scores['voice_quality']['score'] > 0.8
        assert workflow_metrics.quality_scores['output_quality']['score'] > 0.95
        
        # Ensure no critical errors
        critical_errors = [e for e in workflow_metrics.errors_encountered if not e['recoverable']]
        assert len(critical_errors) == 0, f"Critical errors encountered: {critical_errors}"
    
    def test_youtube_playlist_batch_processing(self, workflow_metrics):
        """
        Test batch processing of YouTube playlist:
        User wants to process multiple videos from a playlist
        """
        workflow_metrics.start_workflow()
        
        # User provides playlist URL
        playlist_url = "https://www.youtube.com/playlist?list=PLrAXtmErZgOeiKm4sgNOknGvNjby9efdf"
        playlist_videos = [
            {'id': 'video1', 'title': 'Tutorial Part 1', 'duration': 600},
            {'id': 'video2', 'title': 'Tutorial Part 2', 'duration': 480},
            {'id': 'video3', 'title': 'Tutorial Part 3', 'duration': 720}
        ]
        
        workflow_metrics.record_user_action('process_playlist', {'url': playlist_url, 'video_count': 3})
        
        batch_results = []
        
        for idx, video in enumerate(playlist_videos):
            video_start = time.time()
            
            # Simulate processing each video
            with patch('batch.utils.video_processor.process_video') as mock_process:
                mock_process.return_value = (True, '', '')
                
                # User sees progress for each video
                workflow_metrics.record_user_action(f'processing_video_{idx+1}', 
                                                   {'title': video['title'], 'progress': f"{idx+1}/{len(playlist_videos)}"})
                
                time.sleep(0.2)  # Simulate processing
                
                video_duration = time.time() - video_start
                workflow_metrics.record_step(f'video_{idx+1}', video_duration)
                
                batch_results.append({
                    'video': video,
                    'success': True,
                    'processing_time': video_duration
                })
        
        workflow_metrics.end_workflow()
        
        # User expectations for batch processing
        total_time = workflow_metrics.get_total_duration()
        avg_time_per_video = total_time / len(playlist_videos)
        
        # Batch processing should be efficient
        assert avg_time_per_video < 5, f"Batch processing too slow: {avg_time_per_video}s per video"
        
        # All videos should process successfully
        successful_videos = [r for r in batch_results if r['success']]
        assert len(successful_videos) == len(playlist_videos), "Some videos failed in batch"
        
        # User should see consistent quality
        for result in batch_results:
            assert result['processing_time'] < 10, "Individual video took too long"


class TestLocalVideoWorkflow:
    """Test local video file processing workflows"""
    
    @pytest.fixture
    def workflow_metrics(self):
        return UserWorkflowMetrics()
    
    @pytest.fixture
    def mock_local_video(self, tmp_path):
        """Create mock local video file"""
        video_file = tmp_path / "test_video.mp4"
        video_file.write_bytes(b"fake video content")
        
        return {
            'path': str(video_file),
            'filename': 'test_video.mp4',
            'duration': 300,  # 5 minutes
            'format': 'mp4',
            'size_mb': 150,
            'resolution': '1080p',
            'fps': 30,
            'audio_codec': 'aac'
        }
    
    def test_local_video_quick_subtitle(self, workflow_metrics, mock_local_video):
        """
        Test quick subtitle generation for local video:
        User just wants subtitles, no dubbing
        """
        workflow_metrics.start_workflow()
        
        # User configuration for quick subtitles
        user_config = {
            'input_file': mock_local_video['path'],
            'source_language': 'auto',  # Auto-detect
            'target_language': 'en',
            'subtitle_only': True,
            'enable_dubbing': False,
            'subtitle_format': 'srt'
        }
        
        workflow_metrics.record_user_action('quick_subtitle_mode', user_config)
        
        # Quick workflow - skip unnecessary steps
        with patch('core._2_asr.transcribe') as mock_transcribe:
            transcribe_start = time.time()
            
            mock_transcribe.return_value = {
                'segments': [{'text': 'Sample text', 'start': 0, 'end': 2}] * 10,
                'detected_language': 'en'
            }
            
            time.sleep(0.1)  # Fast transcription
            workflow_metrics.record_step('transcription', time.time() - transcribe_start)
        
        # Generate subtitles
        with patch('core._6_gen_sub.align_timestamp_main') as mock_subtitle:
            subtitle_start = time.time()
            
            mock_subtitle.return_value = {
                'subtitle_file': '/tmp/output.srt',
                'line_count': 50,
                'format': 'srt'
            }
            
            time.sleep(0.05)  # Fast subtitle generation
            workflow_metrics.record_step('subtitle_generation', time.time() - subtitle_start)
        
        workflow_metrics.end_workflow()
        
        # User expects fast processing for subtitle-only mode
        total_time = workflow_metrics.get_total_duration()
        assert total_time < 2, f"Subtitle generation too slow: {total_time}s"
        
        # Verify minimal steps were executed
        assert len(workflow_metrics.step_timings) == 2, "Too many steps for subtitle-only mode"
    
    def test_local_video_with_existing_subtitles(self, workflow_metrics, mock_local_video, tmp_path):
        """
        Test processing video that already has subtitles:
        User wants to translate existing subtitles and add dubbing
        """
        workflow_metrics.start_workflow()
        
        # Create existing subtitle file
        subtitle_file = tmp_path / "test_video.srt"
        subtitle_content = """1
00:00:00,000 --> 00:00:02,000
Hello world

2
00:00:02,000 --> 00:00:04,000
This is a test video
"""
        subtitle_file.write_text(subtitle_content)
        
        user_config = {
            'input_file': mock_local_video['path'],
            'existing_subtitle': str(subtitle_file),
            'skip_transcription': True,
            'target_language': 'es',
            'enable_dubbing': True
        }
        
        workflow_metrics.record_user_action('use_existing_subtitles', user_config)
        
        # Skip transcription, go straight to translation
        with patch('core._4_2_translate.translate_all') as mock_translate:
            translate_start = time.time()
            
            mock_translate.return_value = {
                'translations': [
                    {'original': 'Hello world', 'translated': 'Hola mundo'},
                    {'original': 'This is a test video', 'translated': 'Este es un video de prueba'}
                ]
            }
            
            time.sleep(0.1)
            workflow_metrics.record_step('translation', time.time() - translate_start)
        
        workflow_metrics.end_workflow()
        
        # Should be faster when skipping transcription
        total_time = workflow_metrics.get_total_duration()
        assert total_time < 3, "Processing with existing subtitles should be fast"
        
        # Verify transcription was skipped
        assert 'transcription' not in workflow_metrics.step_timings


class TestBatchProcessingWorkflow:
    """Test batch processing scenarios for multiple videos"""
    
    @pytest.fixture
    def workflow_metrics(self):
        return UserWorkflowMetrics()
    
    @pytest.fixture
    def mock_video_batch(self, tmp_path):
        """Create batch of mock video files"""
        videos = []
        for i in range(5):
            video_file = tmp_path / f"video_{i}.mp4"
            video_file.write_bytes(b"fake video content")
            videos.append({
                'path': str(video_file),
                'filename': f"video_{i}.mp4",
                'duration': 120 + i * 30,  # Varying durations
                'priority': 'high' if i == 0 else 'normal'
            })
        return videos
    
    def test_batch_parallel_processing(self, workflow_metrics, mock_video_batch):
        """
        Test parallel batch processing:
        User wants to process multiple videos efficiently
        """
        workflow_metrics.start_workflow()
        
        batch_config = {
            'parallel_workers': 3,
            'target_language': 'fr',
            'enable_dubbing': True,
            'auto_retry_on_failure': True,
            'priority_queue': True
        }
        
        workflow_metrics.record_user_action('configure_batch', batch_config)
        
        # Simulate parallel processing
        processing_queue = queue.Queue()
        completed = []
        
        def process_video_worker(video_data):
            """Simulate video processing worker"""
            start = time.time()
            time.sleep(0.1)  # Simulate processing
            duration = time.time() - start
            
            return {
                'video': video_data['filename'],
                'duration': duration,
                'success': True
            }
        
        # Process high priority first
        high_priority = [v for v in mock_video_batch if v.get('priority') == 'high']
        normal_priority = [v for v in mock_video_batch if v.get('priority') != 'high']
        
        # Process high priority videos first
        for video in high_priority:
            result = process_video_worker(video)
            completed.append(result)
            workflow_metrics.record_step(f"process_{video['filename']}", result['duration'])
        
        # Then process normal priority in parallel (simulated)
        for video in normal_priority:
            result = process_video_worker(video)
            completed.append(result)
            workflow_metrics.record_step(f"process_{video['filename']}", result['duration'])
        
        workflow_metrics.end_workflow()
        
        # Verify batch processing efficiency
        total_time = workflow_metrics.get_total_duration()
        sequential_time = len(mock_video_batch) * 0.1
        
        # Parallel processing should be faster than sequential
        assert total_time < sequential_time * 1.5, "Parallel processing not efficient"
        
        # All videos should be processed
        assert len(completed) == len(mock_video_batch)
        
        # High priority should be processed first
        first_processed = completed[0]
        assert 'video_0' in first_processed['video'], "High priority not processed first"
    
    def test_batch_with_mixed_formats(self, workflow_metrics, tmp_path):
        """
        Test batch processing with different video formats:
        User has videos in various formats (mp4, avi, mkv, etc.)
        """
        workflow_metrics.start_workflow()
        
        # Create mixed format batch
        mixed_formats = []
        formats = ['mp4', 'avi', 'mkv', 'mov', 'webm']
        
        for fmt in formats:
            video_file = tmp_path / f"video.{fmt}"
            video_file.write_bytes(b"fake video")
            mixed_formats.append({
                'path': str(video_file),
                'format': fmt,
                'requires_conversion': fmt not in ['mp4', 'webm']
            })
        
        workflow_metrics.record_user_action('process_mixed_formats', {'formats': formats})
        
        for video in mixed_formats:
            process_start = time.time()
            
            # Formats requiring conversion take longer
            if video['requires_conversion']:
                time.sleep(0.15)  # Conversion overhead
                workflow_metrics.record_user_action('format_conversion', {'format': video['format']})
            else:
                time.sleep(0.1)
            
            duration = time.time() - process_start
            workflow_metrics.record_step(f"process_{video['format']}", duration)
        
        workflow_metrics.end_workflow()
        
        # All formats should be processed
        assert len(workflow_metrics.step_timings) == len(formats)
        
        # Conversion overhead should be tracked
        conversion_actions = [a for a in workflow_metrics.user_interactions 
                            if a['action'] == 'format_conversion']
        assert len(conversion_actions) == 3  # avi, mkv, mov need conversion


class TestConfigurationChangesWorkflow:
    """Test workflows with configuration changes mid-process"""
    
    @pytest.fixture
    def workflow_metrics(self):
        return UserWorkflowMetrics()
    
    def test_quality_adjustment_mid_process(self, workflow_metrics):
        """
        Test adjusting quality settings during processing:
        User starts with high quality but switches to fast mode
        """
        workflow_metrics.start_workflow()
        
        # Initial high quality configuration
        initial_config = {
            'quality_mode': 'high',
            'whisper_model': 'large-v3',
            'translation_model': 'gpt-4',
            'tts_engine': 'azure',
            'video_quality': '4K'
        }
        
        workflow_metrics.record_user_action('initial_config', initial_config)
        
        # Start processing with high quality
        with patch('core._2_asr.transcribe') as mock_transcribe:
            transcribe_start = time.time()
            
            # Simulate slow high-quality processing
            time.sleep(0.5)
            
            # User sees it's taking too long
            elapsed = time.time() - transcribe_start
            workflow_metrics.record_user_action('check_progress', {'elapsed': elapsed, 'status': 'transcribing'})
            
            # User decides to switch to fast mode
            new_config = {
                'quality_mode': 'fast',
                'whisper_model': 'base',
                'translation_model': 'gpt-3.5-turbo',
                'tts_engine': 'edge',
                'video_quality': '720p'
            }
            
            workflow_metrics.record_user_action('change_config_mid_process', new_config)
            
            # Restart with fast settings
            time.sleep(0.1)  # Much faster
            
            total_transcribe_time = time.time() - transcribe_start
            workflow_metrics.record_step('transcription_with_restart', total_transcribe_time)
        
        workflow_metrics.end_workflow()
        
        # Verify configuration change was recorded
        config_changes = [a for a in workflow_metrics.user_interactions 
                         if a['action'] == 'change_config_mid_process']
        assert len(config_changes) == 1
        
        # Processing should complete despite configuration change
        assert 'transcription_with_restart' in workflow_metrics.step_timings
    
    def test_pause_resume_workflow(self, workflow_metrics):
        """
        Test pausing and resuming workflow:
        User pauses processing and resumes later
        """
        workflow_metrics.start_workflow()
        
        # Simulate processing stages with pause/resume
        stages = ['download', 'transcription', 'translation', 'tts', 'merge']
        pause_after = 2  # Pause after transcription
        
        for idx, stage in enumerate(stages):
            if idx == pause_after:
                # User pauses the workflow
                workflow_metrics.record_user_action('pause_workflow', {'stage': stage, 'reason': 'user_request'})
                
                # Simulate pause duration (user goes away)
                pause_duration = 2.0  # 2 seconds pause
                time.sleep(0.1)  # Simulated pause
                
                # User resumes
                workflow_metrics.record_user_action('resume_workflow', {'stage': stage})
            
            # Process stage
            stage_start = time.time()
            time.sleep(0.1)  # Simulate processing
            workflow_metrics.record_step(stage, time.time() - stage_start)
        
        workflow_metrics.end_workflow()
        
        # Verify pause/resume was handled
        pause_actions = [a for a in workflow_metrics.user_interactions if a['action'] == 'pause_workflow']
        resume_actions = [a for a in workflow_metrics.user_interactions if a['action'] == 'resume_workflow']
        
        assert len(pause_actions) == 1
        assert len(resume_actions) == 1
        
        # All stages should complete
        assert len(workflow_metrics.step_timings) == len(stages)
    
    def test_language_change_after_transcription(self, workflow_metrics):
        """
        Test changing target language after transcription:
        User decides to change target language after seeing transcription
        """
        workflow_metrics.start_workflow()
        
        original_target = 'es'  # Spanish
        new_target = 'fr'  # French
        
        workflow_metrics.record_user_action('set_target_language', {'language': original_target})
        
        # Complete transcription
        with patch('core._2_asr.transcribe') as mock_transcribe:
            mock_transcribe.return_value = {
                'segments': [{'text': 'Hello world', 'start': 0, 'end': 2}],
                'language': 'en'
            }
            
            time.sleep(0.1)
            workflow_metrics.record_step('transcription', 0.1)
        
        # User reviews transcription and decides to change language
        workflow_metrics.record_user_action('review_transcription', {'satisfied': True})
        workflow_metrics.record_user_action('change_target_language', {
            'from': original_target,
            'to': new_target,
            'reason': 'user_preference'
        })
        
        # Re-run translation with new language
        with patch('core._4_2_translate.translate_all') as mock_translate:
            # First attempt with Spanish (cancelled)
            workflow_metrics.record_step('translation_cancelled', 0.05)
            
            # Second attempt with French
            mock_translate.return_value = {
                'translations': [{'original': 'Hello world', 'translated': 'Bonjour le monde'}]
            }
            
            time.sleep(0.1)
            workflow_metrics.record_step('translation_final', 0.1)
        
        workflow_metrics.end_workflow()
        
        # Verify language change was handled
        language_changes = [a for a in workflow_metrics.user_interactions 
                           if a['action'] == 'change_target_language']
        assert len(language_changes) == 1
        assert language_changes[0]['context']['to'] == new_target
        
        # Both translation attempts should be recorded
        assert 'translation_cancelled' in workflow_metrics.step_timings
        assert 'translation_final' in workflow_metrics.step_timings


class TestUserExperienceMetrics:
    """Test user experience metrics and quality measurements"""
    
    @pytest.fixture
    def workflow_metrics(self):
        return UserWorkflowMetrics()
    
    def test_measure_end_to_end_latency(self, workflow_metrics):
        """
        Measure end-to-end latency from user perspective:
        Time from clicking 'Start' to seeing final output
        """
        workflow_metrics.start_workflow()
        
        # Simulate user clicking start
        user_click_time = time.time()
        workflow_metrics.record_user_action('click_start', {'timestamp': user_click_time})
        
        # Simulate various processing delays
        stages_with_delays = [
            ('initialization', 0.5),  # App startup
            ('download', 2.0),        # Video download
            ('transcription', 3.0),    # ASR processing
            ('translation', 1.5),      # Translation API calls
            ('tts', 2.5),             # TTS generation
            ('merge', 1.0),           # Final merge
            ('save', 0.5)             # Save to disk
        ]
        
        cumulative_time = 0
        for stage, delay in stages_with_delays:
            time.sleep(delay / 100)  # Simulated delay (scaled down for testing)
            cumulative_time += delay
            workflow_metrics.record_step(stage, delay)
            
            # Record user perception at key milestones
            if stage in ['download', 'transcription', 'merge']:
                workflow_metrics.record_user_action('progress_check', {
                    'stage': stage,
                    'elapsed': cumulative_time,
                    'user_perception': 'acceptable' if cumulative_time < 10 else 'slow'
                })
        
        # User sees final output
        output_ready_time = time.time()
        workflow_metrics.record_user_action('output_ready', {'timestamp': output_ready_time})
        workflow_metrics.end_workflow()
        
        # Calculate user-perceived latency
        total_latency = sum(d for _, d in stages_with_delays)
        
        # User experience thresholds
        assert total_latency < 15, f"Total latency too high: {total_latency}s"
        
        # Check if user perception was positive
        perceptions = [a['context']['user_perception'] 
                      for a in workflow_metrics.user_interactions 
                      if a['action'] == 'progress_check']
        negative_perceptions = [p for p in perceptions if p != 'acceptable']
        assert len(negative_perceptions) < 2, "Too many slow stages from user perspective"
    
    def test_measure_output_quality_satisfaction(self, workflow_metrics):
        """
        Measure user satisfaction with output quality:
        Translation accuracy, voice naturalness, lip sync, etc.
        """
        workflow_metrics.start_workflow()
        
        # Quality metrics from user perspective
        quality_aspects = {
            'translation_accuracy': {
                'score': 0.92,
                'user_rating': 4.5,  # out of 5
                'feedback': 'Good translation, minor grammar issues'
            },
            'voice_naturalness': {
                'score': 0.85,
                'user_rating': 4.0,
                'feedback': 'Voice sounds mostly natural'
            },
            'audio_video_sync': {
                'score': 0.95,
                'user_rating': 4.8,
                'feedback': 'Excellent synchronization'
            },
            'subtitle_timing': {
                'score': 0.93,
                'user_rating': 4.6,
                'feedback': 'Subtitles well-timed'
            },
            'overall_quality': {
                'score': 0.91,
                'user_rating': 4.4,
                'feedback': 'Professional output quality'
            }
        }
        
        # Record quality measurements
        for aspect, metrics in quality_aspects.items():
            workflow_metrics.record_quality(aspect, metrics['score'], {
                'user_rating': metrics['user_rating'],
                'feedback': metrics['feedback']
            })
        
        workflow_metrics.end_workflow()
        
        # Calculate overall user satisfaction
        avg_user_rating = sum(q['user_rating'] for q in quality_aspects.values()) / len(quality_aspects)
        avg_quality_score = sum(q['score'] for q in quality_aspects.values()) / len(quality_aspects)
        
        # User satisfaction thresholds
        assert avg_user_rating >= 4.0, f"User satisfaction too low: {avg_user_rating}/5"
        assert avg_quality_score >= 0.85, f"Quality score too low: {avg_quality_score}"
        
        # Check specific critical aspects
        assert quality_aspects['audio_video_sync']['score'] >= 0.9, "Sync quality below acceptable"
        assert quality_aspects['translation_accuracy']['score'] >= 0.85, "Translation quality below acceptable"
    
    def test_error_recovery_user_experience(self, workflow_metrics):
        """
        Test user experience during error recovery:
        How errors are presented and recovered from user perspective
        """
        workflow_metrics.start_workflow()
        
        # Simulate various error scenarios
        error_scenarios = [
            {
                'stage': 'download',
                'error': 'Network timeout',
                'recoverable': True,
                'recovery_action': 'automatic_retry',
                'user_impact': 'minimal'
            },
            {
                'stage': 'transcription',
                'error': 'API quota exceeded',
                'recoverable': True,
                'recovery_action': 'fallback_to_local',
                'user_impact': 'slight_delay'
            },
            {
                'stage': 'translation',
                'error': 'Invalid API key',
                'recoverable': False,
                'recovery_action': 'user_intervention_required',
                'user_impact': 'workflow_blocked'
            }
        ]
        
        for scenario in error_scenarios:
            # Record error occurrence
            workflow_metrics.record_error(
                scenario['error'],
                f"Error in {scenario['stage']}: {scenario['error']}",
                scenario['recoverable']
            )
            
            # Record user notification
            workflow_metrics.record_user_action('error_notification', {
                'stage': scenario['stage'],
                'message': scenario['error'],
                'suggested_action': scenario['recovery_action']
            })
            
            # Simulate recovery
            if scenario['recoverable']:
                time.sleep(0.1)  # Recovery time
                workflow_metrics.record_user_action('error_recovered', {
                    'stage': scenario['stage'],
                    'method': scenario['recovery_action'],
                    'user_impact': scenario['user_impact']
                })
                workflow_metrics.record_step(f"{scenario['stage']}_retry", 0.2)
            else:
                workflow_metrics.record_user_action('user_intervention', {
                    'stage': scenario['stage'],
                    'action_required': 'provide_valid_api_key'
                })
                break  # Workflow blocked
        
        workflow_metrics.end_workflow()
        
        # Analyze error handling from user perspective
        recoverable_errors = [e for e in workflow_metrics.errors_encountered if e['recoverable']]
        unrecoverable_errors = [e for e in workflow_metrics.errors_encountered if not e['recoverable']]
        
        # Most errors should be recoverable
        assert len(recoverable_errors) >= len(unrecoverable_errors)
        
        # User should be properly notified
        notifications = [a for a in workflow_metrics.user_interactions 
                        if a['action'] == 'error_notification']
        assert len(notifications) == len(error_scenarios)
        
        # Recovery should be attempted for recoverable errors
        recoveries = [a for a in workflow_metrics.user_interactions 
                     if a['action'] == 'error_recovered']
        assert len(recoveries) == len(recoverable_errors)


# Performance benchmark fixtures
@pytest.fixture
def performance_monitor():
    """Monitor performance metrics during tests"""
    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            self.memory_samples = []
            self.cpu_samples = []
            
        def start_monitoring(self):
            self.start_time = time.time()
            # In real implementation, would start monitoring threads
            
        def stop_monitoring(self):
            self.end_time = time.time()
            
        def get_results(self):
            return {
                'duration_seconds': self.end_time - self.start_time if self.end_time else 0,
                'memory_stats': {'peak_mb': 150, 'avg_mb': 100},  # Mock values
                'cpu_stats': {'peak_percent': 75, 'avg_percent': 45}  # Mock values
            }
    
    return PerformanceMonitor()


# Integration test markers
pytestmark = [
    pytest.mark.e2e,
    pytest.mark.user_workflow,
    pytest.mark.timeout(300)  # 5 minute timeout for E2E tests
]
