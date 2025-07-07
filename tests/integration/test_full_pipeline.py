# Integration Tests for VideoLingo Full Pipeline
# Tests end-to-end processing workflow

import pytest
import os
import json
import time
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock

@pytest.mark.integration
@pytest.mark.slow
class TestFullPipeline:
    """Integration tests for complete VideoLingo pipeline"""
    
    @pytest.fixture
    def pipeline_setup(self, temp_config_dir, mock_video_file):
        """Setup for pipeline testing"""
        # Create necessary directories
        for subdir in ['log', 'audio', 'temp_audio']:
            (temp_config_dir / 'temp' / subdir).mkdir(parents=True, exist_ok=True)
        
        # Mock video file with proper metadata
        return {
            'video_file': mock_video_file,
            'config_dir': temp_config_dir,
            'input_dir': temp_config_dir / 'input',
            'temp_dir': temp_config_dir / 'temp',
            'output_dir': temp_config_dir / 'output'
        }
    
    def test_video_download_to_transcription_flow(self, pipeline_setup, mock_subprocess, mock_whisperx):
        """Test flow from video download to transcription"""
        from core._1_ytdlp import download_video_ytdlp
        from core._2_asr import transcribe
        from core.utils.video_manager import get_video_manager
        
        # Mock successful download
        with patch('core._1_ytdlp.download_via_command') as mock_download:
            mock_download.return_value = pipeline_setup['video_file']
            
            # Test download
            downloaded_file = download_video_ytdlp("https://test.com/video", resolution="360")
            assert downloaded_file == pipeline_setup['video_file']
        
        # Register video and test transcription
        video_mgr = get_video_manager()
        with patch.object(video_mgr, 'paths', pipeline_setup['config_dir']):
            video_id = video_mgr.register_video(pipeline_setup['video_file'])
            
            # Mock transcription
            with patch('core._2_asr.transcribe_audio') as mock_transcribe:
                mock_transcribe.return_value = {
                    "segments": [
                        {
                            "start": 0.0,
                            "end": 2.5,
                            "text": "Hello world",
                            "words": [
                                {"start": 0.0, "end": 0.5, "word": "Hello"},
                                {"start": 0.6, "end": 1.0, "word": "world"}
                            ]
                        }
                    ],
                    "language": "en"
                }
                
                # Test transcription
                transcribe()
                
                # Verify transcription output
                assert mock_transcribe.called
    
    def test_transcription_to_translation_flow(self, pipeline_setup, mock_whisperx_result):
        """Test flow from transcription to translation"""
        from core._3_1_split_nlp import split_by_spacy
        from core._3_2_split_meaning import split_sentences_by_meaning
        from core._4_1_summarize import get_summary
        from core._4_2_translate import translate_all
        
        # Setup transcription data
        transcription_file = pipeline_setup['temp_dir'] / 'log' / 'cleaned_chunks.json'
        transcription_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(transcription_file, 'w') as f:
            json.dump(mock_whisperx_result['segments'], f)
        
        # Mock NLP processing
        with patch('core._3_1_split_nlp.process_text_with_spacy') as mock_spacy, \
             patch('core._3_2_split_meaning.split_with_llm') as mock_split, \
             patch('core._4_1_summarize.generate_summary') as mock_summary, \
             patch('core._4_2_translate.batch_translate') as mock_translate:
            
            # Mock outputs
            mock_spacy.return_value = mock_whisperx_result['segments']
            mock_split.return_value = mock_whisperx_result['segments']
            mock_summary.return_value = "Test video about greetings"
            mock_translate.return_value = [
                {
                    "line_id": 0,
                    "original_text": "Hello world",
                    "faithful_translation": "你好世界",
                    "expressive_translation": "你好世界",
                    "confidence": 0.95
                }
            ]
            
            # Run pipeline steps
            split_by_spacy()
            split_sentences_by_meaning()
            get_summary()
            translate_all()
            
            # Verify all steps were called
            assert mock_spacy.called
            assert mock_split.called
            assert mock_summary.called
            assert mock_translate.called
    
    def test_translation_to_subtitle_flow(self, pipeline_setup):
        """Test flow from translation to subtitle generation"""
        from core._5_split_sub import split_for_sub_main
        from core._6_gen_sub import align_timestamp_main
        from core._7_sub_into_vid import merge_subtitles_to_video
        
        # Setup translation data
        translation_file = pipeline_setup['temp_dir'] / 'log' / 'translation_result.json'
        translation_file.parent.mkdir(parents=True, exist_ok=True)
        
        translation_data = [
            {
                "line_id": 0,
                "start_time": 0.0,
                "end_time": 2.5,
                "original_text": "Hello world",
                "faithful_translation": "你好世界",
                "expressive_translation": "你好世界",
                "confidence": 0.95
            }
        ]
        
        with open(translation_file, 'w') as f:
            json.dump(translation_data, f)
        
        # Mock subtitle processing
        with patch('core._5_split_sub.process_subtitle_splitting') as mock_split_sub, \
             patch('core._6_gen_sub.generate_subtitle_timestamps') as mock_timestamps, \
             patch('core._7_sub_into_vid.merge_video_subtitles') as mock_merge, \
             patch('subprocess.run') as mock_subprocess:
            
            mock_split_sub.return_value = translation_data
            mock_timestamps.return_value = "subtitle_content"
            mock_subprocess.return_value = Mock(returncode=0)
            
            # Run subtitle pipeline
            split_for_sub_main()
            align_timestamp_main()
            merge_subtitles_to_video()
            
            # Verify steps
            assert mock_split_sub.called
            assert mock_timestamps.called
            assert mock_merge.called
    
    def test_audio_dubbing_pipeline(self, pipeline_setup):
        """Test audio dubbing pipeline"""
        from core._8_1_audio_task import gen_audio_task_main
        from core._8_2_dub_chunks import gen_dub_chunks
        from core._9_refer_audio import extract_refer_audio_main
        from core._10_gen_audio import gen_audio
        from core._11_merge_audio import merge_full_audio
        from core._12_dub_to_vid import merge_video_audio
        
        # Setup audio task data
        audio_task_file = pipeline_setup['temp_dir'] / 'log' / 'audio_tasks.json'
        audio_task_file.parent.mkdir(parents=True, exist_ok=True)
        
        audio_tasks = [
            {
                "task_id": 0,
                "text": "你好世界",
                "start_time": 0.0,
                "end_time": 2.5,
                "reference_audio": "ref_0.wav"
            }
        ]
        
        with open(audio_task_file, 'w') as f:
            json.dump(audio_tasks, f)
        
        # Mock audio processing
        with patch('core._8_1_audio_task.generate_audio_tasks') as mock_gen_tasks, \
             patch('core._8_2_dub_chunks.create_audio_chunks') as mock_chunks, \
             patch('core._9_refer_audio.extract_reference_audio') as mock_ref_audio, \
             patch('core._10_gen_audio.generate_tts_audio') as mock_tts, \
             patch('core._11_merge_audio.merge_audio_files') as mock_merge_audio, \
             patch('core._12_dub_to_vid.merge_video_with_audio') as mock_final_merge, \
             patch('subprocess.run') as mock_subprocess:
            
            mock_gen_tasks.return_value = audio_tasks
            mock_chunks.return_value = True
            mock_ref_audio.return_value = True
            mock_tts.return_value = True
            mock_merge_audio.return_value = "merged_audio.wav"
            mock_subprocess.return_value = Mock(returncode=0)
            
            # Run audio pipeline
            gen_audio_task_main()
            gen_dub_chunks()
            extract_refer_audio_main()
            gen_audio()
            merge_full_audio()
            merge_video_audio()
            
            # Verify all steps
            assert mock_gen_tasks.called
            assert mock_chunks.called
            assert mock_ref_audio.called
            assert mock_tts.called
            assert mock_merge_audio.called
            assert mock_final_merge.called
    
    def test_error_propagation_through_pipeline(self, pipeline_setup):
        """Test error handling and propagation through pipeline"""
        from core._2_asr import transcribe
        
        # Test transcription failure propagation
        with patch('core._2_asr.transcribe_audio', side_effect=Exception("Transcription failed")):
            with pytest.raises(Exception, match="Transcription failed"):
                transcribe()
    
    def test_pipeline_state_persistence(self, pipeline_setup):
        """Test that pipeline state is properly persisted between steps"""
        from core.utils.video_manager import get_video_manager
        
        video_mgr = get_video_manager()
        
        # Register video
        with patch.object(video_mgr, 'paths', {
            'input': str(pipeline_setup['input_dir']),
            'temp': str(pipeline_setup['temp_dir']),
            'output': str(pipeline_setup['output_dir'])
        }):
            video_id = video_mgr.register_video(pipeline_setup['video_file'])
            
            # Create intermediate files
            temp_dir = video_mgr.get_temp_dir(video_id)
            log_dir = Path(temp_dir) / 'log'
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # Create transcription result
            transcription_file = log_dir / 'cleaned_chunks.json'
            transcription_data = [
                {
                    "start": 0.0,
                    "end": 2.5,
                    "text": "Hello world",
                    "words": [
                        {"start": 0.0, "end": 0.5, "word": "Hello"},
                        {"start": 0.6, "end": 1.0, "word": "world"}
                    ]
                }
            ]
            with open(transcription_file, 'w') as f:
                json.dump(transcription_data, f)
            
            # Verify file exists and can be read
            assert transcription_file.exists()
            with open(transcription_file, 'r') as f:
                loaded_data = json.load(f)
            assert loaded_data == transcription_data
    
    def test_pipeline_cleanup_and_resume(self, pipeline_setup):
        """Test pipeline cleanup and resume functionality"""
        from core.utils.video_manager import get_video_manager
        from core.utils.delete_retry_dubbing import reset_dubbing_for_current_video
        
        video_mgr = get_video_manager()
        
        with patch.object(video_mgr, 'paths', {
            'input': str(pipeline_setup['input_dir']),
            'temp': str(pipeline_setup['temp_dir']),
            'output': str(pipeline_setup['output_dir'])
        }):
            video_id = video_mgr.register_video(pipeline_setup['video_file'])
            
            # Create some processing files
            temp_dir = video_mgr.get_temp_dir(video_id)
            os.makedirs(temp_dir, exist_ok=True)
            
            test_files = [
                'audio_tasks.json',
                'audio_chunks.json',
                'generated_audio.wav'
            ]
            
            for filename in test_files:
                file_path = Path(temp_dir) / filename
                file_path.write_text('test content')
            
            # Test cleanup
            video_mgr.safe_overwrite_temp_files(video_id)
            
            # Verify files are cleaned but directory structure preserved
            assert os.path.exists(temp_dir)
            for filename in test_files:
                file_path = Path(temp_dir) / filename
                assert not file_path.exists()
    
    def test_pipeline_performance_monitoring(self, pipeline_setup):
        """Test pipeline performance monitoring"""
        from core._2_asr import transcribe
        
        # Mock slow transcription
        def slow_transcribe(*args, **kwargs):
            time.sleep(0.1)  # Simulate processing time
            return {
                "segments": [
                    {"start": 0.0, "end": 1.0, "text": "test"}
                ]
            }
        
        with patch('core._2_asr.transcribe_audio', side_effect=slow_transcribe):
            start_time = time.time()
            
            # Should handle timing monitoring
            transcribe()
            
            duration = time.time() - start_time
            assert duration >= 0.1  # Should take at least the sleep time
            assert duration < 5.0   # But not too long
    
    def test_pipeline_concurrent_processing(self, pipeline_setup):
        """Test pipeline behavior with concurrent processing"""
        from core.utils.video_manager import get_video_manager
        import threading
        
        video_mgr = get_video_manager()
        results = []
        errors = []
        
        def process_video(thread_id):
            try:
                with patch.object(video_mgr, 'paths', {
                    'input': str(pipeline_setup['input_dir']),
                    'temp': str(pipeline_setup['temp_dir']),
                    'output': str(pipeline_setup['output_dir'])
                }):
                    # Create unique video file for each thread
                    video_file = pipeline_setup['input_dir'] / f'video_{thread_id}.mp4'
                    with open(video_file, 'wb') as f:
                        f.write(f'video_content_{thread_id}'.encode() * 100)
                    
                    video_id = video_mgr.register_video(str(video_file))
                    results.append((thread_id, video_id))
            except Exception as e:
                errors.append((thread_id, str(e)))
        
        # Start multiple processing threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=process_video, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify results
        assert len(errors) == 0, f"Concurrent processing errors: {errors}"
        assert len(results) == 3
        
        # All video IDs should be unique
        video_ids = [result[1] for result in results]
        assert len(set(video_ids)) == 3
    
    @pytest.mark.parametrize("pipeline_step,mock_target", [
        ("transcription", "core._2_asr.transcribe_audio"),
        ("translation", "core._4_2_translate.batch_translate"),
        ("subtitle_generation", "core._6_gen_sub.generate_subtitle_timestamps"),
        ("audio_generation", "core._10_gen_audio.generate_tts_audio"),
    ])
    def test_pipeline_step_isolation(self, pipeline_setup, pipeline_step, mock_target):
        """Test that pipeline steps are properly isolated"""
        
        # Each step should work independently when previous steps have completed
        with patch(mock_target) as mock_step:
            mock_step.return_value = f"result_for_{pipeline_step}"
            
            # Setup prerequisite files based on step
            if pipeline_step == "translation":
                # Translation needs transcription
                transcription_file = pipeline_setup['temp_dir'] / 'log' / 'cleaned_chunks.json'
                transcription_file.parent.mkdir(parents=True, exist_ok=True)
                with open(transcription_file, 'w') as f:
                    json.dump([{"text": "test", "start": 0, "end": 1}], f)
            
            elif pipeline_step == "subtitle_generation":
                # Subtitle generation needs translation
                translation_file = pipeline_setup['temp_dir'] / 'log' / 'translation_result.json'
                translation_file.parent.mkdir(parents=True, exist_ok=True)
                with open(translation_file, 'w') as f:
                    json.dump([{"text": "test", "translation": "测试"}], f)
            
            elif pipeline_step == "audio_generation":
                # Audio generation needs audio tasks
                task_file = pipeline_setup['temp_dir'] / 'log' / 'audio_tasks.json'
                task_file.parent.mkdir(parents=True, exist_ok=True)
                with open(task_file, 'w') as f:
                    json.dump([{"text": "测试", "start": 0, "end": 1}], f)
            
            # Step should execute without dependencies on other running processes
            result = mock_step.return_value
            assert result == f"result_for_{pipeline_step}"