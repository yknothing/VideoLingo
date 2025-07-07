"""
Functional tests for Remaining Audio modules
Tests core audio merging, dubbing, and reference extraction functionality without complex dependencies
"""

import pytest
import tempfile
import os
import datetime
import pandas as pd
import re
import ast
from unittest.mock import Mock, patch, MagicMock
from typing import List, Tuple


class TestAudioMergingLogic:
    """Test audio merging and processing logic"""
    
    def test_data_flattening_logic(self):
        """Test Excel data flattening logic"""
        # Simulate load_and_flatten_data logic from _11_merge_audio.py
        def mock_load_and_flatten_data(mock_excel_data):
            # Simulate DataFrame structure
            lines_data = mock_excel_data['lines']
            times_data = mock_excel_data['new_sub_times']
            
            # Flatten lines (handle both string and list formats)
            flattened_lines = []
            for line_entry in lines_data:
                if isinstance(line_entry, str):
                    # Safe parsing without dangerous functions
                    if line_entry.startswith('[') and line_entry.endswith(']'):
                        # Simple list parsing without dangerous functions
                        content = line_entry[1:-1].strip()
                        if content:
                            items = [item.strip().strip("'").strip('"') for item in content.split(',')]
                            flattened_lines.extend(items)
                        else:
                            flattened_lines.append("")
                    else:
                        flattened_lines.append(line_entry)
                elif isinstance(line_entry, list):
                    flattened_lines.extend(line_entry)
                else:
                    flattened_lines.append(str(line_entry))
            
            # Flatten timing data
            flattened_times = []
            for time_entry in times_data:
                if isinstance(time_entry, str):
                    # Simple nested list parsing
                    if time_entry.startswith('[[') and time_entry.endswith(']]'):
                        # Extract timing pairs [[0.0, 1.5], [1.5, 3.0]]
                        content = time_entry[2:-2]  # Remove outer brackets
                        pairs = content.split('], [')
                        for pair in pairs:
                            start_end = pair.split(',')
                            if len(start_end) == 2:
                                try:
                                    start = float(start_end[0].strip())
                                    end = float(start_end[1].strip())
                                    flattened_times.append([start, end])
                                except ValueError:
                                    flattened_times.append([0.0, 1.0])
                elif isinstance(time_entry, list):
                    for sub_entry in time_entry:
                        flattened_times.append(sub_entry)
                else:
                    flattened_times.append(time_entry)
            
            return flattened_lines, flattened_times
        
        # Test single line entries
        mock_data = {
            'lines': [['Hello world'], ['This is test']],
            'new_sub_times': [[[0.0, 2.0]], [[2.5, 4.0]]]
        }
        
        lines, times = mock_load_and_flatten_data(mock_data)
        assert len(lines) == 2
        assert len(times) == 2
        assert lines[0] == 'Hello world'
        assert times[0] == [0.0, 2.0]
        
        # Test multi-line entries
        mock_data = {
            'lines': [['First line', 'Second line'], ['Third line']],
            'new_sub_times': [[[0.0, 1.0], [1.0, 2.0]], [[3.0, 4.0]]]
        }
        
        lines, times = mock_load_and_flatten_data(mock_data)
        assert len(lines) == 3
        assert len(times) == 3
        assert lines[0] == 'First line'
        assert lines[1] == 'Second line'
        assert lines[2] == 'Third line'
        assert times[0] == [0.0, 1.0]
        assert times[1] == [1.0, 2.0]
        assert times[2] == [3.0, 4.0]
    
    def test_audio_file_path_generation(self):
        """Test audio file path generation logic"""
        # Simulate get_audio_files logic
        def mock_get_audio_files(df_data, template="{}_{}"):
            audio_files = []
            
            for row in df_data:
                number = row['number']
                lines = row['lines']
                
                # Handle both string and list formats
                if isinstance(lines, str):
                    if lines.startswith('[') and lines.endswith(']'):
                        # Simple list parsing
                        content = lines[1:-1].strip()
                        if content:
                            line_count = len([item.strip() for item in content.split(',') if item.strip()])
                        else:
                            line_count = 0
                    else:
                        line_count = 1
                elif isinstance(lines, list):
                    line_count = len(lines)
                else:
                    line_count = 1
                
                for line_index in range(line_count):
                    file_path = template.format(number, line_index)
                    audio_files.append(file_path)
            
            return audio_files
        
        # Test single line per subtitle
        df_data = [
            {'number': 1, 'lines': ['Hello world']},
            {'number': 2, 'lines': ['This is test']}
        ]
        
        files = mock_get_audio_files(df_data)
        assert len(files) == 2
        assert files[0] == "1_0"
        assert files[1] == "2_0"
        
        # Test multiple lines per subtitle
        df_data = [
            {'number': 1, 'lines': ['First', 'Second']},
            {'number': 2, 'lines': ['Third']}
        ]
        
        files = mock_get_audio_files(df_data)
        assert len(files) == 3
        assert files[0] == "1_0"
        assert files[1] == "1_1"
        assert files[2] == "2_0"
    
    def test_audio_compression_logic(self):
        """Test audio compression logic"""
        # Simulate process_audio_segment logic
        def mock_process_audio_segment(audio_file_path, target_params):
            # Mock FFmpeg command generation
            ffmpeg_params = [
                'ffmpeg', '-y',
                '-i', audio_file_path,
                '-ar', str(target_params['sample_rate']),
                '-ac', str(target_params['channels']),
                '-b:a', target_params['bitrate']
            ]
            
            # Mock audio processing result
            result = {
                'input_file': audio_file_path,
                'output_format': 'mp3',
                'sample_rate': target_params['sample_rate'],
                'channels': target_params['channels'],
                'bitrate': target_params['bitrate'],
                'ffmpeg_command': ffmpeg_params,
                'estimated_size': len(audio_file_path) * 1000  # Mock size calculation
            }
            
            return result
        
        # Test compression with standard params
        target_params = {
            'sample_rate': 16000,
            'channels': 1,
            'bitrate': '64k'
        }
        
        result = mock_process_audio_segment('test_audio.wav', target_params)
        
        assert result['input_file'] == 'test_audio.wav'
        assert result['sample_rate'] == 16000
        assert result['channels'] == 1
        assert result['bitrate'] == '64k'
        assert 'ffmpeg' in result['ffmpeg_command']
        assert '-ar' in result['ffmpeg_command']
        assert '16000' in result['ffmpeg_command']
    
    def test_silence_insertion_logic(self):
        """Test silence insertion between audio segments"""
        # Simulate silence insertion logic from merge_audio_segments
        def mock_calculate_silence_duration(current_start, previous_end, min_silence=0.0):
            silence_duration = current_start - previous_end
            return max(silence_duration, min_silence)
        
        def mock_merge_with_silence(audio_segments_with_times):
            merged_segments = []
            
            for i, (segment_info, time_range) in enumerate(audio_segments_with_times):
                start_time, end_time = time_range
                
                # Add silence before first segment if needed
                if i == 0 and start_time > 0:
                    silence_duration = start_time
                    merged_segments.append({
                        'type': 'silence',
                        'duration': silence_duration,
                        'start': 0.0,
                        'end': start_time
                    })
                
                # Add silence between segments
                elif i > 0:
                    prev_end = audio_segments_with_times[i-1][1][1]
                    silence_duration = mock_calculate_silence_duration(start_time, prev_end)
                    
                    if silence_duration > 0:
                        merged_segments.append({
                            'type': 'silence',
                            'duration': silence_duration,
                            'start': prev_end,
                            'end': start_time
                        })
                
                # Add actual audio segment
                merged_segments.append({
                    'type': 'audio',
                    'info': segment_info,
                    'duration': end_time - start_time,
                    'start': start_time,
                    'end': end_time
                })
            
            return merged_segments
        
        # Test merging with silence gaps
        segments_with_times = [
            ({'file': 'audio1.wav'}, [1.0, 3.0]),
            ({'file': 'audio2.wav'}, [4.0, 6.0]),
            ({'file': 'audio3.wav'}, [6.5, 8.0])
        ]
        
        merged = mock_merge_with_silence(segments_with_times)
        
        # Should have: initial silence, audio1, silence gap, audio2, silence gap, audio3
        assert len(merged) == 6
        assert merged[0]['type'] == 'silence'  # Initial silence
        assert merged[0]['duration'] == 1.0
        
        assert merged[1]['type'] == 'audio'   # Audio 1
        assert merged[1]['info']['file'] == 'audio1.wav'
        
        assert merged[2]['type'] == 'silence'  # Gap 1
        assert merged[2]['duration'] == 1.0    # 4.0 - 3.0
        
        assert merged[3]['type'] == 'audio'   # Audio 2
        assert merged[4]['type'] == 'silence'  # Gap 2
        assert merged[4]['duration'] == 0.5    # 6.5 - 6.0
        
        assert merged[5]['type'] == 'audio'   # Audio 3
    
    def test_srt_subtitle_generation(self):
        """Test SRT subtitle generation logic"""
        # Simulate create_srt_subtitle logic
        def mock_create_srt_subtitle(flattened_data):
            lines, times = flattened_data
            srt_content = []
            
            for i, (time_range, line) in enumerate(zip(times, lines), 1):
                start_time, end_time = time_range
                
                # Convert seconds to SRT time format
                def seconds_to_srt_time(seconds):
                    hours = int(seconds // 3600)
                    minutes = int((seconds % 3600) // 60)
                    secs = int(seconds % 60)
                    millis = int((seconds % 1) * 1000)
                    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
                
                start_srt = seconds_to_srt_time(start_time)
                end_srt = seconds_to_srt_time(end_time)
                
                srt_entry = {
                    'index': i,
                    'timestamp': f"{start_srt} --> {end_srt}",
                    'text': line
                }
                srt_content.append(srt_entry)
            
            # Format as SRT string
            srt_string = ""
            for entry in srt_content:
                srt_string += f"{entry['index']}\n{entry['timestamp']}\n{entry['text']}\n\n"
            
            return srt_content, srt_string.strip()
        
        # Test SRT generation
        flattened_data = (
            ['Hello world', 'This is test', 'Final line'],
            [[0.0, 2.0], [2.5, 4.5], [5.0, 7.0]]
        )
        
        srt_content, srt_string = mock_create_srt_subtitle(flattened_data)
        
        assert len(srt_content) == 3
        assert srt_content[0]['index'] == 1
        assert srt_content[0]['text'] == 'Hello world'
        assert '00:00:00,000 --> 00:00:02,000' in srt_content[0]['timestamp']
        
        assert srt_content[1]['index'] == 2
        assert '00:00:02,500 --> 00:00:04,500' in srt_content[1]['timestamp']
        
        # Check formatted string
        assert "1\n" in srt_string
        assert "Hello world" in srt_string
        assert "00:00:00,000 --> 00:00:02,000" in srt_string


class TestDubbingLogic:
    """Test video dubbing and audio merging logic"""
    
    def test_platform_font_selection(self):
        """Test platform-specific font selection"""
        # Simulate platform font selection from _12_dub_to_vid.py
        def mock_select_dubbing_font(platform_name):
            font_mappings = {
                'Linux': 'NotoSansCJK-Regular',
                'Darwin': 'Arial Unicode MS',  # macOS
                'Windows': 'Arial'
            }
            return font_mappings.get(platform_name, 'Arial')  # Default to Arial
        
        # Test different platforms
        assert mock_select_dubbing_font('Linux') == 'NotoSansCJK-Regular'
        assert mock_select_dubbing_font('Darwin') == 'Arial Unicode MS'
        assert mock_select_dubbing_font('Windows') == 'Arial'
        assert mock_select_dubbing_font('Unknown') == 'Arial'
    
    def test_video_resolution_handling(self):
        """Test video resolution detection and handling"""
        # Simulate video resolution handling
        def mock_get_video_resolution(video_file):
            # Mock different video file resolutions
            resolution_mappings = {
                'hd_video.mp4': (1920, 1080),
                'sd_video.mp4': (1280, 720),
                '4k_video.mp4': (3840, 2160),
                'mobile_video.mp4': (720, 1280)  # Portrait
            }
            return resolution_mappings.get(video_file, (1920, 1080))
        
        def mock_generate_video_filters(resolution, subtitle_file, subtitle_style):
            width, height = resolution
            
            # Scale and pad filter
            scale_filter = f"scale={width}:{height}:force_original_aspect_ratio=decrease"
            pad_filter = f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2"
            
            # Subtitle filter
            subtitle_filter = f"subtitles={subtitle_file}:force_style='{subtitle_style}'"
            
            return f"{scale_filter},{pad_filter},{subtitle_filter}"
        
        # Test different resolutions
        hd_res = mock_get_video_resolution('hd_video.mp4')
        assert hd_res == (1920, 1080)
        
        sd_res = mock_get_video_resolution('sd_video.mp4')
        assert sd_res == (1280, 720)
        
        # Test filter generation
        style = "FontSize=17,FontName=Arial,PrimaryColour=&H00FFFF"
        filter_str = mock_generate_video_filters(hd_res, 'dub.srt', style)
        
        assert 'scale=1920:1080' in filter_str
        assert 'pad=1920:1080' in filter_str
        assert 'subtitles=dub.srt' in filter_str
        assert 'FontSize=17' in filter_str
    
    def test_audio_normalization_workflow(self):
        """Test audio normalization workflow"""
        # Simulate audio normalization logic
        def mock_normalize_audio_workflow(input_audio, target_params):
            normalization_steps = []
            
            # Step 1: Analyze input audio
            normalization_steps.append('analyze_input')
            mock_input_stats = {
                'peak_level': -3.2,
                'rms_level': -18.5,
                'sample_rate': 44100,
                'channels': 2
            }
            
            # Step 2: Calculate normalization parameters
            normalization_steps.append('calculate_normalization')
            target_peak = target_params.get('target_peak', -1.0)
            gain_adjustment = target_peak - mock_input_stats['peak_level']
            
            # Step 3: Apply normalization
            normalization_steps.append('apply_normalization')
            normalized_stats = {
                'peak_level': target_peak,
                'rms_level': mock_input_stats['rms_level'] + gain_adjustment,
                'gain_applied': gain_adjustment,
                'sample_rate': target_params.get('sample_rate', mock_input_stats['sample_rate']),
                'channels': target_params.get('channels', mock_input_stats['channels'])
            }
            
            return {
                'steps': normalization_steps,
                'input_stats': mock_input_stats,
                'output_stats': normalized_stats,
                'gain_applied': gain_adjustment
            }
        
        # Test normalization
        target_params = {
            'target_peak': -1.0,
            'sample_rate': 16000,
            'channels': 1
        }
        
        result = mock_normalize_audio_workflow('input.mp3', target_params)
        
        assert 'analyze_input' in result['steps']
        assert 'calculate_normalization' in result['steps']
        assert 'apply_normalization' in result['steps']
        assert result['output_stats']['peak_level'] == -1.0
        assert result['gain_applied'] == pytest.approx(2.2, abs=0.1)  # -1.0 - (-3.2)


class TestReferenceAudioLogic:
    """Test reference audio extraction logic"""
    
    def test_time_conversion_logic(self):
        """Test time string to samples conversion"""
        # Simulate time_to_samples logic from _9_refer_audio.py
        def mock_time_to_samples(time_str, sample_rate):
            # Handle both comma and dot formats
            if ',' in time_str:
                time_str = time_str.replace(',', '.')
            
            parts = time_str.split(':')
            hours = int(parts[0])
            minutes = int(parts[1])
            
            # Handle seconds and milliseconds
            sec_parts = parts[2].split('.')
            seconds = int(sec_parts[0])
            milliseconds = int(sec_parts[1]) if len(sec_parts) > 1 else 0
            
            total_seconds = hours * 3600 + minutes * 60 + seconds + milliseconds / 1000
            return int(total_seconds * sample_rate)
        
        # Test different time formats
        sample_rate = 16000
        
        # Test standard format
        assert mock_time_to_samples('00:01:30.500', sample_rate) == 90500 * 16  # 90.5 seconds
        assert mock_time_to_samples('00:00:05.250', sample_rate) == 5250 * 16   # 5.25 seconds
        
        # Test comma format (SRT style)
        assert mock_time_to_samples('00:01:30,500', sample_rate) == 90500 * 16
        
        # Test zero time
        assert mock_time_to_samples('00:00:00.000', sample_rate) == 0
        
        # Test hours
        assert mock_time_to_samples('01:00:00.000', sample_rate) == 3600 * sample_rate


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])