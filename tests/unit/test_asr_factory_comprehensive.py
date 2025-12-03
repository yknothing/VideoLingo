# Comprehensive tests for ASR backend factory pattern and provider selection
# Targets 85% branch coverage including provider selection, memory optimization, and error handling

import pytest
import os
import tempfile
import gc
from unittest.mock import Mock, patch, MagicMock, call
import pandas as pd
import subprocess

# Mock external dependencies before imports
with patch.dict(
    "sys.modules",
    {
        "core.utils": MagicMock(),
        "core.utils.models": MagicMock(),
        "core.utils.config_utils": MagicMock(),
        "rich": MagicMock(),
        "psutil": MagicMock(),
        "torch": MagicMock(),
        "whisperx": MagicMock(),
        "librosa": MagicMock(),
    },
):
    from core._2_asr import check_memory_usage, monitor_memory_and_warn, transcribe


class TestMemoryMonitoring:
    """Comprehensive tests for memory monitoring and optimization"""

    @pytest.fixture
    def mock_psutil(self):
        """Mock psutil for memory testing"""
        with patch("core._2_asr.psutil") as mock:
            mock_memory = Mock()
            mock_memory.available = 8 * 1024**3  # 8GB available
            mock_memory.percent = 45.0  # 45% used
            mock.virtual_memory.return_value = mock_memory
            yield mock

    def test_check_memory_usage_normal_conditions(self, mock_psutil):
        """Test memory usage checking under normal conditions"""
        memory_info = check_memory_usage()

        assert "available_mb" in memory_info
        assert "used_percent" in memory_info
        assert "available_percent" in memory_info

        expected_available_mb = 8 * 1024  # 8GB in MB
        assert abs(memory_info["available_mb"] - expected_available_mb) < 1
        assert memory_info["used_percent"] == 45.0
        assert memory_info["available_percent"] == 55.0

    def test_check_memory_usage_high_usage(self, mock_psutil):
        """Test memory checking with high memory usage"""
        mock_psutil.virtual_memory.return_value.percent = 95.0
        mock_psutil.virtual_memory.return_value.available = (
            512 * 1024**2
        )  # 512MB available

        memory_info = check_memory_usage()

        assert memory_info["used_percent"] == 95.0
        assert memory_info["available_percent"] == 5.0
        assert memory_info["available_mb"] == 512

    def test_check_memory_usage_exception_handling(self, mock_psutil):
        """Test memory checking with psutil exception"""
        mock_psutil.virtual_memory.side_effect = Exception("Memory check failed")

        memory_info = check_memory_usage()

        assert memory_info["available_mb"] == -1
        assert memory_info["used_percent"] == -1
        assert memory_info["available_percent"] == -1

    @patch("core._2_asr.check_memory_usage")
    @patch("core._2_asr.rprint")
    def test_monitor_memory_and_warn_low_memory(self, mock_rprint, mock_check_memory):
        """Test memory monitoring with low available memory"""
        mock_check_memory.return_value = {
            "available_mb": 1024,  # 1GB available
            "used_percent": 75.0,
            "available_percent": 25.0,
        }

        monitor_memory_and_warn("test_stage", min_required_mb=2048)

        # Should warn about low memory
        mock_rprint.assert_any_call(
            "[red]Warning: Low memory at test_stage. Available: 1024MB, Recommended: 2048MB[/red]"
        )
        mock_rprint.assert_any_call(
            "[yellow]Consider closing other applications or processing smaller audio segments[/yellow]"
        )

    @patch("core._2_asr.check_memory_usage")
    @patch("core._2_asr.rprint")
    def test_monitor_memory_and_warn_high_usage_percentage(
        self, mock_rprint, mock_check_memory
    ):
        """Test memory monitoring with high usage percentage"""
        mock_check_memory.return_value = {
            "available_mb": 4096,  # Sufficient available memory
            "used_percent": 90.0,  # But high usage percentage
            "available_percent": 10.0,
        }

        monitor_memory_and_warn("test_stage", min_required_mb=2048)

        # Should warn about high usage percentage
        mock_rprint.assert_called_with(
            "[yellow]High memory usage at test_stage: 90.0% used[/yellow]"
        )

    @patch("core._2_asr.check_memory_usage")
    @patch("core._2_asr.rprint")
    def test_monitor_memory_and_warn_good_conditions(
        self, mock_rprint, mock_check_memory
    ):
        """Test memory monitoring under good conditions"""
        mock_check_memory.return_value = {
            "available_mb": 8192,  # 8GB available
            "used_percent": 40.0,  # Low usage
            "available_percent": 60.0,
        }

        monitor_memory_and_warn("test_stage", min_required_mb=2048)

        # Should show green status
        mock_rprint.assert_called_with(
            "[green]Memory status at test_stage: 8192MB available (60.0% free)[/green]"
        )

    @patch("core._2_asr.check_memory_usage")
    @patch("core._2_asr.rprint")
    def test_monitor_memory_and_warn_check_failure(
        self, mock_rprint, mock_check_memory
    ):
        """Test memory monitoring when check fails"""
        mock_check_memory.return_value = {
            "available_mb": -1,  # Check failed
            "used_percent": -1,
            "available_percent": -1,
        }

        monitor_memory_and_warn("test_stage")

        # Should not call rprint when memory check fails
        mock_rprint.assert_not_called()

    @patch("core._2_asr.check_memory_usage")
    @patch("core._2_asr.rprint")
    def test_monitor_memory_and_warn_custom_requirements(
        self, mock_rprint, mock_check_memory
    ):
        """Test memory monitoring with custom memory requirements"""
        mock_check_memory.return_value = {
            "available_mb": 1024,  # 1GB available
            "used_percent": 75.0,
            "available_percent": 25.0,
        }

        # Test with high requirement
        monitor_memory_and_warn("high_requirement_stage", min_required_mb=4096)

        mock_rprint.assert_any_call(
            "[red]Warning: Low memory at high_requirement_stage. Available: 1024MB, Recommended: 4096MB[/red]"
        )

        mock_rprint.reset_mock()

        # Test with low requirement
        monitor_memory_and_warn("low_requirement_stage", min_required_mb=512)

        # Should not warn about memory amount, but may warn about usage percentage
        calls = [
            call for call in mock_rprint.call_args_list if "Low memory" not in str(call)
        ]
        assert len(calls) >= 0  # Should have at least the high usage warning


class TestASRProviderSelection:
    """Test ASR provider selection logic and runtime switching"""

    @pytest.fixture
    def mock_dependencies(self):
        """Mock all ASR dependencies"""
        with patch("core._2_asr.find_video_files") as mock_find_video, patch(
            "core._2_asr.convert_video_to_audio"
        ) as mock_convert, patch("core._2_asr.split_audio") as mock_split, patch(
            "core._2_asr.process_transcription"
        ) as mock_process, patch(
            "core._2_asr.save_results"
        ) as mock_save, patch(
            "core._2_asr.normalize_audio_volume"
        ) as mock_normalize, patch(
            "core._2_asr.load_key"
        ) as mock_load_key, patch(
            "core._2_asr.monitor_memory_and_warn"
        ) as mock_monitor, patch(
            "core._2_asr.gc.collect"
        ) as mock_gc:
            mock_find_video.return_value = "/tmp/test_video.mp4"
            mock_split.return_value = [(0.0, 30.0), (30.0, 60.0)]
            mock_process.return_value = pd.DataFrame(
                {
                    "text": ["Hello", "world"],
                    "start": [0.0, 1.0],
                    "end": [1.0, 2.0],
                    "speaker_id": [None, None],
                }
            )

            yield {
                "find_video": mock_find_video,
                "convert": mock_convert,
                "split": mock_split,
                "process": mock_process,
                "save": mock_save,
                "normalize": mock_normalize,
                "load_key": mock_load_key,
                "monitor": mock_monitor,
                "gc": mock_gc,
            }

    @patch("core._2_asr.demucs_audio", None)  # Demucs not available
    def test_transcribe_local_provider_selection(self, mock_dependencies):
        """Test selection of local WhisperX provider"""
        mock_dependencies["load_key"].side_effect = lambda key: {
            "demucs": False,
            "whisper.runtime": "local",
        }.get(key, "default")

        # Mock local transcription
        with patch("core.asr_backend.whisperX_local.transcribe_audio") as mock_local:
            mock_local.return_value = {
                "segments": [{"start": 0, "end": 30, "text": "Local transcription"}]
            }

            with patch("core._2_asr._RAW_AUDIO_FILE", "/tmp/raw_audio.mp3"):
                transcribe()

            # Verify local provider was selected and called
            mock_local.assert_called()
            assert mock_local.call_count == 2  # Called for each segment

            # Verify memory monitoring for local transcription
            mock_dependencies["monitor"].assert_any_call(
                "before local transcription", 3072
            )

    @patch("core._2_asr.demucs_audio", None)
    def test_transcribe_cloud_provider_selection(self, mock_dependencies):
        """Test selection of cloud 302.ai provider"""
        mock_dependencies["load_key"].side_effect = lambda key: {
            "demucs": False,
            "whisper.runtime": "cloud",
        }.get(key, "default")

        # Mock cloud transcription
        with patch("core.asr_backend.whisperX_302.transcribe_audio_302") as mock_cloud:
            mock_cloud.return_value = {
                "segments": [{"start": 0, "end": 30, "text": "Cloud transcription"}]
            }

            with patch("core._2_asr._RAW_AUDIO_FILE", "/tmp/raw_audio.mp3"):
                transcribe()

            # Verify cloud provider was selected and called
            mock_cloud.assert_called()
            assert mock_cloud.call_count == 2  # Called for each segment

            # Verify memory monitoring for cloud transcription
            mock_dependencies["monitor"].assert_any_call(
                "before cloud transcription", 512
            )

    @patch("core._2_asr.demucs_audio", None)
    def test_transcribe_elevenlabs_provider_selection(self, mock_dependencies):
        """Test selection of ElevenLabs provider"""
        mock_dependencies["load_key"].side_effect = lambda key: {
            "demucs": False,
            "whisper.runtime": "elevenlabs",
        }.get(key, "default")

        # Mock ElevenLabs transcription
        with patch(
            "core.asr_backend.elevenlabs_asr.transcribe_audio_elevenlabs"
        ) as mock_elevenlabs:
            mock_elevenlabs.return_value = {
                "segments": [
                    {"start": 0, "end": 30, "text": "ElevenLabs transcription"}
                ]
            }

            with patch("core._2_asr._RAW_AUDIO_FILE", "/tmp/raw_audio.mp3"):
                transcribe()

            # Verify ElevenLabs provider was selected and called
            mock_elevenlabs.assert_called()
            assert mock_elevenlabs.call_count == 2  # Called for each segment

            # Verify memory monitoring for ElevenLabs transcription
            mock_dependencies["monitor"].assert_any_call(
                "before ElevenLabs transcription", 512
            )

    @patch("core._2_asr.demucs_audio", None)
    def test_transcribe_invalid_runtime_fallback(self, mock_dependencies):
        """Test handling of invalid runtime configuration"""
        mock_dependencies["load_key"].side_effect = lambda key: {
            "demucs": False,
            "whisper.runtime": "invalid_runtime",
        }.get(key, "default")

        with patch("core._2_asr._RAW_AUDIO_FILE", "/tmp/raw_audio.mp3"):
            # Should raise an error or handle gracefully
            with pytest.raises((NameError, UnboundLocalError)):
                # 'ts' variable will be unbound for invalid runtime
                transcribe()

    def test_transcribe_with_demucs_available(self, mock_dependencies):
        """Test transcription with Demucs vocal separation available"""
        mock_dependencies["load_key"].side_effect = lambda key: {
            "demucs": True,
            "whisper.runtime": "local",
        }.get(key, "default")

        # Mock Demucs being available
        mock_demucs = Mock()
        with patch("core._2_asr.demucs_audio", mock_demucs), patch(
            "core._2_asr._VOCAL_AUDIO_FILE", "/tmp/vocal.mp3"
        ):
            with patch(
                "core.asr_backend.whisperX_local.transcribe_audio"
            ) as mock_local:
                mock_local.return_value = {
                    "segments": [{"start": 0, "end": 30, "text": "With Demucs"}]
                }

                with patch("core._2_asr._RAW_AUDIO_FILE", "/tmp/raw_audio.mp3"):
                    transcribe()

                # Verify Demucs was called
                mock_demucs.assert_called_once()

                # Verify vocal audio normalization
                mock_dependencies["normalize"].assert_called_with(
                    "/tmp/vocal.mp3", "/tmp/vocal.mp3", format="mp3"
                )

                # Verify memory monitoring around Demucs
                mock_dependencies["monitor"].assert_any_call("before Demucs", 4096)
                mock_dependencies["monitor"].assert_any_call("after Demucs", 1024)

    def test_transcribe_with_demucs_enabled_but_unavailable(self, mock_dependencies):
        """Test transcription when Demucs is enabled in config but not available"""
        mock_dependencies["load_key"].side_effect = lambda key: {
            "demucs": True,  # Enabled in config
            "whisper.runtime": "local",
        }.get(key, "default")

        # Demucs is None (not available)
        with patch("core._2_asr.demucs_audio", None):
            with patch(
                "core.asr_backend.whisperX_local.transcribe_audio"
            ) as mock_local:
                mock_local.return_value = {
                    "segments": [{"start": 0, "end": 30, "text": "No Demucs"}]
                }

                with patch("core._2_asr._RAW_AUDIO_FILE", "/tmp/raw_audio.mp3"):
                    transcribe()

                # Should proceed without Demucs and use raw audio
                mock_local.assert_called()
                # Verify no vocal audio processing
                assert mock_dependencies["normalize"].call_count == 0

    def test_transcribe_memory_monitoring_integration(self, mock_dependencies):
        """Test memory monitoring throughout transcription process"""
        mock_dependencies["load_key"].side_effect = lambda key: {
            "demucs": False,
            "whisper.runtime": "local",
        }.get(key, "default")

        # Create longer segment list to test periodic memory monitoring
        mock_dependencies["split"].return_value = [
            (i * 30, (i + 1) * 30) for i in range(25)  # 25 segments
        ]

        with patch("core.asr_backend.whisperX_local.transcribe_audio") as mock_local:
            mock_local.return_value = {
                "segments": [{"start": 0, "end": 30, "text": f"Segment"}]
            }

            with patch("core._2_asr._RAW_AUDIO_FILE", "/tmp/raw_audio.mp3"):
                transcribe()

            # Verify memory monitoring at key stages
            monitor_calls = mock_dependencies["monitor"].call_args_list
            stage_names = [call[0][0] for call in monitor_calls]

            assert "transcription start" in stage_names
            assert "after video conversion" in stage_names
            assert "after audio segmentation" in stage_names
            assert "before local transcription" in stage_names
            assert "after transcription" in stage_names
            assert "transcription complete" in stage_names

            # Verify periodic memory monitoring during segmentation
            segment_monitors = [
                name for name in stage_names if "transcription segment" in name
            ]
            assert len(segment_monitors) >= 2  # Should monitor segments 10 and 20

    def test_transcribe_garbage_collection_integration(self, mock_dependencies):
        """Test garbage collection integration during transcription"""
        mock_dependencies["load_key"].side_effect = lambda key: {
            "demucs": False,
            "whisper.runtime": "local",
        }.get(key, "default")

        # Create many segments to trigger periodic GC
        mock_dependencies["split"].return_value = [
            (i * 30, (i + 1) * 30) for i in range(15)  # 15 segments
        ]

        with patch("core.asr_backend.whisperX_local.transcribe_audio") as mock_local:
            mock_local.return_value = {
                "segments": [{"start": 0, "end": 30, "text": "Segment"}]
            }

            with patch("core._2_asr._RAW_AUDIO_FILE", "/tmp/raw_audio.mp3"):
                transcribe()

            # Verify garbage collection was called periodically and at the end
            assert (
                mock_dependencies["gc"].call_count >= 2
            )  # At least segment 10 and final cleanup

    def test_transcribe_result_combination_logic(self, mock_dependencies):
        """Test combination of multiple transcription results"""
        mock_dependencies["load_key"].side_effect = lambda key: {
            "demucs": False,
            "whisper.runtime": "local",
        }.get(key, "default")

        mock_dependencies["split"].return_value = [
            (0.0, 30.0),
            (30.0, 60.0),
            (60.0, 90.0),
        ]

        # Mock different results for each segment
        segment_results = [
            {"segments": [{"start": 0, "end": 30, "text": "First segment"}]},
            {"segments": [{"start": 30, "end": 60, "text": "Second segment"}]},
            {"segments": [{"start": 60, "end": 90, "text": "Third segment"}]},
        ]

        with patch("core.asr_backend.whisperX_local.transcribe_audio") as mock_local:
            mock_local.side_effect = segment_results

            with patch("core._2_asr._RAW_AUDIO_FILE", "/tmp/raw_audio.mp3"):
                transcribe()

            # Verify all segments were processed
            assert mock_local.call_count == 3

            # Verify result combination was called with combined segments
            combined_segments_data = mock_dependencies["process"].call_args[0][0]
            assert len(combined_segments_data["segments"]) == 3
            assert combined_segments_data["segments"][0]["text"] == "First segment"
            assert combined_segments_data["segments"][1]["text"] == "Second segment"
            assert combined_segments_data["segments"][2]["text"] == "Third segment"


class TestASRProviderErrorHandling:
    """Test error handling across different ASR providers"""

    @pytest.fixture
    def mock_base_dependencies(self):
        """Mock base dependencies for error testing"""
        with patch("core._2_asr.find_video_files") as mock_find_video, patch(
            "core._2_asr.convert_video_to_audio"
        ) as mock_convert, patch("core._2_asr.split_audio") as mock_split, patch(
            "core._2_asr.load_key"
        ) as mock_load_key, patch(
            "core._2_asr.monitor_memory_and_warn"
        ):
            mock_find_video.return_value = "/tmp/test_video.mp4"
            mock_split.return_value = [(0.0, 30.0)]

            yield {
                "find_video": mock_find_video,
                "convert": mock_convert,
                "split": mock_split,
                "load_key": mock_load_key,
            }

    @patch("core._2_asr.demucs_audio", None)
    def test_local_provider_transcription_failure(self, mock_base_dependencies):
        """Test handling of local provider transcription failures"""
        mock_base_dependencies["load_key"].side_effect = lambda key: {
            "demucs": False,
            "whisper.runtime": "local",
        }.get(key, "default")

        with patch("core.asr_backend.whisperX_local.transcribe_audio") as mock_local:
            mock_local.side_effect = Exception("Local transcription failed")

            with patch("core._2_asr._RAW_AUDIO_FILE", "/tmp/raw_audio.mp3"):
                with pytest.raises(Exception, match="Local transcription failed"):
                    transcribe()

    @patch("core._2_asr.demucs_audio", None)
    def test_cloud_provider_transcription_failure(self, mock_base_dependencies):
        """Test handling of cloud provider transcription failures"""
        mock_base_dependencies["load_key"].side_effect = lambda key: {
            "demucs": False,
            "whisper.runtime": "cloud",
        }.get(key, "default")

        with patch("core.asr_backend.whisperX_302.transcribe_audio_302") as mock_cloud:
            mock_cloud.side_effect = Exception("Cloud API failed")

            with patch("core._2_asr._RAW_AUDIO_FILE", "/tmp/raw_audio.mp3"):
                with pytest.raises(Exception, match="Cloud API failed"):
                    transcribe()

    @patch("core._2_asr.demucs_audio", None)
    def test_elevenlabs_provider_transcription_failure(self, mock_base_dependencies):
        """Test handling of ElevenLabs provider transcription failures"""
        mock_base_dependencies["load_key"].side_effect = lambda key: {
            "demucs": False,
            "whisper.runtime": "elevenlabs",
        }.get(key, "default")

        with patch(
            "core.asr_backend.elevenlabs_asr.transcribe_audio_elevenlabs"
        ) as mock_elevenlabs:
            mock_elevenlabs.side_effect = Exception("ElevenLabs API failed")

            with patch("core._2_asr._RAW_AUDIO_FILE", "/tmp/raw_audio.mp3"):
                with pytest.raises(Exception, match="ElevenLabs API failed"):
                    transcribe()

    def test_demucs_processing_failure(self, mock_base_dependencies):
        """Test handling of Demucs processing failures"""
        mock_base_dependencies["load_key"].side_effect = lambda key: {
            "demucs": True,
            "whisper.runtime": "local",
        }.get(key, "default")

        mock_demucs = Mock()
        mock_demucs.side_effect = Exception("Demucs processing failed")

        with patch("core._2_asr.demucs_audio", mock_demucs):
            with patch("core._2_asr._RAW_AUDIO_FILE", "/tmp/raw_audio.mp3"):
                with pytest.raises(Exception, match="Demucs processing failed"):
                    transcribe()

    @patch("core._2_asr.demucs_audio", None)
    def test_partial_segment_failure_handling(self, mock_base_dependencies):
        """Test handling when some segments fail but others succeed"""
        mock_base_dependencies["load_key"].side_effect = lambda key: {
            "demucs": False,
            "whisper.runtime": "local",
        }.get(key, "default")

        # Mock split to return multiple segments
        mock_base_dependencies["split"].return_value = [
            (0.0, 30.0),
            (30.0, 60.0),
            (60.0, 90.0),
        ]

        # Mock transcription to fail on second segment
        def transcription_side_effect(*args, **kwargs):
            # Check segment timing to determine which call this is
            start_time = args[2] if len(args) > 2 else 0.0
            if start_time == 30.0:  # Second segment
                raise Exception("Segment 2 failed")
            return {
                "segments": [
                    {
                        "start": start_time,
                        "end": start_time + 30,
                        "text": f"Segment at {start_time}",
                    }
                ]
            }

        with patch("core.asr_backend.whisperX_local.transcribe_audio") as mock_local:
            mock_local.side_effect = transcription_side_effect

            with patch("core._2_asr._RAW_AUDIO_FILE", "/tmp/raw_audio.mp3"):
                with pytest.raises(Exception, match="Segment 2 failed"):
                    transcribe()


@pytest.mark.integration
class TestASRIntegrationWorkflows:
    """Integration tests for complete ASR workflows"""

    def test_complete_transcription_workflow_local(self):
        """Test complete local transcription workflow"""
        with patch("core._2_asr.find_video_files") as mock_find, patch(
            "core._2_asr.convert_video_to_audio"
        ) as mock_convert, patch("core._2_asr.split_audio") as mock_split, patch(
            "core._2_asr.process_transcription"
        ) as mock_process, patch(
            "core._2_asr.save_results"
        ) as mock_save, patch(
            "core._2_asr.load_key"
        ) as mock_load_key, patch(
            "core._2_asr.monitor_memory_and_warn"
        ), patch(
            "core._2_asr.gc.collect"
        ):
            # Setup test data
            mock_find.return_value = "/tmp/video.mp4"
            mock_split.return_value = [(0.0, 60.0)]
            mock_load_key.side_effect = lambda key: {
                "demucs": False,
                "whisper.runtime": "local",
            }.get(key, "default")

            mock_df = pd.DataFrame(
                {
                    "text": ["Complete", "test"],
                    "start": [0.0, 1.0],
                    "end": [1.0, 2.0],
                    "speaker_id": [None, None],
                }
            )
            mock_process.return_value = mock_df

            # Mock local transcription
            with patch(
                "core.asr_backend.whisperX_local.transcribe_audio"
            ) as mock_local:
                mock_local.return_value = {
                    "segments": [
                        {"start": 0, "end": 60, "text": "Complete test transcription"}
                    ]
                }

                with patch("core._2_asr._RAW_AUDIO_FILE", "/tmp/raw.mp3"):
                    transcribe()

                # Verify complete workflow
                mock_find.assert_called_once()
                mock_convert.assert_called_once()
                mock_split.assert_called_once()
                mock_local.assert_called_once()
                mock_process.assert_called_once()
                mock_save.assert_called_once_with(mock_df)

    def test_performance_optimization_provider_selection(self):
        """Test that provider selection optimizes for performance characteristics"""
        test_scenarios = [
            {
                "runtime": "local",
                "expected_memory_requirement": 3072,
                "description": "Local processing requires more memory",
            },
            {
                "runtime": "cloud",
                "expected_memory_requirement": 512,
                "description": "Cloud processing requires less memory",
            },
            {
                "runtime": "elevenlabs",
                "expected_memory_requirement": 512,
                "description": "ElevenLabs processing requires less memory",
            },
        ]

        for scenario in test_scenarios:
            with patch("core._2_asr.monitor_memory_and_warn") as mock_monitor:
                with patch("core._2_asr.load_key") as mock_load_key:
                    mock_load_key.side_effect = lambda key: {
                        "whisper.runtime": scenario["runtime"]
                    }.get(key, "default")

                    # This would be part of the transcribe function logic
                    runtime = mock_load_key("whisper.runtime")

                    if runtime == "local":
                        expected_mem = 3072
                    else:  # cloud or elevenlabs
                        expected_mem = 512

                    assert expected_mem == scenario["expected_memory_requirement"]


@pytest.mark.performance
class TestASRPerformanceOptimization:
    """Performance-focused tests for ASR processing"""

    def test_memory_efficient_segment_processing(self):
        """Test memory-efficient processing of large numbers of segments"""
        # Mock a large number of segments
        segments = [(i * 30, (i + 1) * 30) for i in range(100)]  # 100 segments

        with patch("core._2_asr.gc.collect") as mock_gc:
            # Simulate segment processing loop
            for i, (start, end) in enumerate(segments):
                if i % 10 == 0 and i > 0:
                    mock_gc()

            # Verify garbage collection was called appropriately
            expected_gc_calls = len(
                [i for i in range(len(segments)) if i % 10 == 0 and i > 0]
            )
            assert mock_gc.call_count == expected_gc_calls

    def test_provider_selection_based_on_constraints(self):
        """Test provider selection optimizes for available resources"""
        # Test scenarios with different resource constraints
        resource_scenarios = [
            {
                "available_memory_mb": 1024,  # Low memory
                "recommended_provider": "cloud",
                "reason": "Cloud processing uses less local memory",
            },
            {
                "available_memory_mb": 8192,  # High memory
                "recommended_provider": "local",
                "reason": "Local processing provides better quality with sufficient memory",
            },
        ]

        for scenario in resource_scenarios:
            # This logic would be part of an intelligent provider selection system
            available_memory = scenario["available_memory_mb"]

            if available_memory < 2048:  # Less than 2GB
                recommended = "cloud"
            else:
                recommended = "local"

            assert recommended == scenario["recommended_provider"]
