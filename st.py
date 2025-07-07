import streamlit as st
import os, sys
from core.st_utils.imports_and_utils import *
from core import *
from core.utils.config_utils import get_storage_paths
from core.utils.video_manager import get_video_manager
from core.utils.path_adapter import get_path_adapter

# SET PATH
current_dir = os.path.dirname(os.path.abspath(__file__))
os.environ['PATH'] += os.pathsep + current_dir
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(page_title="VideoLingo", page_icon="docs/logo.svg")

# Initialize video management system
video_mgr = get_video_manager()
path_adapter = get_path_adapter()

# Get configured output paths
paths = get_storage_paths()
SUB_VIDEO = os.path.join(paths['output'], "output_sub.mp4")
DUB_VIDEO = os.path.join(paths['output'], "output_dub.mp4")

def text_processing_section(current_video_id=None):
    st.header(t("b. Translate and Generate Subtitles"))
    with st.container(border=True):
        if not current_video_id:
            st.warning(t("Please select a video first"))
            return False
            
        st.markdown(f"""
        <p style='font-size: 20px;'>
        {t("This stage includes the following steps:")}
        <p style='font-size: 20px;'>
            1. {t("WhisperX word-level transcription")}<br>
            2. {t("Sentence segmentation using NLP and LLM")}<br>
            3. {t("Summarization and multi-step translation")}<br>
            4. {t("Cutting and aligning long subtitles")}<br>
            5. {t("Generating timeline and subtitles")}<br>
            6. {t("Merging subtitles into the video")}
        """, unsafe_allow_html=True)

        # Check for current video's subtitle output
        current_sub_video = video_mgr.get_output_file(current_video_id, "sub")
        
        if not os.path.exists(current_sub_video):
            if st.button(t("Start Processing Subtitles"), key=f"text_processing_button_{current_video_id}"):
                path_adapter.set_current_video_id(current_video_id)
                process_text()
                st.rerun()
        else:
            if load_key("burn_subtitles"):
                st.video(current_sub_video)
            download_subtitle_zip_button(text=t("Download All Srt Files"))
            
            if st.button(t("Archive to 'history'"), key=f"cleanup_in_text_processing_{current_video_id}"):
                cleanup()
                st.rerun()
            return True

def process_text():
    with st.spinner(t("Using Whisper for transcription...")):
        _2_asr.transcribe()
    with st.spinner(t("Splitting long sentences...")):  
        _3_1_split_nlp.split_by_spacy()
        _3_2_split_meaning.split_sentences_by_meaning()
    with st.spinner(t("Summarizing and translating...")):
        _4_1_summarize.get_summary()
        if load_key("pause_before_translate"):
            input(t("⚠️ PAUSE_BEFORE_TRANSLATE. Go to `output/log/terminology.json` to edit terminology. Then press ENTER to continue..."))
        _4_2_translate.translate_all()
    with st.spinner(t("Processing and aligning subtitles...")): 
        _5_split_sub.split_for_sub_main()
        _6_gen_sub.align_timestamp_main()
    with st.spinner(t("Merging subtitles to video...")):
        _7_sub_into_vid.merge_subtitles_to_video()
    
    st.success(t("Subtitle processing complete! 🎉"))
    st.balloons()

def audio_processing_section(current_video_id=None):
    st.header(t("c. Dubbing"))
    with st.container(border=True):
        if not current_video_id:
            st.warning(t("Please select a video first"))
            return False
            
        # Check if subtitles are processed first
        current_sub_video = video_mgr.get_output_file(current_video_id, "sub")
        if not os.path.exists(current_sub_video):
            st.warning(t("Please process subtitles first"))
            return False
            
        st.markdown(f"""
        <p style='font-size: 20px;'>
        {t("This stage includes the following steps:")}
        <p style='font-size: 20px;'>
            1. {t("Generate audio tasks and chunks")}<br>
            2. {t("Extract reference audio")}<br>
            3. {t("Generate and merge audio files")}<br>
            4. {t("Merge final audio into video")}
        """, unsafe_allow_html=True)
        
        # Check for current video's dubbing output
        current_dub_video = video_mgr.get_output_file(current_video_id, "dub")
        
        if not os.path.exists(current_dub_video):
            if st.button(t("Start Audio Processing"), key=f"audio_processing_button_{current_video_id}"):
                path_adapter.set_current_video_id(current_video_id)
                process_audio()
                st.rerun()
        else:
            st.success(t("Audio processing is complete! You can check the audio files in the `output` folder."))
            if load_key("burn_subtitles"):
                st.video(current_dub_video) 
            if st.button(t("Delete dubbing files"), key=f"delete_dubbing_files_{current_video_id}"):
                delete_dubbing_files()
                st.rerun()
            if st.button(t("Archive to 'history'"), key=f"cleanup_in_audio_processing_{current_video_id}"):
                cleanup()
                st.rerun()

def process_audio():
    with st.spinner(t("Generate audio tasks")): 
        _8_1_audio_task.gen_audio_task_main()
        _8_2_dub_chunks.gen_dub_chunks()
    with st.spinner(t("Extract refer audio")):
        _9_refer_audio.extract_refer_audio_main()
    with st.spinner(t("Generate all audio")):
        _10_gen_audio.gen_audio()
    with st.spinner(t("Merge full audio")):
        _11_merge_audio.merge_full_audio()
    with st.spinner(t("Merge dubbing to the video")):
        _12_dub_to_vid.merge_video_audio()
    
    st.success(t("Audio processing complete! 🎇"))
    st.balloons()

def get_current_video_id():
    """获取当前活跃的视频ID"""
    # 优先返回当前标签页的视频
    if 'download_video_id' in st.session_state:
        return st.session_state.download_video_id
    elif 'upload_video_id' in st.session_state:
        return st.session_state.upload_video_id
    else:
        return None

def main():
    logo_col, _ = st.columns([1,1])
    with logo_col:
        st.image("docs/logo.png", use_column_width=True)
    st.markdown(button_style, unsafe_allow_html=True)
    welcome_text = t("Hello, welcome to VideoLingo. If you encounter any issues, feel free to get instant answers with our Free QA Agent <a href=\"https://share.fastgpt.in/chat/share?shareId=066w11n3r9aq6879r4z0v9rh\" target=\"_blank\">here</a>! You can also try out our SaaS website at <a href=\"https://videolingo.io\" target=\"_blank\">videolingo.io</a> for free!")
    st.markdown(f"<p style='font-size: 20px; color: #808080;'>{welcome_text}</p>", unsafe_allow_html=True)
    
    # 侧边栏设置
    with st.sidebar:
        page_setting()
        st.markdown(give_star_button, unsafe_allow_html=True)
    
    # 获取当前视频ID
    from core.st_utils.video_input_section import render_video_input_section
    current_video_id = render_video_input_section()
    
    # 处理功能部分
    text_processing_section(current_video_id)
    audio_processing_section(current_video_id)

if __name__ == "__main__":
    main()
