import streamlit as st
from translations.translations import translate as t
from translations.translations import DISPLAY_LANGUAGES
from core.utils import *
from core.utils.config_utils import get_system_downloads_dir, _is_path_writable
from core.utils.session_security import (
    get_secure_session_state,
    SessionSecurityContext,
    require_valid_session,
    session_audit_log,
    SessionValidationError,
    SessionIntegrityError,
    session_logger
)
import os
import platform
import subprocess
import shlex
import re
from pathlib import Path


# ------------
# Security utilities
# ------------
def _validate_path(path_str):
    """
    Enhanced path validation with comprehensive traversal attack prevention.
    
    Security measures:
    - URL decoding to prevent encoded bypass (%2e%2e%2f)
    - Windows and Unix traversal pattern detection
    - Symlink resolution and validation
    - Shell metacharacter blocking
    - Allowed directory whitelist validation
    
    Returns (is_valid: bool, sanitized_path: str, error_msg: str)
    """
    import urllib.parse
    
    if not path_str or not isinstance(path_str, str):
        return False, "", "Invalid path format"
    
    try:
        # ------------
        # Step 1: Decode URL-encoded content to prevent bypass attacks
        # ------------
        try:
            decoded_path = urllib.parse.unquote(path_str)
            # Double decode to catch double-encoded attacks
            decoded_path = urllib.parse.unquote(decoded_path)
        except Exception:
            decoded_path = path_str
        
        # ------------
        # Step 2: Remove null bytes and control characters
        # ------------
        sanitized = re.sub(r'[\x00-\x1f\x7f]', '', decoded_path.strip())
        
        # ------------
        # Step 3: Check for malicious patterns (both Unix and Windows)
        # ------------
        dangerous_patterns = [
            r'[;&|`$()]',           # Shell metacharacters
            r'\.\./|\.\.',          # Unix path traversal (..)
            r'\.\.\\',              # Windows path traversal (..\)
            r'%2e%2e',              # URL encoded traversal
            r'%252e%252e',          # Double URL encoded
            r'^\s*[|;&]',           # Command chaining at start
            r'[|;&]\s*$',           # Command chaining at end
            r'\\\\',                # UNC paths (\\server\share)
            r'^\s*~',               # Home directory expansion
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, sanitized, re.IGNORECASE):
                return False, "", "Potentially malicious path pattern detected"
        
        # Also check the original input for encoded attacks
        for pattern in [r'%2e%2e', r'%252e', r'%00']:
            if re.search(pattern, path_str, re.IGNORECASE):
                return False, "", "URL-encoded attack pattern detected"
        
        # ------------
        # Step 4: Convert to Path and resolve symlinks
        # ------------
        path_obj = Path(sanitized)
        
        # Check if path contains symlinks before resolving
        try:
            # Resolve to get the real path (follows symlinks)
            resolved_path = path_obj.resolve()
        except (OSError, RuntimeError) as e:
            return False, "", f"Path resolution error: {str(e)}"
        
        # ------------
        # Step 5: Validate against allowed directories
        # ------------
        allowed_roots = _get_allowed_path_roots()
        
        path_in_allowed = False
        for allowed_root in allowed_roots:
            try:
                resolved_path.relative_to(allowed_root)
                path_in_allowed = True
                break
            except ValueError:
                continue
        
        if not path_in_allowed:
            return False, "", "Path is outside allowed directories"
        
        # ------------
        # Step 6: Existence and type checks
        # ------------
        if not resolved_path.exists():
            return False, str(resolved_path), "Path does not exist"
            
        if not resolved_path.is_dir():
            return False, str(resolved_path), "Path is not a directory"
        
        # ------------
        # Step 7: Symlink safety check - ensure final path is still in allowed area
        # ------------
        if path_obj.is_symlink():
            # Verify the symlink target is also in allowed directories
            symlink_target = resolved_path
            target_in_allowed = False
            for allowed_root in allowed_roots:
                try:
                    symlink_target.relative_to(allowed_root)
                    target_in_allowed = True
                    break
                except ValueError:
                    continue
            
            if not target_in_allowed:
                return False, "", "Symlink target is outside allowed directories"
            
        return True, str(resolved_path), ""
        
    except Exception as e:
        return False, "", f"Path validation error: {str(e)}"


def _get_allowed_path_roots():
    """
    Get list of allowed root directories for path validation.
    Returns list of Path objects representing allowed directories.
    """
    import platform
    
    allowed = []
    
    # Always allow user home directory
    try:
        allowed.append(Path.home().resolve())
    except Exception:
        pass
    
    # Platform-specific allowed directories
    if platform.system() in ("Linux", "Darwin", "FreeBSD", "OpenBSD"):
        # Unix-like systems
        allowed.extend([
            Path("/tmp").resolve() if Path("/tmp").exists() else None,
            Path("/Users").resolve() if Path("/Users").exists() else None,
            Path("/home").resolve() if Path("/home").exists() else None,
        ])
        # macOS external drives
        if platform.system() == "Darwin":
            volumes = Path("/Volumes")
            if volumes.exists():
                allowed.append(volumes.resolve())
        # Linux mount points
        for mount_point in ["/media", "/mnt"]:
            mp = Path(mount_point)
            if mp.exists():
                allowed.append(mp.resolve())
                
    elif platform.system() == "Windows":
        # Windows user directories
        import os
        system_drive = os.environ.get("SYSTEMDRIVE", "C:")
        users_path = Path(f"{system_drive}\\Users")
        if users_path.exists():
            allowed.append(users_path.resolve())
        # Windows temp
        temp_path = Path(f"{system_drive}\\Temp")
        if temp_path.exists():
            allowed.append(temp_path.resolve())
    
    # Also allow project root and output directory
    try:
        project_root = Path(".").resolve()
        allowed.append(project_root)
    except Exception:
        pass
    
    # Filter out None values
    return [p for p in allowed if p is not None]


def _safe_shell_escape(path_str):
    """
    Safely escape path for shell commands using shlex.
    Additional validation for known safe characters only.
    """
    is_valid, sanitized_path, error = _validate_path(path_str)
    if not is_valid:
        raise ValueError(f"Invalid path: {error}")
    
    # Only allow alphanumeric, common path characters
    if not re.match(r'^[a-zA-Z0-9\s\-_./\\:]+$', sanitized_path):
        raise ValueError("Path contains unsafe characters")
    
    return shlex.quote(sanitized_path)


# ------------
# safe helpers
# ------------
def _safe_index(options_list, current_value, default_value=None):
    """Return a valid index for Streamlit selectbox"""
    try:
        return options_list.index(current_value)
    except Exception:
        if default_value is not None and default_value in options_list:
            try:
                return options_list.index(default_value)
            except Exception:
                pass
        return 0


@session_audit_log("config_input")
def config_input(label, key, help=None):
    """Generic config input handler with session security"""
    try:
        with SessionSecurityContext() as secure_session:
            # Validate config key format to prevent injection
            if not _validate_config_key(key):
                st.error(f"🔒 Invalid configuration key: {key}")
                return load_key(key)
            
            # Get current value and user input
            current_val = load_key(key)
            val = st.text_input(label, value=current_val, help=help)
            
            # Validate input before updating
            if val != current_val:
                if _validate_config_value(key, val):
                    update_key(key, val)
                    # Log configuration change
                    session_logger.info(f"Configuration updated: {key}", extra={'session_id': 'config'})
                else:
                    st.warning(f"⚠️ Invalid value for {label}. Value not saved.")
                    return current_val
            
            return val
    except (SessionValidationError, SessionIntegrityError) as e:
        st.error(f"🔒 Session security error in configuration: {str(e)}")
        return load_key(key)


@require_valid_session
@session_audit_log("page_setting")
def page_setting():
    # ------------
    # Display language with fallback index
    # ------------
    _dl_values = list(DISPLAY_LANGUAGES.values())
    _dl_keys = list(DISPLAY_LANGUAGES.keys())
    _dl_current = load_key("display_language")
    _dl_idx = _safe_index(_dl_values, _dl_current, "zh-CN")
    display_language = st.selectbox("Display Language 🌐", options=_dl_keys, index=_dl_idx)
    if DISPLAY_LANGUAGES[display_language] != load_key("display_language"):
        update_key("display_language", DISPLAY_LANGUAGES[display_language])
        st.rerun()

    # with st.expander(t("Youtube Settings"), expanded=True):
    #     config_input(t("Cookies Path"), "youtube.cookies_path")

    with st.expander(t("Storage Settings"), expanded=True):

        # ------------
        # Current Storage Status Display (Visibility of System Status)
        # ------------
        current_path = load_key("video_storage.base_path") or get_system_downloads_dir() or "output"

        # Visual status indicator with clear current path
        if os.path.exists(current_path) and _is_path_writable(current_path):
            status_icon = "✅"
            status_color = "green"
            status_text = t("Ready")
        else:
            status_icon = "⚠️"
            status_color = "orange"
            status_text = t("Needs attention")

        st.markdown(
            f"""
        **{t("Current Storage Location")}:**  
        `{current_path}`  
        <span style="color: {status_color};">{status_icon} {status_text}</span>
        """,
            unsafe_allow_html=True,
        )

        # ------------
        # SECURE Directory Selection Interface
        # ------------
        def open_secure_folder_dialog():
            """
            Secure implementation of folder dialog using only safe, parameterized commands.
            All user input is validated and properly escaped.
            """
            result = {"path": None, "status": "error", "error_msg": None}
            
            try:
                current = load_key("video_storage.base_path") or get_system_downloads_dir() or os.getcwd()
                
                # Validate current path before using it
                is_valid, validated_current, validation_error = _validate_path(current)
                if not is_valid:
                    # Fall back to safe defaults if current path is invalid
                    validated_current = os.getcwd()
                
                sysname = platform.system()
                
                if sysname == "Darwin":
                    # macOS AppleScript - using parameterized approach
                    try:
                        # Use subprocess.run with list arguments to prevent injection
                        escaped_path = _safe_shell_escape(validated_current)
                        
                        applescript_commands = [
                            "osascript", 
                            "-e", f'set defaultFolder to POSIX file {escaped_path}',
                            "-e", 'set theFolder to (choose folder with prompt "Choose storage directory:" default location defaultFolder)',
                            "-e", "POSIX path of theFolder"
                        ]
                        
                        r = subprocess.run(
                            applescript_commands,
                            capture_output=True,
                            text=True,
                            timeout=30  # Add timeout for security
                        )
                        
                        if r.returncode == 0 and r.stdout.strip():
                            selected_path = r.stdout.strip().rstrip("/")
                            # Validate the returned path
                            is_valid, validated_path, error = _validate_path(selected_path)
                            if is_valid:
                                result["path"] = validated_path
                                result["status"] = "success"
                            else:
                                result["error_msg"] = f"Invalid path returned: {error}"
                        elif r.returncode == 1:
                            result["status"] = "cancelled"
                        else:
                            result["error_msg"] = "AppleScript dialog failed"
                            
                    except Exception as e:
                        result["error_msg"] = f"macOS dialog error: {str(e)}"
                        
                elif sysname == "Windows":
                    # Windows PowerShell - using secure parameter passing
                    try:
                        escaped_path = _safe_shell_escape(validated_current)
                        
                        # Use parameter-based PowerShell execution to prevent injection
                        ps_script = """
                        param($InitialPath)
                        Add-Type -AssemblyName System.Windows.Forms
                        $fb = New-Object System.Windows.Forms.FolderBrowserDialog
                        $fb.Description = "Choose storage directory"
                        $fb.SelectedPath = $InitialPath
                        $fb.ShowNewFolderButton = $true
                        $r = $fb.ShowDialog()
                        if ($r -eq [System.Windows.Forms.DialogResult]::OK) {
                            Write-Output $fb.SelectedPath
                            exit 0
                        } else {
                            exit 1
                        }
                        """
                        
                        r = subprocess.run(
                            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps_script, "-InitialPath", validated_current],
                            capture_output=True,
                            text=True,
                            timeout=30
                        )
                        
                        if r.returncode == 0 and r.stdout.strip():
                            selected_path = r.stdout.strip()
                            # Validate the returned path
                            is_valid, validated_path, error = _validate_path(selected_path)
                            if is_valid:
                                result["path"] = validated_path
                                result["status"] = "success"
                            else:
                                result["error_msg"] = f"Invalid path returned: {error}"
                        elif r.returncode == 1:
                            result["status"] = "cancelled"
                        else:
                            result["error_msg"] = "PowerShell dialog failed"
                            
                    except Exception as e:
                        result["error_msg"] = f"Windows dialog error: {str(e)}"
                        
                else:
                    # Linux: Use safe parameter passing for zenity/kdialog
                    try:
                        if subprocess.run(["which", "zenity"], capture_output=True, timeout=5).returncode == 0:
                            escaped_path = _safe_shell_escape(validated_current)
                            
                            r = subprocess.run(
                                [
                                    "zenity",
                                    "--file-selection",
                                    "--directory",
                                    "--title=Choose storage directory",
                                    f"--filename={escaped_path}/"
                                ],
                                capture_output=True,
                                text=True,
                                timeout=30
                            )
                            
                            if r.returncode == 0 and r.stdout.strip():
                                selected_path = r.stdout.strip()
                                # Validate the returned path
                                is_valid, validated_path, error = _validate_path(selected_path)
                                if is_valid:
                                    result["path"] = validated_path
                                    result["status"] = "success"
                                else:
                                    result["error_msg"] = f"Invalid path returned: {error}"
                            elif r.returncode == 1:
                                result["status"] = "cancelled"
                            else:
                                result["error_msg"] = "zenity failed"
                                
                        elif subprocess.run(["which", "kdialog"], capture_output=True, timeout=5).returncode == 0:
                            escaped_path = _safe_shell_escape(validated_current)
                            
                            r = subprocess.run(
                                [
                                    "kdialog",
                                    "--getexistingdirectory",
                                    escaped_path,
                                    "--title",
                                    "Choose storage directory"
                                ],
                                capture_output=True,
                                text=True,
                                timeout=30
                            )
                            
                            if r.returncode == 0 and r.stdout.strip():
                                selected_path = r.stdout.strip()
                                # Validate the returned path
                                is_valid, validated_path, error = _validate_path(selected_path)
                                if is_valid:
                                    result["path"] = validated_path
                                    result["status"] = "success"
                                else:
                                    result["error_msg"] = f"Invalid path returned: {error}"
                            elif r.returncode == 1:
                                result["status"] = "cancelled"
                            else:
                                result["error_msg"] = "kdialog failed"
                        else:
                            result["error_msg"] = "No native file dialog available (zenity/kdialog not found)"
                            
                    except Exception as e:
                        result["error_msg"] = f"Linux dialog error: {str(e)}"
                        
            except Exception as e:
                result["error_msg"] = f"Unexpected error: {str(e)}"
                
            return result

        def update_storage_path(new_path, source_description):
            """Unified path update with comprehensive validation and user feedback"""
            
            # First validate the path securely
            is_valid, validated_path, validation_error = _validate_path(new_path)
            if not is_valid:
                st.error(f"{t('Invalid path')}: {validation_error}")
                return False
                
            # Additional checks for directory accessibility
            if not os.path.isdir(validated_path):
                st.error(t("Selected path is not a valid directory"))
                return False
            elif not _is_path_writable(validated_path):
                st.error(t("Selected path is not writable. Please choose another directory."))
                return False
            else:
                # Update the configuration with validated path
                success = update_key("video_storage.base_path", validated_path)
                if success:
                    st.success(f"✅ {t('Storage location updated to')} {source_description}")
                    st.rerun()
                    return True
                else:
                    st.error("Failed to save configuration")
                    return False

        # ------------
        # Streamlined Action Buttons
        # ------------
        col1, col2 = st.columns(2)

        with col1:
            if st.button(
                "📂 " + t("Choose Custom Directory"), use_container_width=True, type="primary"
            ):
                dlg = open_secure_folder_dialog()
                if dlg and dlg.get("status") == "success" and dlg.get("path"):
                    update_storage_path(dlg["path"], t("custom directory"))
                elif dlg and dlg.get("status") == "cancelled":
                    st.info("📁 " + t("Directory selection cancelled"))
                else:
                    st.error("❌ " + (dlg.get("error_msg") or t("Failed to open directory picker")))

        with col2:
            if st.button("⬇️ " + t("Use Downloads Folder"), use_container_width=True):
                downloads = get_system_downloads_dir()
                if downloads and _is_path_writable(downloads):
                    update_storage_path(downloads, t("system Downloads folder"))
                else:
                    st.error(t("System Downloads folder is not accessible"))

        # ------------
        # Advanced Options (Non-nested approach)
        # ------------
        show_advanced = st.checkbox("🔧 " + t("Advanced Path Options"), key="show_advanced_storage")

        if show_advanced:
            st.caption(t("For advanced users: manually specify a custom path"))

            manual_path = st.text_input(
                t("Manual Path Entry"),
                value=current_path,
                help=t(
                    "Enter a custom directory path manually. Path will be validated when you press Enter."
                ),
                key="manual_path_input",
            )

            adv_col1, adv_col2 = st.columns(2)

            with adv_col1:
                # Auto-validate on Enter/change
                if st.button(t("Apply Manual Path"), key="manual_path_apply"):
                    if manual_path.strip() and manual_path != current_path:
                        update_storage_path(manual_path.strip(), t("manually entered path"))

            with adv_col2:
                # Reset to default option
                if st.button(
                    "🔄 " + t("Reset to Default"),
                    help=t("Reset storage to system Downloads or project folder"),
                    key="reset_default",
                ):
                    downloads = get_system_downloads_dir()
                    if downloads and _is_path_writable(downloads):
                        update_storage_path(downloads, t("default Downloads folder"))
                    else:
                        update_storage_path("output", t("project output folder"))

        # ------------
        # Directory Information Panel
        # ------------
        if os.path.exists(current_path):
            try:
                # Show basic directory info for user confidence
                free_space = (
                    os.statvfs(current_path).f_bavail * os.statvfs(current_path).f_frsize
                    if hasattr(os, "statvfs")
                    else None
                )
                if free_space:
                    free_space_gb = free_space / (1024**3)
                    if free_space_gb < 1:
                        space_warning = "⚠️ " + t("Low disk space")
                        st.warning(
                            f"{space_warning} ({free_space_gb:.1f} GB " + t("available") + ")"
                        )
                    else:
                        st.info(f"💾 {free_space_gb:.1f} GB " + t("available storage space"))
            except:
                pass  # Ignore if we can't get disk space info

    with st.expander(t("LLM Configuration"), expanded=True):
        config_input(t("API_KEY"), "api.key")
        config_input(
            t("BASE_URL"),
            "api.base_url",
            help=t("Openai format, will add /v1/chat/completions automatically"),
        )

        c1, c2 = st.columns([4, 1])
        with c1:
            config_input(t("MODEL"), "api.model", help=t("click to check API validity") + " 👉")
        with c2:
            if st.button("📡", key="api"):
                st.toast(
                    t("API Key is valid") if check_api() else t("API Key is invalid"),
                    icon="✅" if check_api() else "❌",
                )
        llm_support_json = st.toggle(
            t("LLM JSON Format Support"),
            value=load_key("api.llm_support_json"),
            help=t("Enable if your LLM supports JSON mode output"),
        )
        if llm_support_json != load_key("api.llm_support_json"):
            update_key("api.llm_support_json", llm_support_json)
            st.rerun()
    with st.expander(t("Subtitles Settings"), expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            langs = {
                "🇺🇸 English": "en",
                "🇨🇳 简体中文": "zh",
                "🇪🇸 Español": "es",
                "🇷🇺 Русский": "ru",
                "🇫🇷 Français": "fr",
                "🇩🇪 Deutsch": "de",
                "🇮🇹 Italiano": "it",
                "🇯🇵 日本語": "ja",
            }
            # ------------
            # Recognition language with fallback index
            # ------------
            _lang_values = list(langs.values())
            _lang_keys = list(langs.keys())
            _lang_idx = _safe_index(_lang_values, load_key("whisper.language"), "en")
            lang = st.selectbox(t("Recog Lang"), options=_lang_keys, index=_lang_idx)
            if langs[lang] != load_key("whisper.language"):
                update_key("whisper.language", langs[lang])
                st.rerun()

        # ------------
        # Runtime with fallback index
        # ------------
        _runtime_options = ["local", "cloud", "elevenlabs"]
        _rt_idx = _safe_index(_runtime_options, load_key("whisper.runtime"), "local")
        runtime = st.selectbox(
            t("WhisperX Runtime"),
            options=_runtime_options,
            index=_rt_idx,
            help=t(
                "Local runtime requires >8GB GPU, cloud runtime requires 302ai API key, elevenlabs runtime requires ElevenLabs API key"
            ),
        )
        if runtime != load_key("whisper.runtime"):
            update_key("whisper.runtime", runtime)
            st.rerun()
        if runtime == "cloud":
            config_input(t("WhisperX 302ai API"), "whisper.whisperX_302_api_key")
        if runtime == "elevenlabs":
            config_input(("ElevenLabs API"), "whisper.elevenlabs_api_key")

        with c2:
            target_language = st.text_input(
                t("Target Lang"),
                value=load_key("target_language"),
                help=t("Input any language in natural language, as long as llm can understand"),
            )
            if target_language != load_key("target_language"):
                update_key("target_language", target_language)
                st.rerun()

        demucs = st.toggle(
            t("Vocal separation enhance"),
            value=load_key("demucs"),
            help=t(
                "Recommended for videos with loud background noise, but will increase processing time"
            ),
        )
        if demucs != load_key("demucs"):
            update_key("demucs", demucs)
            st.rerun()

        burn_subtitles = st.toggle(
            t("Burn-in Subtitles"),
            value=load_key("burn_subtitles"),
            help=t("Whether to burn subtitles into the video, will increase processing time"),
        )
        if burn_subtitles != load_key("burn_subtitles"):
            update_key("burn_subtitles", burn_subtitles)
            st.rerun()
    with st.expander(t("Dubbing Settings"), expanded=True):
        tts_methods = [
            "azure_tts",
            "openai_tts",
            "fish_tts",
            "sf_fish_tts",
            "edge_tts",
            "gpt_sovits",
            "custom_tts",
            "sf_cosyvoice2",
            "f5tts",
        ]
        _tts_idx = _safe_index(tts_methods, load_key("tts_method"), "openai_tts")
        select_tts = st.selectbox(t("TTS Method"), options=tts_methods, index=_tts_idx)
        if select_tts != load_key("tts_method"):
            update_key("tts_method", select_tts)
            st.rerun()

        # sub settings for each tts method
        if select_tts == "sf_fish_tts":
            config_input(t("SiliconFlow API Key"), "sf_fish_tts.api_key")

            # Add mode selection dropdown
            mode_options = {
                "preset": t("Preset"),
                "custom": t("Refer_stable"),
                "dynamic": t("Refer_dynamic"),
            }
            selected_mode = st.selectbox(
                t("Mode Selection"),
                options=list(mode_options.keys()),
                format_func=lambda x: mode_options[x],
                index=list(mode_options.keys()).index(load_key("sf_fish_tts.mode"))
                if load_key("sf_fish_tts.mode") in mode_options.keys()
                else 0,
            )
            if selected_mode != load_key("sf_fish_tts.mode"):
                update_key("sf_fish_tts.mode", selected_mode)
                st.rerun()
            if selected_mode == "preset":
                config_input("Voice", "sf_fish_tts.voice")

        elif select_tts == "openai_tts":
            config_input("302ai API", "openai_tts.api_key")
            config_input(t("OpenAI Voice"), "openai_tts.voice")

        elif select_tts == "fish_tts":
            config_input("302ai API", "fish_tts.api_key")
            _char_dict = load_key("fish_tts.character_id_dict") or {}
            _char_keys = list(_char_dict.keys()) or ["default"]
            _char_idx = _safe_index(_char_keys, load_key("fish_tts.character"), _char_keys[0])
            fish_tts_character = st.selectbox(
                t("Fish TTS Character"), options=_char_keys, index=_char_idx
            )
            if fish_tts_character != load_key("fish_tts.character"):
                update_key("fish_tts.character", fish_tts_character)
                st.rerun()

        elif select_tts == "azure_tts":
            config_input("302ai API", "azure_tts.api_key")
            config_input(t("Azure Voice"), "azure_tts.voice")

        elif select_tts == "gpt_sovits":
            st.info(t("Please refer to Github homepage for GPT_SoVITS configuration"))
            config_input(t("SoVITS Character"), "gpt_sovits.character")

            refer_mode_options = {
                1: t("Mode 1: Use provided reference audio only"),
                2: t("Mode 2: Use first audio from video as reference"),
                3: t("Mode 3: Use each audio from video as reference"),
            }
            _rm_keys = list(refer_mode_options.keys())
            _rm_current = load_key("gpt_sovits.refer_mode")
            try:
                _rm_current = int(_rm_current) if _rm_current is not None else None
            except Exception:
                _rm_current = None
            _rm_idx = _safe_index(_rm_keys, _rm_current, 1)
            selected_refer_mode = st.selectbox(
                t("Refer Mode"),
                options=_rm_keys,
                format_func=lambda x: refer_mode_options[x],
                index=_rm_idx,
                help=t("Configure reference audio mode for GPT-SoVITS"),
            )
            if selected_refer_mode != load_key("gpt_sovits.refer_mode"):
                update_key("gpt_sovits.refer_mode", selected_refer_mode)
                st.rerun()

        elif select_tts == "edge_tts":
            config_input(t("Edge TTS Voice"), "edge_tts.voice")

        elif select_tts == "sf_cosyvoice2":
            config_input(t("SiliconFlow API Key"), "sf_cosyvoice2.api_key")

        elif select_tts == "f5tts":
            config_input("302ai API", "f5tts.302_api")


@require_valid_session
@session_audit_log("check_api")
def check_api():
    """Check API with secure session management"""
    try:
        with SessionSecurityContext() as secure_session:
            # Check if API test is already in progress to prevent concurrent calls
            if secure_session.get("api_test_in_progress", False):
                return False
            
            secure_session["api_test_in_progress"] = True
            
            try:
                resp = ask_gpt(
                    "This is a test, response 'message':'success' in json format.",
                    resp_type="json",
                    log_title="None",
                )
                result = resp.get("message") == "success"
                
                # Log API test result
                session_logger.info(f"API test result: {result}", extra={'session_id': 'api_test'})
                return result
                
            finally:
                secure_session["api_test_in_progress"] = False
                
    except (SessionValidationError, SessionIntegrityError) as e:
        session_logger.warning(f"Session error during API test: {e}")
        return False
    except Exception as e:
        session_logger.error(f"API test failed: {e}")
        return False


# Security validation functions for configuration
def _validate_config_key(key: str) -> bool:
    """Validate configuration key format to prevent injection attacks"""
    if not key or not isinstance(key, str):
        return False
    
    # Allow only alphanumeric characters, dots, and underscores
    import re
    if not re.match(r'^[a-zA-Z0-9._-]+$', key):
        return False
    
    # Prevent path traversal patterns
    if '..' in key or key.startswith('/') or key.startswith('\\'):
        return False
    
    # Limit key length
    if len(key) > 100:
        return False
    
    return True


def _validate_config_value(key: str, value: str) -> bool:
    """Validate configuration value based on key type"""
    if not isinstance(value, str):
        return False
    
    # Limit value length
    if len(value) > 1000:
        session_logger.warning(f"Config value too long for key: {key}")
        return False
    
    # Special validation for sensitive keys
    if 'api' in key.lower() and 'key' in key.lower():
        # API key validation
        if len(value) < 8 or len(value) > 200:
            return False
        # Check for common invalid patterns
        if value in ['', 'your-api-key', 'sk-xxx', 'replace-with-your-key']:
            return False
    
    if 'url' in key.lower():
        # URL validation
        import re
        url_pattern = r'^https?://[^\s/$.?#].[^\s]*$'
        if value and not re.match(url_pattern, value):
            return False
    
    if 'path' in key.lower():
        # Path validation
        if '..' in value or value.startswith('//'):
            return False
    
    # Check for potential injection patterns
    dangerous_patterns = [
        r'[;&|`$(){}[\]\\]',  # Shell metacharacters
        r'<script',           # Script injection
        r'javascript:',       # JavaScript injection
        r'data:',            # Data URI injection
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, value, re.IGNORECASE):
            session_logger.warning(f"Dangerous pattern detected in config value for key: {key}")
            return False
    
    return True


if __name__ == "__main__":
    check_api()
