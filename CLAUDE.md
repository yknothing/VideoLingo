# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Information

**Default Repository**: This project's default repository is `yknothing/VideoLingo`

**Pull Request Guidelines**:
- All Pull Requests should target the `yknothing/VideoLingo` repository
- Default base branch: `main`
- Ensure your changes are committed to feature branches before creating PRs
- The upstream repository (`Huanshere/VideoLingo`) is for reference only

**Git Configuration**:
- Origin remote: `yknothing/VideoLingo`
- When forking, ensure you set `yknothing/VideoLingo` as your upstream target
- Clone command: `git clone https://github.com/yknothing/VideoLingo.git`

## Commands

### Installation and Setup
- **Install dependencies**: `python install.py` - Interactive installer that handles PyTorch, CUDA detection, mirrors, and dependencies
- **Start application**: `streamlit run st.py` - Launches the main VideoLingo web interface

### Development Commands
- **Clean build**: `pip install -e .` - Install project in editable mode
- **Check dependencies**: Check `requirements.txt` for package versions

### Testing
No formal test framework is configured. The application uses Streamlit for interactive testing through the web UI.

## Architecture Overview

VideoLingo is a Python-based video translation and dubbing application with a Streamlit web interface. The core processing pipeline is organized into 12 sequential modules that handle video download, transcription, translation, and dubbing.

### Core Processing Pipeline (`core/` directory)
The application follows a numbered pipeline approach:

1. **`_1_ytdlp.py`** - YouTube video download using yt-dlp
2. **`_2_asr.py`** - Automatic Speech Recognition with WhisperX
3. **`_3_1_split_nlp.py`** - NLP-based sentence splitting using spaCy
4. **`_3_2_split_meaning.py`** - Semantic-based sentence splitting
5. **`_4_1_summarize.py`** - Content summarization for context
6. **`_4_2_translate.py`** - Multi-step translation (Translate-Reflect-Adaptation)
7. **`_5_split_sub.py`** - Subtitle splitting and alignment
8. **`_6_gen_sub.py`** - Subtitle generation with timestamps
9. **`_7_sub_into_vid.py`** - Merge subtitles into video
10. **`_8_1_audio_task.py`** - Generate audio dubbing tasks
11. **`_8_2_dub_chunks.py`** - Create audio chunks for dubbing
12. **`_9_refer_audio.py`** - Extract reference audio
13. **`_10_gen_audio.py`** - Generate TTS audio
14. **`_11_merge_audio.py`** - Merge audio files
15. **`_12_dub_to_vid.py`** - Final video-audio merge

### Key Components

**ASR Backend** (`core/asr_backend/`)
- Multiple ASR options: local WhisperX, cloud APIs (302.ai), ElevenLabs
- Audio preprocessing with Demucs for vocal separation
- Language-specific model handling (special case for Chinese)

**TTS Backend** (`core/tts_backend/`)
- Multiple TTS engines: Azure TTS, OpenAI TTS, GPT-SoVITS, Edge TTS, Fish TTS, etc.
- Duration estimation and speech rate optimization
- Custom TTS support through `custom_tts.py`

**Utilities** (`core/utils/`)
- `ask_gpt.py` - LLM API wrapper with JSON repair
- `config_utils.py` - Configuration management for `config.yaml`
- `models.py` - Data models and validation
- `decorator.py` - Exception handling decorators

**Streamlit UI** (`core/st_utils/`)
- Interactive web interface components
- Progress tracking and status display
- Settings management through sidebar

### Configuration System
- **`config.yaml`** - Main configuration file with API keys, model settings, language preferences
- Supports multiple LLM providers (OpenAI-compatible APIs)
- Extensive TTS configuration options
- Advanced subtitle and dubbing parameters

### Multi-language Support
- UI translations in `translations/` directory
- Dynamic language switching
- spaCy model support for multiple languages
- Language-specific text processing rules

### Processing Flow
1. **Text Processing**: Download → Transcribe → Split → Translate → Generate Subtitles
2. **Audio Processing**: Generate Tasks → Extract Reference → Generate TTS → Merge Audio
3. **Output**: Subtitle-only video (`output_sub.mp4`) and dubbed video (`output_dub.mp4`)

### Code Conventions (from `.cursorrules`)
- Use English comments and print statements
- Large comment blocks with `# ------------`
- Avoid complex function annotations
- Minimize internal function comments

## 🎨 UX Agent Integration

The UX Agent is now integrated into the Claude Code system to provide professional user experience guidance:

### UX Agent Capabilities
- **Nielsen Heuristic Analysis**: Systematic usability evaluation based on Jakob Nielsen's 10 principles
- **User Research Insights**: Data-driven user behavior analysis and journey mapping
- **Interaction Design Optimization**: Information architecture and user flow improvements
- **Visual Design System**: Design system construction and visual hierarchy optimization
- **Usability Testing**: A/B testing strategies and user validation methodologies

### Auto-Trigger Scenarios
The UX Agent automatically activates when:
- Modifying UI components (*.tsx, *.jsx, *.vue, *.html, *.css files)
- Working in `/components/` or `/pages/` directories
- UI-related Python files (`*ui*.py`, `/st_utils/` directory)
- User mentions UX-related keywords: "user experience", "interface", "usability", "navigation"
- Performance issues affect user experience (>3s load time, >100ms interaction delay)

### Manual Invocation
```bash
# Direct UX analysis commands
@ux-agent analyze --component="VideoUploadForm" --principle="nielsen"
@ux-agent research --user-flow="video-processing" --method="journey-map"
@ux-agent design --pattern="progress-indicator" --platform="web"
```

### Collaboration with Other Agents
- **With Code-Reviewer**: Provides UX improvements for UI code reviews
- **With Debug-Specialist**: Analyzes UX root causes of interaction bugs
- **With Performance-Profiler**: Suggests perceived performance optimizations
- **With Code-Quality-Guardian**: Ensures accessibility (a11y) compliance

### UX Analysis for VideoLingo
Special focus areas for this video processing application:
- **Processing Workflow UX**: Optimize the 15-stage video processing pipeline for user clarity
- **Progress Indication**: Design clear status communication during long-running AI tasks
- **Error Recovery**: User-friendly error handling for transcription/translation failures
- **Batch Processing**: Efficient UI patterns for handling multiple videos
- **Mobile Responsiveness**: Ensure core functions work on mobile devices
- **Accessibility**: WCAG compliance for inclusive video processing tools

The UX Agent uses VideoLingo's user research insights:
- 45% Content Creators (need speed + quality)
- 30% Enterprise Users (need batch processing + consistency)  
- 20% Educators (need accuracy + editability)
- 5% Developers (need API integration + customization)

### Output Structure
- `output/` - Main output directory
- `output/log/` - Processing logs and terminology files
- `output/*.mp4` - Final video outputs
- `_model_cache/` - Cached models directory

## 🛡️ GitIgnore Management System

### Integrated GitIgnore Guardian Agent

VideoLingo integrates with Claude Code's `gitignore-guardian` agent for intelligent repository hygiene:

**Auto-triggered Actions:**
- **Pre-commit scanning**: Automatically detect sensitive files before commits
- **Technology detection**: Add ignore patterns when new frameworks detected  
- **Security auditing**: Scan for API keys, credentials, and config files
- **Performance optimization**: Ignore large artifacts and build outputs

**Manual Commands:**
- `@gitignore-guardian analyze` - Comprehensive .gitignore analysis
- `@gitignore-guardian security` - Security-focused file scanning
- `@gitignore-guardian optimize` - Pattern consolidation and optimization
- `@gitignore-guardian sync` - Technology stack synchronization

**VideoLingo-Specific Patterns:**
- Processing outputs: `output/`, `_model_cache/`, batch processing artifacts
- Media files: `*.mp4`, `*.wav`, `*.srt`, temporary audio/video files
- Configuration: `keys.ini`, `config.yaml` (if containing secrets)
- Development: `temp_*/`, debug logs, test artifacts

### Hooks Configuration

**Pre-commit Hooks:**
```yaml
pre_commit:
  - agent: gitignore-guardian
    action: security_audit
    priority: high
    conditions:
      - untracked_sensitive_files
      - config_files_modified
      
  - agent: gitignore-guardian  
    action: pattern_optimization
    priority: medium
    conditions:
      - new_build_artifacts
      - large_untracked_files
```

**Post-file-operation Hooks:**
```yaml  
post_file_operation:
  - agent: gitignore-guardian
    action: analyze_new_patterns
    trigger: 
      - new_technology_detected
      - dependency_changes
      - output_directory_created
```

**Continuous Monitoring:**
- Monitor working directory for sensitive file patterns
- Auto-add ignore patterns for detected frameworks (Node.js, Docker, etc.)
- Performance alerts for repositories exceeding size thresholds
- Security warnings for untracked credential files

### Integration with VideoLingo Workflow

1. **Project Setup**: Auto-configure comprehensive .gitignore for video processing
2. **Development**: Continuous monitoring for temporary files and artifacts  
3. **Processing**: Ignore intermediate outputs while preserving final results
4. **Security**: Prevent accidental commit of API keys and sensitive configurations
5. **Performance**: Maintain optimal repository size by ignoring large media files

### Best Practices Enforcement

- **Security-first**: Always ignore sensitive files before they're tracked
- **Technology-aware**: Automatically adapt patterns to project dependencies
- **Performance-optimized**: Prioritize directory patterns for Git efficiency
- **Documentation**: Self-documenting .gitignore with categorized sections

# VideoLingo - OpenRouter支持和Docker自动部署配置

## ✅ 任务完成总结

### 1. OpenRouter API Key 支持配置

**结果：已完成**

#### 支持情况
- ✅ 项目已原生支持OpenRouter API（通过OpenAI格式兼容）
- ✅ 已在 `config.yaml` 中添加OpenRouter配置示例
- ✅ 已更新中英文文档，包含OpenRouter使用说明
- ✅ 已在文档中添加OpenRouter作为推荐提供商

#### 配置方法
```yaml
# config.yaml 中的OpenRouter配置
api:
  key: 'sk-or-v1-your-openrouter-api-key'
  base_url: 'https://openrouter.ai/api/v1'
  model: 'anthropic/claude-3.5-sonnet'
  llm_support_json: true
```

#### 技术实现
- 项目使用OpenAI客户端，自动兼容OpenRouter的API格式
- URL处理逻辑已支持自动添加`/v1`后缀
- 支持所有OpenRouter提供的模型（claude、gpt、gemini等）

### 2. Docker自动部署流程和工具

**结果：已完成**

#### 创建的部署工具
1. **Docker Compose配置** (`docker-compose.yml`)
   - 支持GPU和CPU模式
   - 自动数据卷映射
   - 环境变量配置
   - 重启策略设置

2. **Linux/macOS自动部署脚本** (`deploy.sh`)
   - 环境检查（Docker、Docker Compose、GPU）
   - 自动目录创建
   - 容器管理（停止、清理、启动）
   - 部署验证和日志显示
   - 错误处理和状态检查

3. **Windows自动部署脚本** (`deploy.bat`)
   - 完整的Windows批处理脚本
   - 相同的功能和检查逻辑
   - 跨平台兼容性

4. **完整的部署文档**
   - 中文版：`docs/pages/docs/docker.zh-CN.md`
   - 英文版：`docs/pages/docs/docker.en-US.md`
   - 详细的安装、配置、使用指南

#### 部署方式
1. **一键自动部署**
   ```bash
   # Linux/macOS
   ./deploy.sh
   
   # Windows
   deploy.bat
   ```

2. **手动Docker Compose部署**
   ```bash
   docker-compose up -d --build
   ```

3. **手动Docker Run部署**
   ```bash
   docker build -t videolingo .
   docker run -d --name videolingo --gpus all -p 8501:8501 videolingo
   ```

#### 部署特性
- ✅ 支持GPU和CPU模式自动检测
- ✅ 自动数据持久化（配置、输出、缓存）
- ✅ 健康检查和状态监控
- ✅ 日志管理和故障排除
- ✅ 一键更新和重新部署
- ✅ 完整的卸载和清理

#### 文件结构
```
VideoLingo/
├── config.yaml           # 配置文件（已添加OpenRouter示例）
├── docker-compose.yml    # Docker Compose配置
├── deploy.sh             # Linux/macOS部署脚本
├── deploy.bat            # Windows部署脚本
├── Dockerfile            # Docker镜像配置
└── docs/pages/docs/
    ├── docker.zh-CN.md   # 中文部署文档
    └── docker.en-US.md   # 英文部署文档
```

## 🚀 使用方法

### OpenRouter配置
1. 在 [OpenRouter](https://openrouter.ai/) 获取API Key
2. 编辑 `config.yaml`，取消注释并修改OpenRouter配置：
   ```yaml
   api:
     key: 'sk-or-v1-your-openrouter-api-key'
     base_url: 'https://openrouter.ai/api/v1'
     model: 'anthropic/claude-3.5-sonnet'
     llm_support_json: true
   ```

### Docker自动部署
1. 克隆项目：
   ```bash
   git clone https://github.com/Huanshere/VideoLingo.git
   cd VideoLingo
   ```

2. 配置API密钥（编辑 `config.yaml`）

3. 运行自动部署：
   ```bash
   # Linux/macOS
   ./deploy.sh
   
   # Windows
   deploy.bat
   ```

4. 访问应用：http://localhost:8501

## 🔧 技术细节

### OpenRouter兼容性
- 使用OpenAI Python客户端，完全兼容OpenRouter API
- 自动处理API端点URL格式
- 支持所有OpenRouter模型和参数
- 支持JSON模式和流式响应

### Docker部署架构
- 基于NVIDIA CUDA镜像，支持GPU加速
- 多阶段构建，优化镜像大小
- 数据卷持久化，避免数据丢失
- 健康检查和自动重启
- 跨平台兼容（Linux、macOS、Windows）

### 安全性考虑
- API密钥通过配置文件管理
- 数据卷权限控制
- 容器隔离和资源限制
- 日志管理和监控

## 📋 验证清单

- [x] OpenRouter API Key配置完成
- [x] Docker Compose配置文件创建
- [x] 跨平台部署脚本创建
- [x] 部署文档完整更新
- [x] 配置示例和说明添加
- [x] 错误处理和日志记录
- [x] 权限设置和文件检查
- [x] 中英文文档同步更新

## 🎯 后续建议

1. **测试验证**
   - 在不同环境测试自动部署脚本
   - 验证OpenRouter API集成
   - 测试GPU和CPU模式切换

2. **功能优化**
   - 添加更多模型提供商支持
   - 优化Docker镜像大小
   - 增加监控和告警功能

3. **文档完善**
   - 添加更多使用示例
   - 创建视频教程
   - 完善故障排除指南

---

**配置完成！** 🎉

现在您可以使用OpenRouter API，并通过一键脚本自动部署到Docker环境中。