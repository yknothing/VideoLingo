# VideoLingo 新架构设计文档

## 🏗️ 架构概述

VideoLingo 已升级为基于视频ID的文件管理架构，实现了安全、可追溯的文件组织系统。

### 核心原则
1. **基于ID的文件映射** - 每个视频有唯一ID，所有相关文件都通过ID关联
2. **1:1:n 映射关系** - 一个原始视频 → 一个temp文件夹 → 多个输出文件
3. **禁止删除，允许覆盖** - 不删除用户数据，只允许同视频ID覆盖
4. **完全隔离** - 不同视频的处理过程互不干扰

## 📁 文件组织结构

### 新架构文件布局
```
video_storage/
├── input/                          # 原始视频文件
│   ├── abc12345_def6_789012.mp4   # 格式: {video_id}.{ext}
│   └── xyz98765_abc1_234567.mp4
├── temp/                           # 按视频ID组织的临时文件
│   ├── abc12345_def6_789012/       # 视频ID目录
│   │   ├── .metadata.json         # 视频元数据
│   │   ├── log/                    # 处理日志
│   │   ├── gpt_log/               # GPT交互日志
│   │   ├── audio/                 # 音频处理文件
│   │   │   ├── raw.mp3
│   │   │   ├── vocal.mp3
│   │   │   ├── segs/              # 音频片段
│   │   │   └── refers/            # 参考音频
│   │   ├── subtitles.srt          # 字幕文件
│   │   ├── translation.json       # 翻译结果
│   │   └── logs/                  # 操作日志
│   └── xyz98765_abc1_234567/
├── output/                         # 最终输出文件
│   ├── abc12345_def6_789012_sub.mp4      # 带字幕版本
│   ├── abc12345_def6_789012_dub.mp4      # 配音版本
│   ├── abc12345_def6_789012_summary.txt  # 总结文件
│   └── xyz98765_abc1_234567_sub.mp4
└── history/                        # 历史存档 (可选)
    └── completed_videos/
```

### 视频ID生成规则
- **格式**: `{content_hash}_{name_hash}_{timestamp}`
- **示例**: `a1b2c3d4_ef56_789012`
- **组成**:
  - `content_hash`: 文件内容前1MB的MD5哈希 (8位)
  - `name_hash`: 文件名的MD5哈希 (4位)
  - `timestamp`: 时间戳后6位

## 🔧 核心组件

### 1. VideoFileManager (`core/utils/video_manager.py`)
```python
# 主要功能
- generate_video_id()      # 生成唯一视频ID
- register_video()         # 注册新视频到系统
- get_video_paths()        # 获取视频相关路径
- safe_overwrite_*()       # 安全覆盖操作
```

### 2. PathAdapter (`core/utils/path_adapter.py`)
```python
# 为现有代码提供兼容层
- get_temp_file()          # 获取临时文件路径
- get_output_file()        # 获取输出文件路径
- get_audio_dir()          # 获取音频目录
- reset_for_new_video()    # 重置视频上下文
```

### 3. 安全操作原则
```python
# ✅ 允许的操作
video_mgr.safe_overwrite_temp_files(video_id)    # 覆盖临时文件
video_mgr.safe_overwrite_output_files(video_id)  # 覆盖输出文件

# ❌ 禁止的操作  
shutil.rmtree(input_dir)                         # 删除input目录
os.remove(other_video_files)                     # 删除其他视频文件
```

## 🔄 处理流程

### 1. 视频导入流程
```python
# 1. 用户上传/下载视频
video_path = "downloaded_video.mp4"

# 2. 系统生成ID并注册
video_id = video_mgr.register_video(video_path)
# 结果: abc12345_def6_789012

# 3. 文件自动组织
# input/abc12345_def6_789012.mp4
# temp/abc12345_def6_789012/ (created)
```

### 2. 处理阶段路径获取
```python
# 获取当前视频的路径
adapter = get_path_adapter()
temp_dir = adapter.get_temp_dir()          # temp/abc12345_def6_789012/
audio_dir = adapter.get_audio_dir()        # temp/abc12345_def6_789012/audio/
output_file = adapter.get_output_file("sub") # output/abc12345_def6_789012_sub.mp4
```

### 3. 重新处理流程
```python
# 用户想重新处理同一视频
current_id = adapter.get_current_video_id()

# 安全重置（不删除input）
video_mgr.safe_overwrite_temp_files(current_id)
video_mgr.safe_overwrite_output_files(current_id)

# 开始新的处理流程...
```

## 🛡️ 安全特性

### 1. 文件保护机制
- **Input保护**: 原始视频文件永不删除
- **隔离处理**: 每个视频有独立的处理空间
- **覆盖日志**: 所有覆盖操作都有日志记录
- **元数据追踪**: 每个视频都有完整的处理历史

### 2. 错误恢复
- **优雅降级**: 新架构失败时自动回退到旧模式
- **路径验证**: 所有操作前都验证路径合法性
- **异常处理**: 详细的错误信息和恢复建议

### 3. 数据完整性
- **原子操作**: 文件操作要么全部成功要么全部回滚
- **一致性检查**: 系统启动时验证文件结构完整性
- **备份机制**: 重要操作前自动创建备份点

## 🔄 迁移指南

### 现有代码适配
```python
# 旧代码
output_path = "output/result.mp4"

# 新代码 (推荐)
adapter = get_path_adapter()
output_path = adapter.get_output_file("result")

# 或者直接使用video manager
video_mgr = get_video_manager()
output_path = video_mgr.get_output_file(video_id, "result")
```

### 常用模式
```python
# 1. 开始处理新视频
@with_video_context
def process_video():
    paths = get_current_video_paths()
    # 使用paths进行处理...

# 2. 安全重置
def reset_video_processing():
    adapter = get_path_adapter()
    video_id = adapter.get_current_video_id()
    if video_id:
        video_mgr.safe_overwrite_temp_files(video_id)

# 3. 获取历史文件
def get_video_history(video_id):
    return video_mgr.list_temp_files(video_id)
```

## 📊 性能优化

### 1. 缓存机制
- 视频ID缓存避免重复计算
- 路径缓存减少文件系统调用
- 元数据缓存提升查询性能

### 2. 并行处理
- 不同视频可以完全并行处理
- 同一视频的不同阶段可以流水线处理
- 临时文件读写使用缓冲区优化

### 3. 存储优化
- 按需创建目录结构
- 自动清理过期临时文件
- 压缩存储非活跃数据

## 🔍 监控和调试

### 1. 日志系统
```python
# 操作日志
temp/video_id/logs/overwrite_operations.log

# 处理日志  
temp/video_id/log/processing.log

# GPT交互日志
temp/video_id/gpt_log/interactions.log
```

### 2. 调试工具
```python
# 查看当前状态
video_mgr.get_current_video_id()
video_mgr.list_temp_files(video_id)
video_mgr.list_output_files(video_id)

# 验证文件结构
adapter.get_current_video_paths()
```

## 📈 未来扩展

### 1. 计划功能
- 多用户支持（用户ID + 视频ID）
- 分布式处理支持
- 云存储集成
- 自动存档管理

### 2. API设计
- RESTful API接口
- WebSocket实时状态更新
- 批量处理API
- 历史查询API

---

**架构优势总结**:
- ✅ 完全的文件安全性
- ✅ 清晰的文件组织结构  
- ✅ 可追溯的处理历史
- ✅ 优雅的错误处理
- ✅ 向后兼容性
- ✅ 高性能和可扩展性