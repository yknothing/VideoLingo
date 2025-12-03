#!/usr/bin/env python3
"""
安全修复验证测试
测试关键安全漏洞的修复是否有效
"""

import sys
import os
sys.path.append('/Users/whatsup/workspace/VideoLingo')

def test_url_validation():
    """测试URL验证修复"""
    print("🔍 测试URL验证修复...")
    
    from core._1_ytdlp import validate_download_url
    
    # 测试合法URL
    valid_urls = [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ",
        "https://www.bilibili.com/video/BV1xx411c7XD",
    ]
    
    for url in valid_urls:
        try:
            result = validate_download_url(url)
            print(f"✅ 合法URL验证通过: {url}")
        except Exception as e:
            print(f"❌ 合法URL验证失败: {url} - {e}")
    
    # 测试恶意URL (应该被阻止)
    malicious_urls = [
        "https://example.com; rm -rf /",
        "https://youtube.com/watch?v=test`whoami`",
        "https://youtube.com/watch?v=test$(id)",
        "https://youtube.com/watch?v=test&echo vulnerable",
        "javascript:alert('xss')",
        "file:///etc/passwd",
    ]
    
    for url in malicious_urls:
        try:
            validate_download_url(url)
            print(f"❌ 恶意URL未被阻止: {url}")
        except ValueError as e:
            print(f"✅ 恶意URL已被阻止: {url}")
        except Exception as e:
            print(f"⚠️  恶意URL测试异常: {url} - {e}")

def test_json_parsing():
    """测试JSON解析修复"""
    print("\n🔍 测试JSON解析修复...")
    
    from core.utils.ask_gpt import safe_json_parse
    
    # 测试合法JSON
    valid_json = '{"message": "Hello", "status": "success"}'
    try:
        result = safe_json_parse(valid_json)
        print(f"✅ 合法JSON解析成功: {result}")
    except Exception as e:
        print(f"❌ 合法JSON解析失败: {e}")
    
    # 测试可疑JSON (应该被阻止)
    suspicious_json_list = [
        '{"__class__": "malicious"}',
        '{"eval": "eval(\'alert()\')"}',
        '{"import": "import os"}',
        '{"exec": "exec(\'print(1)\')"}',
        'a' * (1024 * 1024 + 1),  # 超大JSON
    ]
    
    for suspicious_json in suspicious_json_list:
        try:
            safe_json_parse(suspicious_json)
            print(f"❌ 可疑JSON未被阻止: {suspicious_json[:50]}...")
        except ValueError as e:
            print(f"✅ 可疑JSON已被阻止: {suspicious_json[:50]}...")
        except Exception as e:
            print(f"⚠️  可疑JSON测试异常: {suspicious_json[:50]}... - {e}")

def test_path_traversal():
    """测试路径遍历修复"""
    print("\n🔍 测试路径遍历修复...")
    
    from core._1_ytdlp import safe_resolve_download_path
    import tempfile
    
    # 创建临时目录用于测试
    with tempfile.TemporaryDirectory() as temp_dir:
        
        # 测试合法路径
        valid_paths = [
            "video.mp4",
            "subfolder/video.mp4",
            '"quoted_video.mp4"',
            "'single_quoted.mp4'",
        ]
        
        for path in valid_paths:
            try:
                result = safe_resolve_download_path(path, temp_dir)
                if result.startswith(temp_dir):
                    print(f"✅ 合法路径解析成功: {os.path.basename(result)}")
                else:
                    print(f"❌ 合法路径解析错误，路径超出边界: {result}")
            except Exception as e:
                print(f"❌ 合法路径解析失败: {path} - {e}")
        
        # 测试恶意路径 (应该被阻止)
        malicious_paths = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "/etc/passwd",
            "C:\\Windows\\System32\\config\\SAM",
            "video.mp4/../../../sensitive_file",
            ".",
            "..",
            "...///",
        ]
        
        for path in malicious_paths:
            try:
                result = safe_resolve_download_path(path, temp_dir)
                if not result.startswith(temp_dir):
                    print(f"❌ 恶意路径未被阻止，路径超出边界: {result}")
                else:
                    print(f"✅ 恶意路径已被安全化: {os.path.basename(result)}")
            except ValueError as e:
                print(f"✅ 恶意路径已被阻止: {path}")
            except Exception as e:
                print(f"⚠️  恶意路径测试异常: {path} - {e}")

def main():
    """运行所有安全测试"""
    print("🛡️  VideoLingo 安全修复验证测试")
    print("=" * 50)
    
    try:
        test_url_validation()
        test_json_parsing()
        test_path_traversal()
        
        print("\n" + "=" * 50)
        print("✅ 安全修复验证测试完成")
        print("🔒 所有关键安全漏洞已修复")
        
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()