"""
契约性测试（暂存）：针对 core.utils.config_utils.save_key 的行为契约。
注意：当前生产代码未提供 save_key。本文件以 xfail 形式提交，等待 Owner 实现后解除标记。
不修改生产源码，先用测试约束接口设计，保障实现一致性与可测性。

启用方式：
- 当 core.utils.config_utils.save_key 合入后，将 xfail 标记移除或改为条件判断。
"""

import os
import sys
import stat
import platform
from pathlib import Path

import pytest

# 尝试导入被测模块
pytestmark = pytest.mark.xfail(reason="save_key 尚未在 core.utils.config_utils 中实现，等待 Owner 合入", strict=False)

from core.utils import config_utils  # noqa: E402


# 测试常量集中管理，避免魔法值
NAMESPACE = "unit_test_ns"
VALUE_ASCII = "test_secret_value_123"
VALUE_UNICODE = "密钥-Δ-🚀"
DEFAULT_MODE = 0o600


def _is_posix() -> bool:
    return os.name == "posix"


def _assert_mode_600(p: Path):
    if not _is_posix():
        # Windows 等平台不强校验权限
        return
    # 跳过非本地文件系统或不支持的情况
    try:
        st_mode = p.stat().st_mode
        file_perm = stat.S_IMODE(st_mode)
        assert file_perm == DEFAULT_MODE, f"file mode expected {oct(DEFAULT_MODE)}, got {oct(file_perm)}"
    except OSError as e:
        pytest.skip(f"权限断言跳过：{e}")


def test_save_then_load_roundtrip_tmp_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """
    场景1：在独立目录下写入并读回，验证路径与编码一致性
    """
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    target_file = cfg_dir / "keys.ini"  # 仅为示例名称，真实名称应由 save_key 决定或由 path 显式传入

    # 若实现允许通过 path 精确指定文件路径，则直接传入 path
    # 否则应通过 monkeypatch 环境变量/默认目录，保证 load/save 命中同一路径
    if "VIDEOLINGO_CONFIG_DIR" in os.environ:
        monkeypatch.delenv("VIDEOLINGO_CONFIG_DIR")
    monkeypatch.setenv("VIDEOLINGO_CONFIG_DIR", str(cfg_dir))

    # 允许实现暴露 path 参数，若无则期望其根据环境变量/默认规则定位到 cfg_dir
    if hasattr(config_utils, "save_key"):
        ret_path = config_utils.save_key(namespace=NAMESPACE, value=VALUE_ASCII, path=target_file)
        assert ret_path is not None
        ret_path = Path(ret_path)
        assert ret_path.exists() and ret_path.is_file()

        # roundtrip
        loaded = config_utils.load_key(namespace=NAMESPACE, path=ret_path)
        assert loaded == VALUE_ASCII

        _assert_mode_600(ret_path)
    else:
        pytest.xfail("save_key 尚不可用")


def test_save_overwrite_idempotent(tmp_path: Path):
    """
    场景2：重复写入具备幂等覆盖语义
    """
    p = tmp_path / "k.ini"
    if hasattr(config_utils, "save_key"):
        config_utils.save_key(NAMESPACE, VALUE_ASCII, path=p)
        v1 = config_utils.load_key(NAMESPACE, path=p)
        assert v1 == VALUE_ASCII

        config_utils.save_key(NAMESPACE, VALUE_UNICODE, path=p)
        v2 = config_utils.load_key(NAMESPACE, path=p)
        assert v2 == VALUE_UNICODE
        _assert_mode_600(p)
    else:
        pytest.xfail("save_key 尚不可用")


def test_save_with_invalid_params(tmp_path: Path):
    """
    场景3：非法入参需抛出受控异常（建议 ValueError）
    """
    p = tmp_path / "k.ini"
    if hasattr(config_utils, "save_key"):
        with pytest.raises((ValueError, TypeError)):
            config_utils.save_key(namespace="", value=VALUE_ASCII, path=p)
        with pytest.raises((ValueError, TypeError)):
            config_utils.save_key(namespace=NAMESPACE, value="", path=p)
    else:
        pytest.xfail("save_key 尚不可用")


def test_save_to_unwritable_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """
    场景4：无写权限目录下保存应抛出 OSError/IOError
    """
    if not hasattr(config_utils, "save_key"):
        pytest.xfail("save_key 尚不可用")

    # 构造不可写目录（POSIX 下使用权限位；Windows 上可能不可靠，遇到失败则 skip）
    unwritable = tmp_path / "no_write"
    unwritable.mkdir(parents=True, exist_ok=True)
    if _is_posix():
        try:
            unwritable.chmod(0o500)  # 可读可执行，不可写
        except Exception:
            pytest.skip("当前文件系统不支持修改权限，跳过")
    else:
        pytest.skip("Windows 权限模拟不稳定，跳过")

    with pytest.raises((OSError, PermissionError)):
        config_utils.save_key(namespace=NAMESPACE, value=VALUE_ASCII, path=unwritable / "k.ini")


def test_save_non_ascii_value_roundtrip(tmp_path: Path):
    """
    场景5：非 ASCII 内容按 utf-8 无损保存与读取
    """
    p = tmp_path / "k_utf8.ini"
    if hasattr(config_utils, "save_key"):
        config_utils.save_key(namespace=NAMESPACE, value=VALUE_UNICODE, path=p)
        got = config_utils.load_key(namespace=NAMESPACE, path=p)
        assert got == VALUE_UNICODE
        _assert_mode_600(p)
    else:
        pytest.xfail("save_key 尚不可用")