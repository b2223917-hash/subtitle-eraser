import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="留英文 · 中文字幕局部修补")
    parser.add_argument("--demo", action="store_true", help="启动合成动态样例")
    parser.add_argument("--self-test",action="store_true",help="检查桌面界面、打包模型及原生依赖")
    parser.add_argument("--window-test", action="store_true", help="短暂显示生成的测试窗口，检查采集、缩放和全屏恢复")
    parser.add_argument("--selection-test", action="store_true", help="用生成的中文样例检查全屏框选、切换和修补")
    parser.add_argument("--hotkey-test", action="store_true", help="在生成的全屏播放器中检查真实快捷键输入与覆盖恢复")
    parser.add_argument("--output",type=Path,default=Path("outputs/selftest.json"))
    args = parser.parse_args()
    if args.hotkey_test:
        from .hotkey_check import run
        return run(args.output)
    if args.selection_test:
        from .diagnostics import native_selection_check
        return native_selection_check(args.output)
    if args.window_test:
        from .diagnostics import native_transition_check
        return native_transition_check(args.output)
    if args.self_test:
        from .selftest import run
        return run(args.output)
    from .ui import run
    return run(demo=args.demo)


if __name__ == "__main__":
    sys.exit(main())
