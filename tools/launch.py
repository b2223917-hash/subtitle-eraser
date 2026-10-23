"""GUI entry point for the portable executable."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))

if __name__ == "__main__":
    import os
    import traceback
    log_root=Path(os.environ.get("LOCALAPPDATA",str(Path.home())))/"SubtitleEraser"
    log_root.mkdir(parents=True,exist_ok=True)
    log_path=log_root/"runtime.log"
    log=log_path.open("a",encoding="utf-8",buffering=1)
    if sys.stdout is None:
        sys.stdout=log
    if sys.stderr is None:
        sys.stderr=log
    try:
        from subtitle_eraser.__main__ import main
        sys.exit(main())
    except Exception:
        traceback.print_exc(file=log)
        diagnostic_flags = {"--self-test", "--window-test", "--selection-test", "--hotkey-test"}
        if not diagnostic_flags.intersection(sys.argv):
            import ctypes
            ctypes.windll.user32.MessageBoxW(None,f"程序未能启动，错误已记录在：\n{log_path}","留英文",0x10)
        sys.exit(1)
