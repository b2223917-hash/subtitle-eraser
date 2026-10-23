"""Interactive check using real key input in the generated fullscreen player."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from subtitle_eraser.hotkey_check import run

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('outputs/hotkey-fix/native-hotkey.json'))
    raise SystemExit(run(parser.parse_args().output))
