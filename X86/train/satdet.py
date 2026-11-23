"""训练入口（等价于 python -m satdet train --config ...）。"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from satdet.cli import main

if __name__ == "__main__":
    raise SystemExit(main())