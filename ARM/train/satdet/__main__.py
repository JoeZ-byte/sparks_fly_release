"""允许 ``python -m satdet ...``。"""

from satdet.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
