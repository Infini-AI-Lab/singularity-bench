"""python3 -m breakwater: the breakwater command (cli.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cli import main  # noqa: E402

raise SystemExit(main())
