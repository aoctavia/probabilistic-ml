"""e2_misspecification — stub.

To be implemented in the relevant phase. See BRIEF.md §9.
"""
import sys
from pathlib import Path

def main(config_path: Path) -> None:
    raise NotImplementedError("Not yet implemented — see BRIEF.md for phase ordering.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <config.yaml>")
        sys.exit(1)
    main(Path(sys.argv[1]))
