"""E1 — Recovery under known truth.

Status: stub — to be implemented in Phase 1.
"""

import sys
from pathlib import Path


def main(config_path: Path) -> None:
    raise NotImplementedError(
        "E1 is not yet implemented. "
        "Complete Phase 1 (M0 + CAVI) first."
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <config.yaml>")
        sys.exit(1)
    main(Path(sys.argv[1]))
