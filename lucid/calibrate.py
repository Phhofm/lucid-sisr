"""Entry point for the ``lucid-calibrate`` console script.

Wraps ``lucid/scripts/calibrate_icnet.py`` so users can run ``lucid-calibrate
--input <folder>`` after ``pip install lucid-sisr``. The wrapper:

- Forwards all CLI flags through to the script
- Defaults ``--output`` to the installed ``lucid/resources/icnet_calibration.json``
  so a fresh run overwrites the shipped calibration in place
- Falls back to a self-contained report if the script is not available
"""
from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    pkg_root = Path(__file__).resolve().parent
    candidates = [
        pkg_root / "scripts" / "calibrate_icnet.py",
    ]
    script_path = next((c for c in candidates if c.exists()), None)
    if script_path is None:
        # Installed from a wheel without the script (very slim build);
        # fall back to a self-contained report.
        print(
            "lucid-calibrate: scripts/calibrate_icnet.py not found in install.\n"
            "Falling back to bundled calibration report.",
            file=sys.stderr,
        )
        from lucid.resources.icnet_calibration import calibration_report
        print(calibration_report())
        return 0

    scripts_dir = script_path.parent
    repo_root = pkg_root.parent  # for resolving default --output

    # Forward all CLI args to the script, but default --output to the
    # installed calibration JSON so a user can run
    #   lucid-calibrate --input /my/folder
    # without having to think about where to write.
    argv = sys.argv[1:]
    if "--output" not in argv and "-o" not in argv:
        default_out = (
            repo_root / "lucid" / "resources" / "icnet_calibration.json"
        )
        if default_out.exists():
            argv = [*argv, "--output", str(default_out)]

    sys.path.insert(0, str(scripts_dir))
    from calibrate_icnet import main as _main

    # Push our resolved argv into the script's argparse
    old_argv = sys.argv
    sys.argv = ["calibrate_icnet", *argv]
    try:
        return _main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    raise SystemExit(main())
