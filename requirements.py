"""Project dependency definitions.

This repo currently uses:
- Ultralytics YOLO (object detection + pose)
- OpenCV for video/image capture and display
- RTMPose runtime library (via `rtmlib`)

This file is intentionally a *Python* representation of dependencies so you
can reuse it programmatically.
"""

from __future__ import annotations

from typing import List


def install_requires() -> List[str]:
    """Return a list of pip-installable requirement specifiers."""

    # Note:
    # - ultralytics will pull in torch automatically.
    # - opencv-python provides `cv2`.
    # - rtmlib provides `Wholebody` and `draw_skeleton` (imported in 3.py).
    return [
        "ultralytics",
        "opencv-python",
        "rtmlib",
        "numpy",
    ]


def pip_install_command() -> str:
    """Return a convenient pip install command string."""

    pkgs = " ".join(install_requires())
    return f"python -m pip install {pkgs}"


if __name__ == "__main__":
    # Prints a requirements-like list for easy copy/paste.
    print("\n".join(install_requires()))
    print("\n# Suggested install command:")
    print(pip_install_command())

