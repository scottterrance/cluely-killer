"""Windows-only stealth: hide our overlay from screen-capture APIs.

Uses SetWindowDisplayAffinity(WDA_EXCLUDEFROMCAPTURE) which is supported
on Windows 10 version 2004 (build 19041) and later. The window stays
fully visible to the user but appears black or empty in any process
that captures the screen — Zoom, Teams, Meet, OBS, Discord, etc.
"""
from __future__ import annotations

import ctypes
import sys

WDA_NONE = 0x00000000
WDA_EXCLUDEFROMCAPTURE = 0x00000011  # Win 10 2004+


def exclude_window_from_capture(hwnd: int, enable: bool = True) -> bool:
    """Return True on success."""
    if sys.platform != "win32" or not hwnd:
        return False
    try:
        user32 = ctypes.windll.user32
        affinity = WDA_EXCLUDEFROMCAPTURE if enable else WDA_NONE
        return bool(user32.SetWindowDisplayAffinity(int(hwnd), affinity))
    except Exception as e:
        print(f"[stealth] SetWindowDisplayAffinity failed: {e}")
        return False
