"""Global hotkey listener using pynput.

Hotkeys must be re-bound after the user changes them in Settings,
so we always tear the listener down and start a fresh one.
"""
from __future__ import annotations

import traceback
from typing import Callable

from pynput import keyboard


class HotkeyManager:
    def __init__(self) -> None:
        self._listener: keyboard.GlobalHotKeys | None = None

    def set_hotkeys(self, mapping: dict[str, Callable[[], None]]) -> None:
        self.stop()
        # Filter out empty strings so pynput doesn't choke.
        mapping = {k: v for k, v in mapping.items() if k}
        if not mapping:
            return
        try:
            self._listener = keyboard.GlobalHotKeys(mapping)
            self._listener.start()
        except Exception:
            traceback.print_exc()
            self._listener = None

    def stop(self) -> None:
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception:
                pass
            self._listener = None
