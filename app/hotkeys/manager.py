"""Win32 global hotkey manager.

Uses Win32 RegisterHotKey() so the keystroke is *consumed* by our app and
does NOT propagate to YouTube / Zoom / VS Code / the focused window.

Why we replaced pynput
----------------------
`pynput.keyboard.GlobalHotKeys` installs a low-level keyboard hook in
"observe" mode: it sees keystrokes but never swallows them. That works
for invisible side-features but breaks the moment the hotkey conflicts
with another app:

  - Ctrl+Space  -> YouTube pauses the video, VS Code triggers IntelliSense
  - Space alone -> Zoom toggles "push to talk while muted"
  - F-keys      -> Teams / Discord meeting shortcuts

`RegisterHotKey` is the Windows-native way to register a global hotkey
that ALSO consumes the keystroke. WM_HOTKEY is delivered to our thread's
message queue; the focused window never sees the original keys.

Spec strings keep the old pynput-style format (`<ctrl>+<space>`,
`<ctrl>+<shift>+s`, ...) so existing config files keep working unchanged.
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes
from typing import Callable

from PyQt6.QtCore import QAbstractNativeEventFilter
from PyQt6.QtWidgets import QApplication


# --- Win32 modifier flags ---
MOD_ALT      = 0x0001
MOD_CONTROL  = 0x0002
MOD_SHIFT    = 0x0004
MOD_WIN      = 0x0008
MOD_NOREPEAT = 0x4000  # Win 7+: don't fire WM_HOTKEY on key auto-repeat

WM_HOTKEY = 0x0312

# --- Virtual-Key codes (Microsoft VK_*) ---
_VK_NAMED: dict[str, int] = {
    "space":        0x20,
    "tab":          0x09,
    "enter":        0x0D,
    "return":       0x0D,
    "escape":       0x1B,
    "esc":          0x1B,
    "backspace":    0x08,
    "delete":       0x2E,
    "del":          0x2E,
    "insert":       0x2D,
    "ins":          0x2D,
    "home":         0x24,
    "end":          0x23,
    "page_up":      0x21,
    "page_down":    0x22,
    "pageup":       0x21,
    "pagedown":     0x22,
    "up":           0x26,
    "down":         0x28,
    "left":         0x25,
    "right":        0x27,
    "pause":        0x13,
    "scroll_lock":  0x91,
    "caps_lock":    0x14,
    "print_screen": 0x2C,
    "f1":  0x70, "f2":  0x71, "f3":  0x72, "f4":  0x73,
    "f5":  0x74, "f6":  0x75, "f7":  0x76, "f8":  0x77,
    "f9":  0x78, "f10": 0x79, "f11": 0x7A, "f12": 0x7B,
}

# OEM single-character keys on the US layout.
_OEM_CHAR: dict[str, int] = {
    ";":  0xBA, "/":  0xBF, "`":  0xC0,
    "[":  0xDB, "\\": 0xDC, "]":  0xDD,
    "'":  0xDE, "-":  0xBD, "=":  0xBB,
    ",":  0xBC, ".":  0xBE,
}


def _parse(spec: str) -> tuple[int, int] | None:
    """`'<ctrl>+<space>'` -> `(MOD_CONTROL, 0x20)`. Returns None on failure."""
    if not spec:
        return None
    parts = [p.strip().lower() for p in spec.split("+") if p.strip()]
    mods = 0
    vk: int | None = None
    for raw in parts:
        token = raw[1:-1] if raw.startswith("<") and raw.endswith(">") else raw
        if token in ("ctrl", "control"):
            mods |= MOD_CONTROL
        elif token == "alt":
            mods |= MOD_ALT
        elif token == "shift":
            mods |= MOD_SHIFT
        elif token in ("win", "cmd", "meta"):
            mods |= MOD_WIN
        elif token in _VK_NAMED:
            vk = _VK_NAMED[token]
        elif len(token) == 1:
            ch = token
            if ch.isalnum():
                vk = ord(ch.upper())
            elif ch in _OEM_CHAR:
                vk = _OEM_CHAR[ch]
            else:
                return None
        else:
            return None
    if vk is None:
        return None
    return mods, vk


class _MSG(ctypes.Structure):
    _fields_ = [
        ("hwnd",    wintypes.HWND),
        ("message", wintypes.UINT),
        ("wParam",  wintypes.WPARAM),
        ("lParam",  wintypes.LPARAM),
        ("time",    wintypes.DWORD),
        ("pt_x",    wintypes.LONG),
        ("pt_y",    wintypes.LONG),
    ]


class _HotkeyEventFilter(QAbstractNativeEventFilter):
    """Catches WM_HOTKEY messages from Qt's native event loop."""

    def __init__(self, callbacks: dict[int, Callable[[], None]]):
        super().__init__()
        self._callbacks = callbacks

    def nativeEventFilter(self, eventType, message):  # type: ignore[override]
        et = bytes(eventType) if not isinstance(eventType, bytes) else eventType
        if et != b"windows_generic_MSG":
            return False, 0
        try:
            msg = ctypes.cast(int(message), ctypes.POINTER(_MSG)).contents
        except Exception:
            return False, 0
        if msg.message != WM_HOTKEY:
            return False, 0
        cb = self._callbacks.get(int(msg.wParam))
        if cb is not None:
            try:
                cb()
            except Exception:
                import traceback
                traceback.print_exc()
            return True, 0  # consume
        return False, 0


class HotkeyManager:
    """Global hotkey manager backed by Win32 `RegisterHotKey`.

    Each `set_hotkeys()` call unregisters the previous batch and
    registers the new one. The hotkey IDs are managed internally;
    the caller passes pynput-style spec strings.

    Important: hotkeys registered with `hwnd=NULL` are scoped to the
    calling thread, so this class must be constructed AND have
    `set_hotkeys()` called on the main (Qt GUI) thread.
    """

    # One filter per process, shared by all manager instances.
    _shared_callbacks: dict[int, Callable[[], None]] = {}
    _shared_filter: _HotkeyEventFilter | None = None

    def __init__(self) -> None:
        self._user32 = ctypes.windll.user32
        self._user32.RegisterHotKey.argtypes = [
            wintypes.HWND, ctypes.c_int, wintypes.UINT, wintypes.UINT
        ]
        self._user32.RegisterHotKey.restype = wintypes.BOOL
        self._user32.UnregisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int]
        self._user32.UnregisterHotKey.restype = wintypes.BOOL

        self._next_id = 0xC001  # well above any system-reserved range
        self._registered: list[int] = []

        # Install the native event filter on the QApplication once.
        if HotkeyManager._shared_filter is None:
            app = QApplication.instance()
            if app is None:
                raise RuntimeError(
                    "HotkeyManager must be constructed after QApplication."
                )
            HotkeyManager._shared_filter = _HotkeyEventFilter(
                HotkeyManager._shared_callbacks
            )
            app.installNativeEventFilter(HotkeyManager._shared_filter)

    def set_hotkeys(self, mapping: dict[str, Callable[[], None]]) -> None:
        # Unregister any prior batch owned by this manager.
        for hid in self._registered:
            self._user32.UnregisterHotKey(None, hid)
            HotkeyManager._shared_callbacks.pop(hid, None)
        self._registered.clear()

        for spec, cb in mapping.items():
            parsed = _parse(spec)
            if parsed is None:
                print(f"[hotkeys] invalid spec, skipping: {spec!r}", flush=True)
                continue
            mods, vk = parsed
            hid = self._next_id
            self._next_id += 1
            ok = self._user32.RegisterHotKey(None, hid, mods | MOD_NOREPEAT, vk)
            if not ok:
                err = ctypes.GetLastError()
                hint = (
                    "another app already owns this combination"
                    if err == 1409  # ERROR_HOTKEY_ALREADY_REGISTERED
                    else f"WinError {err}"
                )
                print(
                    f"[hotkeys] RegisterHotKey({spec!r}) failed: {hint}. "
                    "Change it in Settings -> Hotkeys.",
                    flush=True,
                )
                continue
            self._registered.append(hid)
            HotkeyManager._shared_callbacks[hid] = cb
            print(f"[hotkeys] registered {spec!r} -> id={hid}", flush=True)

    def stop(self) -> None:
        for hid in self._registered:
            self._user32.UnregisterHotKey(None, hid)
            HotkeyManager._shared_callbacks.pop(hid, None)
        self._registered.clear()
