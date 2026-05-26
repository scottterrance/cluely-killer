"""Named persona presets - bundles of {about, resume, JD, custom prompt}.

Each persona is a saved snapshot of the four "Your Context" fields,
under a friendly name. The active persona's content is mirrored into
Settings on Save, so the runtime path is unchanged: prompt builder
still reads from `settings.about_me` / `settings.resume_text` / etc.

Stored at ~/.cluely_killer/personas.json:

  {
    "active": "Stripe Senior PM",
    "personas": {
      "Default":          {"about_me": "...", "resume_text": "...", ...},
      "Stripe Senior PM": {...},
      "Junior Dev":       {...}
    }
  }

Kept in its own file (not in config.json) because resumes are big and
backing up / sharing personas independently is convenient.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

PERSONAS_FILE = Path.home() / ".cluely_killer" / "personas.json"
DEFAULT_NAME = "Default"


@dataclass
class Persona:
    name: str
    about_me: str = ""
    resume_text: str = ""
    job_description: str = ""
    custom_system_prompt: str = ""

    def to_dict(self) -> dict:
        return {
            "about_me": self.about_me,
            "resume_text": self.resume_text,
            "job_description": self.job_description,
            "custom_system_prompt": self.custom_system_prompt,
        }


class PersonaStore:
    def __init__(self) -> None:
        self._personas: dict[str, Persona] = {}
        self._active: str = DEFAULT_NAME
        self._load()

    # --------------------------- I/O ---------------------------------
    def _load(self) -> None:
        if not PERSONAS_FILE.exists():
            return
        try:
            data = json.loads(PERSONAS_FILE.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[personas] failed to load, starting empty: {e}")
            return
        active = data.get("active")
        if isinstance(active, str) and active:
            self._active = active
        for name, blob in (data.get("personas") or {}).items():
            if not isinstance(name, str) or not isinstance(blob, dict):
                continue
            self._personas[name] = Persona(
                name=name,
                about_me=blob.get("about_me", "") or "",
                resume_text=blob.get("resume_text", "") or "",
                job_description=blob.get("job_description", "") or "",
                custom_system_prompt=blob.get("custom_system_prompt", "") or "",
            )
        # Self-heal: active must exist.
        if self._active not in self._personas and self._personas:
            self._active = next(iter(self._personas))

    def _persist(self) -> None:
        PERSONAS_FILE.parent.mkdir(parents=True, exist_ok=True)
        PERSONAS_FILE.write_text(
            json.dumps(
                {
                    "active": self._active,
                    "personas": {n: p.to_dict() for n, p in self._personas.items()},
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    # --------------------------- API ---------------------------------
    def names(self) -> list[str]:
        return list(self._personas.keys())

    def get(self, name: str) -> Persona | None:
        return self._personas.get(name)

    def get_active(self) -> Persona | None:
        return self._personas.get(self._active)

    def active_name(self) -> str:
        return self._active

    def set_active(self, name: str) -> bool:
        if name not in self._personas:
            return False
        self._active = name
        self._persist()
        return True

    def upsert(self, persona: Persona) -> None:
        """Create or update a persona by name."""
        if not persona.name:
            raise ValueError("persona.name cannot be empty")
        # Defensive copy so external mutation doesn't sneak in.
        self._personas[persona.name] = Persona(
            name=persona.name,
            about_me=persona.about_me,
            resume_text=persona.resume_text,
            job_description=persona.job_description,
            custom_system_prompt=persona.custom_system_prompt,
        )
        self._persist()

    def rename(self, old_name: str, new_name: str) -> bool:
        new_name = (new_name or "").strip()
        if (
            not new_name
            or new_name == old_name
            or old_name not in self._personas
            or new_name in self._personas
        ):
            return False
        p = self._personas.pop(old_name)
        p.name = new_name
        self._personas[new_name] = p
        if self._active == old_name:
            self._active = new_name
        self._persist()
        return True

    def delete(self, name: str) -> bool:
        # Refuse to delete the last one - at least one persona must exist.
        if name not in self._personas or len(self._personas) <= 1:
            return False
        del self._personas[name]
        if self._active == name:
            self._active = next(iter(self._personas))
        self._persist()
        return True

    def ensure_seeded(self, fallback: Persona) -> None:
        """If the store is empty, seed it with `fallback` and make it active."""
        if self._personas:
            return
        self._personas[fallback.name] = fallback
        self._active = fallback.name
        self._persist()
