# wot_output.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TurnContext:
    """Acumula trazas mecánicas (tiradas, daño, TS, etc.) y log de eventos."""
    mechanics: List[str] = field(default_factory=list)
    log: List[str] = field(default_factory=list)

    def mech(self, line: str) -> None:
        self.mechanics.append(line)

    def log_event(self, line: str) -> None:
        self.log.append(line)


def format_turn_output(
    *,
    ctx: TurnContext,
    interpretation: str,
    narration: str,
    options: Optional[List[str]] = None,
) -> str:
    """
    Construye SIEMPRE el formato fijo con RESOLUCIÓN (mecánica).
    Si no hay mecánica, imprime ' (sin tiradas)' para cumplir transparencia.
    """
    options = options or []

    mechanics_block = "\n".join(ctx.mechanics).strip()
    if not mechanics_block:
        mechanics_block = "(sin tiradas)"

    options_block = "\n".join(f"- {opt}" for opt in options).strip()
    if not options_block:
        options_block = "(sin opciones adicionales)"

    log_block = "\n".join(f"- {line}" for line in ctx.log).strip()
    if not log_block:
        log_block = "(sin cambios relevantes)"

    # Formato fijo
    return (
        "RESOLUCIÓN (mecánica):\n"
        f"{mechanics_block}\n\n"
        "INTERPRETACIÓN:\n"
        f"{interpretation.strip()}\n\n"
        "NARRACIÓN:\n"
        f"{narration.strip()}\n\n"
        "OPCIONES:\n"
        f"{options_block}\n\n"
        "LOG:\n"
        f"{log_block}"
    )
