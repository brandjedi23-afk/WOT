# wot_dice.py
from __future__ import annotations
import random
import re
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

Mode = Literal["normal", "adv", "dis"]


@dataclass(frozen=True)
class RollResult:
    total: int
    detail: str   # texto humano
    raw: dict     # estructura útil si quieres


_DICE_TERM = re.compile(r"([+-]?)(\d*)d(\d+)")


def _roll_die(sides: int) -> int:
    return random.randint(1, sides)


def roll_d20(mode: Mode = "normal") -> Tuple[int, str, dict]:
    a = _roll_die(20)
    if mode == "normal":
        return a, f"d20: {a}", {"d20": [a], "mode": mode, "picked": a}

    b = _roll_die(20)
    picked = max(a, b) if mode == "adv" else min(a, b)
    label = "ADV" if mode == "adv" else "DIS"
    return picked, f"d20 ({label}): {a}, {b} -> {picked}", {"d20": [a, b], "mode": mode, "picked": picked}


def roll_expr(expr: str) -> RollResult:
    """
    Soporta expresiones tipo:
      - 2d6+3
      - d8+4
      - 3d4-1
      - 2d6+1d4+2
    (sin paréntesis; fácil de ampliar si lo necesitas)
    """
    s = expr.replace(" ", "")
    if not s:
        raise ValueError("Expresión vacía")

    # Asegura signo explícito para tokenizar constantes
    if s[0] not in "+-":
        s = "+" + s

    total = 0
    parts: List[str] = []
    raw_terms = []

    i = 0
    # Tokenización por términos de dado o constantes
    while i < len(s):
        # Dado
        m = _DICE_TERM.match(s, i)
        if m:
            sign_str, n_str, sides_str = m.groups()
            sign = -1 if sign_str == "-" else 1
            n = int(n_str) if n_str else 1
            sides = int(sides_str)

            rolls = [_roll_die(sides) for _ in range(n)]
            subtotal = sum(rolls) * sign
            total += subtotal

            sign_label = "-" if sign < 0 else "+"
            parts.append(f"{sign_label}{n}d{sides}[{', '.join(map(str, rolls))}]={subtotal}")
            raw_terms.append({"type": "dice", "sign": sign, "n": n, "sides": sides, "rolls": rolls, "subtotal": subtotal})

            i = m.end()
            continue

        # Constante +/-N
        m2 = re.match(r"([+-])(\d+)", s[i:])
        if m2:
            sign = -1 if m2.group(1) == "-" else 1
            val = int(m2.group(2)) * sign
            total += val
            parts.append(f"{m2.group(1)}{m2.group(2)}")
            raw_terms.append({"type": "const", "value": val})
            i += len(m2.group(0))
            continue

        raise ValueError(f"No puedo parsear cerca de: '{s[i:]}'")

    detail = f"{expr} => " + " ".join(parts) + f" | TOTAL={total}"
    return RollResult(total=total, detail=detail, raw={"expr": expr, "terms": raw_terms, "total": total})


def skill_check(*, bonus: int, dc: int, mode: Mode = "normal") -> Tuple[bool, int, str, dict]:
    d20, d20_txt, d20_raw = roll_d20(mode)
    total = d20 + bonus
    ok = total >= dc
    outcome = "ÉXITO" if ok else "FALLO"
    txt = f"{d20_txt} | +{bonus} = {total} vs CD {dc} ({outcome})"
    raw = {"d20": d20_raw, "bonus": bonus, "dc": dc, "total": total, "ok": ok}
    return ok, total, txt, raw


def attack_roll(*, attack_bonus: int, target_ac: int, mode: Mode = "normal") -> Tuple[bool, bool, int, str, dict]:
    d20, d20_txt, d20_raw = roll_d20(mode)
    is_nat20 = (d20 == 20)
    is_nat1 = (d20 == 1)
    total = d20 + attack_bonus

    hit = (not is_nat1) and (is_nat20 or total >= target_ac)
    outcome = "IMPACTA" if hit else "FALLA"
    crit = hit and is_nat20
    crit_txt = " (CRÍTICO)" if crit else ""
    txt = f"{d20_txt} | +{attack_bonus} = {total} vs CA {target_ac} => {outcome}{crit_txt}"
    raw = {"d20": d20_raw, "attack_bonus": attack_bonus, "target_ac": target_ac, "total": total, "hit": hit, "crit": crit}
    return hit, crit, total, txt, raw
