# tools/normalize_spells.py
# ---------------------------------------------------------
# Convierte data/spells_raw.json -> dm/spells.json
# Salida con la MISMA estructura base que bestiary/items:
# {
#   "schema_version": "1.0",
#   "meta": {...},
#   "indexes": { "by_name": { "<norm name>": "<spell_id>" } },
#   "spells": { "<spell_id>": { ...spell data... } }
# }
#
# Uso:
#   python tools/normalize_spells.py
#   python tools/normalize_spells.py --in data/spells_raw.json --out dm/spells.json
#
# Nota: El script es robusto con formatos de entrada comunes:
# - lista de spells: [ {name:...}, ... ]
# - dict con clave "spells" o "data"
# - dict id->spell (mapa)
# ---------------------------------------------------------

import argparse
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# -------------------------
# Normalización
# -------------------------
def _norm(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def _spell_key_from_name(name: str) -> str:
    """
    "Cure Wounds" -> "cure_wounds"
    "Tasha's Hideous Laughter" -> "tashas_hideous_laughter"
    """
    s = _norm(name)
    s = s.replace("’", "'")  # apóstrofo unicode
    # quita comillas/apóstrofes
    s = s.replace("'", "")
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "spell"


def _ensure_unique_key(key: str, used: set) -> str:
    if key not in used:
        used.add(key)
        return key
    i = 2
    while True:
        cand = f"{key}_{i}"
        if cand not in used:
            used.add(cand)
            return cand
        i += 1


# -------------------------
# Lectura & flatten de raw
# -------------------------
def _load_json(path: Path) -> Any:
    """
    Loader robusto:
    - JSON estándar (un único objeto/array)
    - JSONL/NDJSON (un JSON por línea)
    - JSON concatenado (varios objetos/arrays pegados)
    - Archivos con prefijo/sufijo no-JSON (p.ej. 'const x = ...', 'export default ...', ';' final)
    """
    if not path.exists():
        raise FileNotFoundError(f"No existe: {path}")

    text = path.read_text(encoding="utf-8")

    # Quita BOM si existe
    if text.startswith("\ufeff"):
        text = text.lstrip("\ufeff")

    # Limpieza suave de wrappers típicos (sin depender de regex complicada)
    # Si hay un ';' final típico de JS, lo quitamos
    text = text.strip()
    if text.endswith(";"):
        text = text[:-1].rstrip()

    # 1) Intento normal: JSON puro
    try:
        return json.loads(text)
    except json.JSONDecodeError as e1:
        # 2) Si parece JSONL: un JSON por línea
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        jsonl_items = []
        ok = True
        for ln in lines:
            try:
                jsonl_items.append(json.loads(ln))
            except Exception:
                ok = False
                break
        if ok and jsonl_items:
            return jsonl_items

        # 3) Parseo tolerante: ignora prefijos no-JSON y soporta concatenación
        decoder = json.JSONDecoder()
        n = len(text)
        idx = 0
        items = []

        # caracteres que pueden iniciar un valor JSON
        starters = set('{["-0123456789tfn')  # true/false/null también

        # función para avanzar hasta un inicio plausible de JSON
        def _advance_to_starter(i: int) -> int:
            while i < n and text[i] not in starters:
                i += 1
            return i

        idx = _advance_to_starter(idx)
        while idx < n:
            try:
                obj, end = decoder.raw_decode(text, idx)
                items.append(obj)
                idx = end
                # saltar whitespace y basura intermedia, y seguir
                idx = _advance_to_starter(idx)
            except json.JSONDecodeError:
                # si falla en este idx, avanzamos 1 y volvemos a buscar un starter
                idx += 1
                idx = _advance_to_starter(idx)

        if items:
            return items[0] if len(items) == 1 else items

        # Si llegamos aquí, no pudimos rescatar nada: re-lanzamos el error original (el del json.loads)
        raise e1


def _extract_spell_objects(raw: Any) -> List[Dict[str, Any]]:
    """
    Intenta extraer una lista de dicts de spells desde distintos formatos.
    """
    # 1) Formato ideal: lista
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, dict)]

    # 2) Dict con "spells" o "data"
    if isinstance(raw, dict):
        for k in ("spells", "Spells", "data", "Data"):
            v = raw.get(k)
            if isinstance(v, list):
                return [x for x in v if isinstance(x, dict)]
            if isinstance(v, dict):
                # dict id->spell
                out = []
                for _id, obj in v.items():
                    if isinstance(obj, dict):
                        o = deepcopy(obj)
                        o.setdefault("_raw_id", str(_id))
                        out.append(o)
                return out

        # 3) Dict id->spell directamente
        # Heurística: si muchos values son dict con "name"
        values = list(raw.values())
        if values and all(isinstance(v, dict) for v in values):
            # si al menos 1 tiene name, lo tratamos como mapa
            if any("name" in v for v in values):
                out = []
                for _id, obj in raw.items():
                    if isinstance(obj, dict):
                        o = deepcopy(obj)
                        o.setdefault("_raw_id", str(_id))
                        out.append(o)
                return out

    return []


# -------------------------
# Limpieza mínima opcional
# -------------------------
def _coerce_level(sp: Dict[str, Any]) -> None:
    """
    Normaliza level si viene como string. Conserva cantrip como 0 si detecta "cantrip".
    """
    lvl = sp.get("level", sp.get("lvl", sp.get("spell_level")))
    if lvl is None:
        return

    if isinstance(lvl, str):
        t = _norm(lvl)
        if "cantrip" in t:
            sp["level"] = 0
            return
        m = re.search(r"\d+", t)
        if m:
            sp["level"] = int(m.group(0))
            return

    if isinstance(lvl, (int, float)):
        sp["level"] = int(lvl)


def _ensure_name(sp: Dict[str, Any]) -> Optional[str]:
    """
    Encuentra nombre en campos típicos.
    """
    for k in ("name", "Name", "spell", "title"):
        v = sp.get(k)
        if isinstance(v, str) and v.strip():
            sp["name"] = v.strip()
            return sp["name"]
    return None


# -------------------------
# Normalización principal
# -------------------------
def normalize_spells(raw: Any) -> Dict[str, Any]:
    spells_list = _extract_spell_objects(raw)

    used_keys = set()
    by_name: Dict[str, str] = {}
    spells_out: Dict[str, Dict[str, Any]] = {}

    for sp in spells_list:
        if not isinstance(sp, dict):
            continue

        name = _ensure_name(sp)
        if not name:
            continue

        _coerce_level(sp)

        base_key = _spell_key_from_name(name)
        spell_id = _ensure_unique_key(base_key, used_keys)

        # Índice por nombre (normalizado) -> id interno
        by_name[_norm(name)] = spell_id

        # Copia profunda para no mutar el raw
        obj = deepcopy(sp)
        obj.setdefault("_spell_id", spell_id)

        spells_out[spell_id] = obj

    out = {
        "schema_version": "1.0",
        "meta": {
            "system": "D&D 5e",
            "notes": "Hechizos normalizados en formato 'motor' para el agente. Index por nombre y dict de spells por id.",
            "source_raw": "data/spells_raw.json",
        },
        "indexes": {"by_name": by_name},
        "spells": spells_out,
    }
    return out


# -------------------------
# Escritura segura
# -------------------------
def write_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default="data/spells_raw.json", help="Ruta a spells_raw.json")
    ap.add_argument("--out", dest="out_path", default="dm/spells.json", help="Ruta de salida dm/spells.json")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)

    raw = _load_json(in_path)
    normalized = normalize_spells(raw)

    # Validación mínima
    if not isinstance(normalized.get("spells"), dict) or not normalized["spells"]:
        raise RuntimeError(
            "No se generaron spells. Revisa el formato de data/spells_raw.json "
            "(debe contener objetos con campo 'name' o equivalente)."
        )

    write_json_atomic(out_path, normalized)

    print(f"OK. Hechizos normalizados: {len(normalized['spells'])}")
    print(f"Escrito: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
