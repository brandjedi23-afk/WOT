#!/usr/bin/env python3
# tools/normalize_items.py

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


# =========================================================
# Utilidades
# =========================================================
def _norm(s: str) -> str:
    """Normaliza para indexes.by_name: minúsculas y espacios colapsados."""
    return " ".join((s or "").strip().lower().split())


def _slug_id(name: str) -> str:
    """
    ID estilo bestiary: snake_case, sin símbolos raros.
    Ej: "Wings of Flying" -> "wings_of_flying"
    """
    s = (name or "").strip().lower()
    # reemplaza cualquier cosa no alfanum por underscore
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "item"


def _to_bool(x: Any, default: bool = False) -> bool:
    if isinstance(x, bool):
        return x
    if x is None:
        return default
    if isinstance(x, (int, float)):
        return bool(x)
    s = str(x).strip().lower()
    if s in ("true", "yes", "y", "1", "si", "sí"):
        return True
    if s in ("false", "no", "n", "0"):
        return False
    return default


def _ensure_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def _clean_type(t: str) -> str:
    """
    Corrige el typo común: 'wonderous-item' -> 'wondrous-item'
    Mantiene otros tipos tal cual.
    """
    s = (t or "").strip()
    if s.lower() == "wonderous-item":
        return "wondrous-item"
    return s


# =========================================================
# Carga JSON tolerante a formatos "sucios"
# =========================================================
def load_json_loose(path: str) -> Any:
    """
    Soporta:
      1) JSON estándar: objeto o lista
      2) NDJSON/JSONL: un JSON por línea
      3) Múltiples JSON concatenados (raw_decode)
      4) Fragmentos pegados que empiezan con '},' o comas sueltas:
         - salta basura inicial hasta encontrar '{' o '['
         - salta separadores sueltos entre objetos
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        s = f.read()

    if not s or not s.strip():
        raise ValueError(f"{path} está vacío.")

    # 0) Recorta basura inicial antes del primer '{' o '[' (por si pegas desde mitad de archivo)
    first_obj = min(
        [i for i in (s.find("{"), s.find("[")) if i != -1] or [-1]
    )
    if first_obj > 0:
        s = s[first_obj:]

    s_strip = s.strip()

    # 1) Intento normal
    try:
        return json.loads(s_strip)
    except json.JSONDecodeError:
        pass

    # 2) NDJSON/JSONL
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    if lines:
        nd = []
        ok = True
        for ln in lines:
            try:
                nd.append(json.loads(ln))
            except json.JSONDecodeError:
                ok = False
                break
        if ok and nd:
            return nd

    # 3) Múltiples JSON concatenados / fragmentos
    dec = json.JSONDecoder()
    objs = []
    i = 0
    n = len(s)
    while i < n:
        # salta espacios y separadores típicos
        while i < n and (s[i].isspace() or s[i] in ",;\n\r\t"):
            i += 1
        if i >= n:
            break

        # salta basura hasta próximo inicio válido (muy útil si el texto empieza con "}, {")
        while i < n and s[i] not in "{[":
            i += 1
        if i >= n:
            break

        try:
            obj, end = dec.raw_decode(s, i)
            objs.append(obj)
            i = end
        except json.JSONDecodeError:
            # Si falla, avanza 1 char y prueba de nuevo (tolerante a trozos corruptos)
            i += 1

    if objs:
        # Si el usuario pegó una "lista sin corchetes", esto devolverá [obj1,obj2,...]
        return objs

    raise ValueError(
        f"{path} no parece ser JSON válido (ni lista/objeto/NDJSON/concat). "
        "Revisa si hay texto extra o si pegaste un fragmento incompleto."
    )


# =========================================================
# Normalización de items
# =========================================================
def normalize_item(it: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normaliza un item a un esquema consistente.
    Mantiene campos clave del raw y añade opcionales (tags/summary) si quieres.
    """
    name = str(it.get("name") or it.get("Name") or "").strip()

    out: Dict[str, Any] = {
        "name": name,
        "type": _clean_type(str(it.get("type") or it.get("Type") or "").strip()),
        "rarity": str(it.get("rarity") or it.get("Rarity") or "").strip().lower(),
        "magic": _to_bool(it.get("magic"), default=True),
        "attunement": _to_bool(it.get("attunement"), default=False),
        "text": [str(x).strip() for x in _ensure_list(it.get("text") or it.get("Text")) if str(x).strip()],
        "source": [str(x).strip() for x in _ensure_list(it.get("source") or it.get("Source")) if str(x).strip()],
    }

    # tags / summary opcionales (útiles para el agente, no obligatorios)
    tags = []
    blob = " ".join(out.get("text") or []).lower()

    # heurísticas simples (puedes ampliar)
    if "flying speed" in blob or "fly" in blob or "wings" in blob:
        tags.append("flight")
    if "invisible" in blob or "invisibility" in blob:
        tags.append("invisibility")
    if "resistance" in blob:
        tags.append("resistance")

    if tags:
        out["tags"] = sorted(set(tags))

    if out["text"]:
        # resumen corto = primera frase razonable del primer párrafo
        first = out["text"][0]
        m = re.split(r"(?<=[.!?])\s+", first, maxsplit=1)
        out["summary"] = (m[0] if m else first)[:220]

    # Limpieza mínima: elimina claves vacías
    for k in list(out.keys()):
        v = out[k]
        if v == "" or v == []:
            out.pop(k)

    return out


def build_engine(items_raw: Any) -> Dict[str, Any]:
    """
    Construye el JSON "motor" como bestiary.json:
      - schema_version/meta
      - indexes.by_name
      - items {id: {...}}
    """
    engine: Dict[str, Any] = {
        "schema_version": "1.0",
        "meta": {
            "system": "D&D 5e",
            "notes": "Items mágicos en formato 'motor' para el agente. Indexados por nombre."
        },
        "indexes": {"by_name": {}},
        "items": {}
    }

    # Normaliza entrada: lista directa, o contenedores típicos
    if isinstance(items_raw, list):
        items_list = items_raw
    elif isinstance(items_raw, dict):
        items_list = (
            items_raw.get("items")
            or items_raw.get("Items")
            or items_raw.get("data")
            or items_raw.get("results")
            or []
        )
        if isinstance(items_list, dict):
            # ya venía como dict de id->obj
            items_list = list(items_list.values())
        if not isinstance(items_list, list):
            items_list = [items_raw]
    else:
        raise ValueError("El JSON de entrada debe ser una lista u objeto.")

    by_name = engine["indexes"]["by_name"]
    items_out = engine["items"]

    for it in items_list:
        if not isinstance(it, dict):
            continue
        normed = normalize_item(it)
        name = normed.get("name")
        if not name:
            continue

        item_id = _slug_id(name)

        # evita colisiones: si ya existe, añade sufijo _2, _3...
        base = item_id
        n = 2
        while item_id in items_out:
            item_id = f"{base}_{n}"
            n += 1

        items_out[item_id] = normed
        by_name[_norm(name)] = item_id

    return engine


def save_json(path: str, data: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# =========================================================
# Main
# =========================================================
def main(argv: List[str]) -> int:
    """
    Uso:
      python tools/normalize_items.py [input] [output]

    Defaults:
      input  = data/items_raw.json
      output = dm/items.json
    """
    in_path = argv[1] if len(argv) > 1 else "data/items_raw.json"
    out_path = argv[2] if len(argv) > 2 else "dm/items.json"

    raw = load_json_loose(in_path)
    engine = build_engine(raw)
    save_json(out_path, engine)

    count = len(engine.get("items", {}))
    print(f"OK: {count} items normalizados -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
