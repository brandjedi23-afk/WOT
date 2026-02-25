#!/usr/bin/env python3
# tools/normalize_monsters.py

import json
import re
import sys
from typing import Any, Dict, List

def clean_str(x: Any) -> str:
    """Convierte a string y limpia comillas y espacios como hacía el .replace(/['"]/g,"") de JS."""
    if x is None:
        return ""
    s = str(x)
    # Quita comillas simples y dobles (equivalente a /['"]/g)
    s = re.sub(r"[\"']", "", s)
    # Normaliza espacios
    s = re.sub(r"\s+", " ", s).strip()
    return s

def to_int(x: Any, default: int = 0) -> int:
    if x is None:
        return default
    if isinstance(x, (int, float)):
        return int(x)
    s = str(x).strip()
    # Extrae el primer entero (sirve para "12 (3d8)" -> 12)
    m = re.search(r"-?\d+", s)
    return int(m.group(0)) if m else default

def slugify(name: str) -> str:
    """
    'Adult Blue Dragon' -> 'adult_blue_dragon'
    Mantiene solo [a-z0-9_] y colapsa espacios/puntuación a '_'.
    """
    s = clean_str(name).lower()
    s = re.sub(r"[’']", "", s)          # quita apóstrofes
    s = re.sub(r"[^a-z0-9]+", "_", s)   # todo lo demás -> _
    s = re.sub(r"_+", "_", s).strip("_")
    return s

def ensure_unique_key(base: str, used: set) -> str:
    if base not in used:
        used.add(base)
        return base
    i = 2
    while f"{base}_{i}" in used:
        i += 1
    key = f"{base}_{i}"
    used.add(key)
    return key

def normalize_monster(m: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normaliza un monstruo a un esquema consistente.
    Ajusta aquí los campos a TU esquema “del primer monstruo”.
    """
    out: Dict[str, Any] = {}

    # Nombre / ID
    name = m.get("name") or m.get("Name") or m.get("nombre") or ""
    out["name"] = clean_str(name)

    # Tipo / tamaño / alineamiento
    out["size"] = clean_str(m.get("size") or m.get("Size") or "")
    out["type"] = clean_str(m.get("type") or m.get("Type") or "")
    out["alignment"] = clean_str(m.get("alignment") or m.get("Alignment") or "")

    # AC / HP / Speed
    out["ac"] = to_int(m.get("ac") or m.get("AC") or m.get("armor_class"), 10)
    out["hp"] = to_int(m.get("hp") or m.get("HP") or m.get("hit_points"), 1)
    out["speed"] = clean_str(m.get("speed") or m.get("Speed") or "")

    # Stats
    stats = m.get("stats") or {}
    out["str"] = to_int(m.get("str") or m.get("STR") or stats.get("str"), 10)
    out["dex"] = to_int(m.get("dex") or m.get("DEX") or stats.get("dex"), 10)
    out["con"] = to_int(m.get("con") or m.get("CON") or stats.get("con"), 10)
    out["int"] = to_int(m.get("int") or m.get("INT") or stats.get("int"), 10)
    out["wis"] = to_int(m.get("wis") or m.get("WIS") or stats.get("wis"), 10)
    out["cha"] = to_int(m.get("cha") or m.get("CHA") or stats.get("cha"), 10)

    # CR / XP (si existen)
    out["cr"] = clean_str(m.get("cr") or m.get("CR") or "")
    out["xp"] = to_int(m.get("xp") or m.get("XP") or 0, 0)

    # Rasgos / acciones (normaliza a listas de {name, desc})
    def norm_named_list(v: Any) -> List[Dict[str, str]]:
        if not v:
            return []
        if isinstance(v, list):
            res = []
            for item in v:
                if isinstance(item, dict):
                    res.append({
                        "name": clean_str(item.get("name") or item.get("Name") or ""),
                        "desc": clean_str(item.get("desc") or item.get("Desc") or item.get("description") or "")
                    })
                else:
                    res.append({"name": "", "desc": clean_str(item)})
            return res
        if isinstance(v, dict):
            # a veces viene como { "TraitName": "desc", ... }
            return [{"name": clean_str(k), "desc": clean_str(val)} for k, val in v.items()]
        # string suelto
        return [{"name": "", "desc": clean_str(v)}]

    out["traits"] = norm_named_list(m.get("traits") or m.get("Traits"))
    out["actions"] = norm_named_list(m.get("actions") or m.get("Actions"))
    out["reactions"] = norm_named_list(m.get("reactions") or m.get("Reactions"))
    out["legendary_actions"] = norm_named_list(
        m.get("legendary_actions") or m.get("LegendaryActions") or m.get("legendary")
    )

    # Limpieza final: elimina claves vacías si quieres
    for k in list(out.keys()):
        if out[k] == "" and k in ("size", "type", "alignment", "cr", "speed"):
            out.pop(k)

    return out

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        s = f.read().strip()

    # 1) Intento normal: JSON único (lista u objeto)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    # 2) Intento NDJSON/JSONL: un JSON por línea
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    ndjson_objs = []
    ok = True
    for ln in lines:
        try:
            ndjson_objs.append(json.loads(ln))
        except json.JSONDecodeError:
            ok = False
            break
    if ok and ndjson_objs:
        return ndjson_objs

    # 3) Intento “múltiples JSON seguidos” usando raw_decode
    dec = json.JSONDecoder()
    objs = []
    i = 0
    n = len(s)
    while i < n:
        while i < n and s[i].isspace():
            i += 1
        if i >= n:
            break
        obj, end = dec.raw_decode(s, i)
        objs.append(obj)
        i = end
        # admite separadores tipo \n o comas sueltas
        while i < n and (s[i].isspace() or s[i] == ","):
            i += 1

    if objs:
        return objs

    raise ValueError(
        f"{path} no es JSON único, NDJSON, ni múltiples JSON concatenados. "
        "Revisa que no haya texto extra o comillas/comas rotas."
    )

def save_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def main(argv: List[str]) -> int:
    if len(argv) < 3:
        print("Uso: python tools/normalize_monsters.py <input.json> <output.json>")
        return 2

    in_path, out_path = argv[1], argv[2]
    raw = load_json(in_path)

    # Acepta: lista directa, o {monsters:[...]} etc.
    if isinstance(raw, list):
        monsters = raw
    elif isinstance(raw, dict):
        monsters = raw.get("monsters") or raw.get("Monsters") or raw.get("data") or []
        if isinstance(monsters, dict):
            # Ya viene como {id: monster}; lo convertimos a lista
            monsters = list(monsters.values())
        elif not isinstance(monsters, list):
            # Si es un dict único, lo envolvemos
            monsters = [raw]
    else:
        raise ValueError("El JSON de entrada debe ser una lista o un objeto.")

    # Salida: {"monsters": { "<slug>": {...}, ... } }
    out_monsters: Dict[str, Dict[str, Any]] = {}
    used_keys: set = set()

    for m in monsters:
        if not isinstance(m, dict):
            continue
        nm = normalize_monster(m)
        name = nm.get("name", "")
        if not name:
            continue

        key_base = slugify(name)
        if not key_base:
            continue
        key = ensure_unique_key(key_base, used_keys)

        out_monsters[key] = nm

    save_json(out_path, {"monsters": out_monsters})
    print(f"OK: {len(out_monsters)} monstruos normalizados -> {out_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
