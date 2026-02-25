import os
import time
import json
import re
import random
import difflib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable, Set
import inspect
from copy import deepcopy
from pathlib import Path
from wot_output import TurnContext, format_turn_output
from wot_dice import roll_expr, skill_check, attack_roll

from dotenv import load_dotenv
from openai import OpenAI

# =========================================================
# Setup (ROOT -> .env -> client)
# =========================================================
ROOT = Path(__file__).resolve().parent
load_dotenv(dotenv_path=ROOT / ".env", override=False)

client = OpenAI()  # lee OPENAI_API_KEY del entorno

def _require_model() -> str:
    m = (os.getenv("OPENAI_MODEL") or "").strip()
    if not m:
        raise RuntimeError("Falta OPENAI_MODEL (ponlo en .env o en variables de entorno).")
    return m

# Rutas ancladas a tu proyecto (evita problemas con uvicorn)
CANON_PATH  = ROOT / "dm" / "canon.json"
LOG_PATH    = ROOT / "data" / "logs" / "session.md"
MEMORY_PATH = ROOT / "data" / "memory.json"

# =========================================================
# Compendios (bestiary + items + spells)
# =========================================================
BESTIARY_PATH = ROOT / "dm" / "bestiary.json"
ITEMS_PATH    = ROOT / "dm" / "items.json"
SPELLS_PATH   = ROOT / "dm" / "spells.json"
MODIFIERS_PATH = ROOT / "dm" / "modifiers.json"
MODIFIERS_PATHS = [MODIFIERS_PATH]  # List of possible paths to check for modifiers

# Cache separado para evitar conflicto de formatos:
# - COMPENDIO: {"indexes": {"by_name": ...}, "monsters": {...}}
# - PLANO: {"adult_blue_dragon": {...}, ...}
_BESTIARY_COMP_CACHE: Optional[dict] = None
_BESTIARY_FLAT_CACHE: Optional[dict] = None
_ITEMS_CACHE: Optional[dict] = None
_SPELLS_CACHE: Optional[dict] = None
_MODIFIERS_CACHE: Optional[list] = None

# =========================================================
# Módulos (PDF) — indexado y consulta (fidelidad de aventura)
# =========================================================
MODULES_DIR   = ROOT / "dm" / "modules"
MODULES_DIR.mkdir(parents=True, exist_ok=True)

# Registro de módulos conocidos (puedes añadir más)
KNOWN_MODULES = {
    "expedition_ruins_greyhawk": {
        "title": "Expedition to the Ruins of Greyhawk",
        "pdf_path": str(ROOT / "dm" / "modules" / "Expedition to the Ruins of Greyhawk.pdf")
    }
}

_MODULE_INDEX_CACHE: Dict[str, dict] = {}

def _slug(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_")


def _module_index_path(module_id: str) -> Path:
    return MODULES_DIR / f"{_slug(module_id)}.index.json"


def _pdf_pages_via_fitz(pdf_path: Path) -> List[str]:
    """
    Extractor robusto para indexado de módulos: PyMuPDF (fitz) únicamente.
    Evita pypdf porque algunos PDFs antiguos rompen el trailer/xref.
    """
    import fitz  # PyMuPDF

    doc = fitz.open(str(pdf_path))
    pages: List[str] = []
    for i in range(doc.page_count):
        page = doc.load_page(i)
        txt = page.get_text("text") or ""
        txt = re.sub(r"[ \t]+", " ", txt)
        txt = re.sub(r"\n{3,}", "\n\n", txt).strip()
        pages.append(txt)
    doc.close()
    return pages

def _build_module_index(module_id: str, pdf_path: Path, *, chunk_chars: int = 1800, overlap: int = 200) -> dict:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF no encontrado: {pdf_path}")

    try:
        pages = _pdf_pages_via_fitz(pdf_path)  # ✅ principal (robusto)
        if pages and all(not (p or "").strip() for p in pages):
            raise RuntimeError("El PDF parece no contener texto seleccionable (posible escaneado).")
    except Exception as e:
        raise RuntimeError(f"No pude extraer texto del PDF: {e}")

    chunks: List[dict] = []

    for p_idx, txt in enumerate(pages, start=1):
        if not txt:
            continue

        # corta en bloques para retrieval
        start = 0
        cnum = 0
        while start < len(txt):
            end = min(len(txt), start + chunk_chars)
            block = txt[start:end].strip()
            if block:
                chunks.append({
                    "id": f"p{p_idx:03d}_{cnum:03d}",
                    "page": p_idx,
                    "text": block
                })
            cnum += 1
            # overlap para que no se pierdan frases al cortar
            start = end - overlap if end - overlap > start else end

    meta = KNOWN_MODULES.get(module_id, {})
    return {
        "schema_version": "1.0",
        "module_id": module_id,
        "title": meta.get("title", module_id),
        "pdf": str(pdf_path),
        "chunk_chars": chunk_chars,
        "overlap": overlap,
        "chunks": chunks
    }

def get_module_index(module_id: str) -> Optional[dict]:
    """
    Carga índice desde cache o disco.
    """
    mid = _slug(module_id)
    if mid in _MODULE_INDEX_CACHE:
        return _MODULE_INDEX_CACHE[mid]

    idx_path = _module_index_path(mid)
    if not idx_path.exists():
        return None

    try:
        obj = json.loads(idx_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    _MODULE_INDEX_CACHE[mid] = obj
    return obj

def reset_module_cache() -> None:
    _MODULE_INDEX_CACHE.clear()

def tool_module_load(module_id: str = "expedition_ruins_greyhawk", pdf_path: str = "") -> str:
    """
    Genera (o regenera) el índice del módulo.
    Si pdf_path viene vacío, usa KNOWN_MODULES[module_id].pdf_path
    Guarda en dm/modules/<module_id>.index.json
    """
    mid = _slug(module_id)

    if not pdf_path:
        meta = KNOWN_MODULES.get(mid) or KNOWN_MODULES.get(module_id)
        if not meta:
            return f"Error: module_id '{module_id}' desconocido y pdf_path vacío."
        pdf_path = meta.get("pdf_path", "")

    pdfp = Path(pdf_path)
    if pdfp.exists() and pdfp.parent.as_posix().startswith("/mnt/data"):
        dest = MODULES_DIR / pdfp.name
        if not dest.exists():
            dest.write_bytes(pdfp.read_bytes())
        pdfp = dest

    try:
        idx = _build_module_index(mid, pdfp)
    except Exception as e:
        return f"Error indexando módulo: {e}"

    out_path = _module_index_path(mid)
    out_path.write_text(json.dumps(idx, ensure_ascii=False, indent=2), encoding="utf-8")
    _MODULE_INDEX_CACHE[mid] = idx

    # guarda módulo activo en canon (sin volcar chunks)
    canon = canon_load()
    session = _ensure_dict(canon, "session")

    prev = (session.get("module") or {}).get("progress") or {}
    if not isinstance(prev, dict):
        prev = {}

    progress = {
        "chapter": prev.get("chapter", ""),
        "scene": prev.get("scene", ""),
        "flags": prev.get("flags", []) if isinstance(prev.get("flags", []), list) else [],
    }

    session["module"] = {
        "id": mid,
        "title": idx.get("title", mid),
        "index_path": str(out_path),
        "progress": progress,
    }

    canon_save(canon)

    return json.dumps({
        "ok": True,
        "module_id": mid,
        "title": idx.get("title", mid),
        "index_path": str(out_path),
        "chunks": len(idx.get("chunks", [])),
        "note": "Módulo cargado y marcado como activo en canon.session.module"
    }, ensure_ascii=False, indent=2)

def tool_module_query(query: str, top_k: int = 6, page_min: int = 0, page_max: int = 0) -> str:
    """
    Búsqueda simple por score de términos (substring) sobre chunks.
    Devuelve top_k resultados con id/page/snippet.
    page_min/page_max opcional para acotar (0 = sin límite).
    """
    canon = canon_load()
    mid = _slug(((canon.get("session", {}) or {}).get("module", {}) or {}).get("id", "")) or "expedition_ruins_greyhawk"
    idx = get_module_index(mid)
    if not idx:
        return "Error: no hay índice cargado. Ejecuta module_load() primero."

    q = _norm(query)
    if not q:
        return "Error: query vacío."

    try:
        top_k = int(top_k or 6)
    except Exception:
        top_k = 6
    top_k = max(1, min(top_k, 12))

    terms = [t for t in re.split(r"\s+", q) if t]
    chunks = idx.get("chunks", []) or []

    scored: List[Tuple[int, dict]] = []
    for ch in chunks:
        if not isinstance(ch, dict):
            continue
        page = int(ch.get("page", 0) or 0)
        if page_min and page < int(page_min):
            continue
        if page_max and page > int(page_max):
            continue

        text = ch.get("text", "")
        tnorm = _norm(text)

        score = 0
        # scoring súper simple: cuenta ocurrencias por término
        for term in terms:
            if term and term in tnorm:
                score += 2
        # bonus si la query completa aparece
        if q in tnorm:
            score += 3

        if score > 0:
            scored.append((score, ch))

    scored.sort(key=lambda x: (x[0], -int(x[1].get("page", 0) or 0)), reverse=True)
    hits = [ch for _, ch in scored[:top_k]]

    results = []
    for ch in hits:
        txt = ch.get("text", "") or ""
        snippet = txt[:350].replace("\n", " ").strip()
        results.append({
            "id": ch.get("id"),
            "page": ch.get("page"),
            "snippet": snippet
        })

    return json.dumps({
        "module_id": mid,
        "query": query,
        "count": len(results),
        "results": results
    }, ensure_ascii=False, indent=2)

def tool_module_quote(chunk_id: str, max_chars: int = 1200) -> str:
    """
    Devuelve el texto literal de un chunk (para read-aloud o verificación).
    OJO: úsalo con moderación en mesa para no spoilear.
    """
    canon = canon_load()
    mid = _slug(((canon.get("session", {}) or {}).get("module", {}) or {}).get("id", "")) or "expedition_ruins_greyhawk"
    idx = get_module_index(mid)
    if not idx:
        return "Error: no hay índice cargado. Ejecuta module_load() primero."

    cid = (chunk_id or "").strip()
    if not cid:
        return "Error: chunk_id vacío."

    try:
        max_chars = int(max_chars or 1200)
    except Exception:
        max_chars = 1200
    max_chars = max(200, min(max_chars, 4000))

    for ch in (idx.get("chunks", []) or []):
        if isinstance(ch, dict) and ch.get("id") == cid:
            txt = (ch.get("text", "") or "")[:max_chars]
            return json.dumps({
                "module_id": mid,
                "chunk_id": cid,
                "page": ch.get("page", 0),
                "text": txt
            }, ensure_ascii=False, indent=2)

    return f"Error: chunk_id '{cid}' no encontrado."

def tool_module_set_progress(chapter: str = "", scene: str = "", add_flags_json: str = "[]") -> str:
    """
    Actualiza canon.session.module.progress del módulo activo.
    - chapter/scene: si vienen vacíos, se mantienen.
    - add_flags_json: JSON lista de strings para AÑADIR (sin duplicados).
    """
    canon = canon_load()
    session = _ensure_dict(canon, "session")

    mod = session.get("module") or {}
    if not isinstance(mod, dict) or not (mod.get("id") or ""):
        return "Error: no hay módulo activo en canon.session.module. Ejecuta module_load() primero."

    mid = _slug(mod.get("id", ""))
    prev = (mod.get("progress") or {})
    if not isinstance(prev, dict):
        prev = {}

    # parse flags a añadir
    try:
        add_flags = json.loads(add_flags_json or "[]")
        if not isinstance(add_flags, list):
            add_flags = []
    except Exception:
        add_flags = []

    add_flags = [str(f).strip() for f in add_flags if str(f).strip()]
    prev_flags = prev.get("flags", [])
    if not isinstance(prev_flags, list):
        prev_flags = []

    merged_flags = list(dict.fromkeys([*prev_flags, *add_flags]))  # sin duplicados, preserva orden

    progress = {
        "chapter": chapter if str(chapter or "").strip() else prev.get("chapter", ""),
        "scene": scene if str(scene or "").strip() else prev.get("scene", ""),
        "flags": merged_flags,
    }

    # actualiza SOLO progress, preserva el resto del objeto module
    mod["progress"] = progress
    session["module"] = mod

    canon_save(canon)

    st = _get_state()
    if st is not None:
        st.module_progress["module_id"] = mod.get("id") or st.module_progress.get("module_id", "")
        st.module_progress["chapter"] = progress.get("chapter", "")
        st.module_progress["scene"] = progress.get("scene", "")
        st.module_progress["flags"] = progress.get("flags", [])

    return json.dumps({
        "ok": True,
        "module_id": mid,
        "progress": progress
    }, ensure_ascii=False, indent=2)

# =========================================================
# Persistencia: canon / log / memoria
# =========================================================
def canon_load() -> dict:
    if not CANON_PATH.exists():
        return {}
    try:
        return json.loads(CANON_PATH.read_text(encoding="utf-8") or "{}")
    except Exception:
        return {}

def canon_save(canon: dict) -> None:
    """
    Guarda dm/canon.json de forma segura:
    - crea directorio si no existe
    - escritura atómica (tmp -> replace) para evitar archivos corruptos
    """
    CANON_PATH.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = CANON_PATH.with_suffix(CANON_PATH.suffix + ".tmp")
    data = json.dumps(canon, ensure_ascii=False, indent=2)

    # Escribe a tmp y reemplaza (atómico en la mayoría de OS)
    tmp_path.write_text(data, encoding="utf-8")
    tmp_path.replace(CANON_PATH)

def deep_merge(a: dict, b: dict) -> dict:
    for k, v in b.items():
        if k in a and isinstance(a[k], dict) and isinstance(v, dict):
            a[k] = deep_merge(a[k], v)
        else:
            a[k] = v
    return a

def tool_canon_get(key: str) -> str:
    canon = canon_load()
    # soporte "dot path": party.members[0].name, session.location, etc.
    try:
        return json.dumps(_canon_get_path(canon, key), ensure_ascii=False)
    except Exception:
        return json.dumps(canon.get(key), ensure_ascii=False)

def tool_canon_patch(json_text: str) -> str:
    """
    Aplica un patch JSON (string) al canon con merge recursivo y guardado seguro.
    Soporta que el input venga envuelto en ```json ... ``` (muy típico cuando lo genera el LLM).
    """
    if not isinstance(json_text, str) or not json_text.strip():
        return "Error: json_text vacío."

    raw = json_text.strip()

    # 1) Quitar fences tipo ```json ... ``` o ``` ... ```
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", raw, flags=re.DOTALL | re.IGNORECASE)
    if m:
        raw = m.group(1).strip()

    # 2) Parsear JSON
    try:
        patch = json.loads(raw)
        if not isinstance(patch, dict):
            return "Error: el patch debe ser un JSON objeto (dict)."
    except Exception as e:
        return f"Error: JSON inválido: {e}"

    # 3) Cargar canon actual
    canon = canon_load()
    if not isinstance(canon, dict):
        canon = {}

    # 4) Merge + save (usa tu canon_save atómico)
    canon = deep_merge(canon, patch)
    try:
        canon_save(canon)
    except Exception as e:
        return f"Error: no se pudo guardar canon.json: {e}"

    return "OK. canon.json actualizado (patch aplicado)."

def tool_update_recap(text: str) -> str:
    st = _ACTIVE_STATE
    if st is None:
        return "WARN: no active state"
    st.scene["recap"] = (text or "").strip()
    return "OK: recap actualizado"

def _make_item_instance(item_def: dict, *, qty: int = 1) -> dict:
    """
    Crea una instancia de item para inventario/loot.
    Guardamos lo esencial + un snapshot del texto para no depender del compendio en runtime.
    """
    return {
        "name": item_def.get("name", ""),
        "type": item_def.get("type", ""),
        "rarity": item_def.get("rarity", ""),
        "magic": bool(item_def.get("magic", True)),
        "attunement": bool(item_def.get("attunement", False)),
        "qty": int(qty or 1),
        "text": list(item_def.get("text", []) or []),
        "source": list(item_def.get("source", []) or []),
        "tags": list(item_def.get("tags", []) or []),
    }

def canon_add_item_to_party(canon: dict, member_name: str, item_name: str, qty: int = 1) -> bool:
    member = _find_party_member(canon, member_name)
    if not member:
        return False

    item_def = _get_item_def_by_name(item_name)
    if not item_def:
        return False

    inv = member.setdefault("inventory", [])
    inv.append(_make_item_instance(item_def, qty=qty))
    return True

def canon_add_item_to_loot(canon: dict, item_name: str, qty: int = 1) -> bool:
    item_def = _get_item_def_by_name(item_name)
    if not item_def:
        return False
    loot = canon.setdefault("loot", [])
    loot.append(_make_item_instance(item_def, qty=qty))
    return True

def tool_give_item(member: str, item: str, qty: int = 1) -> str:
    canon = canon_load()
    ok = canon_add_item_to_party(canon, member, item, qty=int(qty or 1))
    if not ok:
        return f"Error: no pude dar '{item}' a '{member}'. (¿existe el PJ y el item en dm/items.json?)"
    canon_save(canon)
    return f"OK. '{item}' x{int(qty or 1)} añadido al inventario de {member}."

def tool_add_loot(item: str, qty: int = 1) -> str:
    canon = canon_load()
    ok = canon_add_item_to_loot(canon, item, qty=int(qty or 1))
    if not ok:
        return f"Error: no pude añadir '{item}' al loot. (¿existe en dm/items.json?)"
    canon_save(canon)
    return f"OK. Loot: '{item}' x{int(qty or 1)} añadido."

def tool_log_event(text: str) -> str:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(text.strip() + "\n")
    return "OK. Evento registrado."

def memory_load() -> dict:
    if not MEMORY_PATH.exists():
        MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        MEMORY_PATH.write_text("{}", encoding="utf-8")
    try:
        return json.loads(MEMORY_PATH.read_text(encoding="utf-8") or "{}")
    except Exception:
        return {}

def memory_save(mem: dict) -> None:
    MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = MEMORY_PATH.with_suffix(MEMORY_PATH.suffix + ".tmp")
    tmp.write_text(json.dumps(mem, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(MEMORY_PATH)

def tool_memory_get(key: str) -> str:
    mem = memory_load()
    return json.dumps(mem.get(key), ensure_ascii=False)

def tool_memory_set(key: str, value: str) -> str:
    mem = memory_load()
    mem[key] = value
    memory_save(mem)
    return f"OK. Guardado en memoria['{key}']."

# =========================================================
# Helpers generales
# =========================================================
def _norm(s: str) -> str:
    return " ".join((s or "").strip().lower().split())

def _ensure_list(d: dict, key: str) -> list:
    if key not in d or not isinstance(d[key], list):
        d[key] = []
    return d[key]

def _ensure_dict(d: dict, key: str) -> dict:
    if key not in d or not isinstance(d[key], dict):
        d[key] = {}
    return d[key]

def _canon_get_path(obj: Any, path: str) -> Any:
    """
    Soporta:
      - foo.bar
      - foo.bar[0].baz
    """
    if not path or not isinstance(path, str):
        return None
    cur = obj
    parts = [p for p in path.split(".") if p]
    for part in parts:
        m = re.fullmatch(r"([a-zA-Z0-9_\-]+)(?:\[(\d+)\])?", part)
        if not m:
            return None
        key, idx = m.group(1), m.group(2)
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
        if idx is not None:
            if not isinstance(cur, list):
                return None
            i = int(idx)
            if i < 0 or i >= len(cur):
                return None
            cur = cur[i]
    return cur

def _find_party_member(canon: dict, name: str) -> Optional[dict]:
    target = _norm(name)
    for m in canon.get("party", {}).get("members", []):
        if _norm(m.get("name", "")) == target:
            return m
    for m in canon.get("party", {}).get("members", []):
        if target and target in _norm(m.get("name", "")):
            return m
    return None

def _find_enemy(canon: dict, name: str) -> Optional[Tuple[str, dict]]:
    enemies = canon.get("enemies", {}) or {}
    target = _norm(name)
    for k, v in enemies.items():
        if _norm(k) == target:
            return (k, v)
    for k, v in enemies.items():
        if target and target in _norm(k):
            return (k, v)
    return None

def _get_target_container(canon: dict, target: str) -> Optional[Tuple[str, dict, str]]:
    """
    Devuelve (kind, obj, canonical_name)
    kind: "party" | "enemy"
    """
    # 1) Party
    m = _find_party_member(canon, target)
    if m:
        return ("party", m, m.get("name", target))

    # 2) Enemigo ya existente
    fe = _find_enemy(canon, target)
    if fe:
        k, v = fe
        return ("enemy", v, k)

    # 3) Enemigo NO existente -> intenta auto-crear desde bestiary.json
    spawned = _ensure_enemy_from_bestiary(canon, target)
    if spawned:
        k, v = spawned
        return ("enemy", v, k)

    # 4) Si hay combate activo y el objetivo está en iniciativa, crea stub
    if _ensure_enemy_stub_from_combat(canon, target):
        fe2 = _find_enemy(canon, target)
        if fe2:
            k, v = fe2
            return ("enemy", v, k)

    return None

def _ensure_enemy_stub_anywhere(canon: dict, name: str, defaults: dict | None = None) -> bool:
    """
    Garantiza que exista canon['enemies'][name]. Útil cuando el LLM narra enemigos
    sin registrarlos previamente. Devuelve True si existe o se creó.

    IMPORTANTE: los stubs ya NO usan HP=999 (eso rompe la consistencia). Usamos valores
    modestos y marcamos stub=True para que luego pueda “enriquecerse” vía bestiary/spawn.
    """
    name = (name or "").strip()
    if not name:
        return False

    # Si es PJ, no crear enemigo
    if _find_party_member(canon, name):
        return False

    canon.setdefault("enemies", {})
    enemies = canon["enemies"]

    def _hp_defaults() -> tuple[int, int]:
        # Defaults razonables para un “desconocido”
        return (10, 10)  # (hp_current, max_hp)

    if name in enemies:
        e = enemies[name]
        e.setdefault("name", name)
        e.setdefault("conditions", [])
        e.setdefault("ac", 10)

        # Unificar runtime HP
        hp_cur = e.get("hp_current", e.get("hp", None))
        hp_max = e.get("max_hp", e.get("hp_max", None))
        if hp_cur is None or hp_max is None:
            dcur, dmax = _hp_defaults()
            hp_cur = dcur if hp_cur is None else hp_cur
            hp_max = dmax if hp_max is None else hp_max

        try:
            hp_cur_i = int(hp_cur)
        except Exception:
            hp_cur_i = _hp_defaults()[0]
        try:
            hp_max_i = int(hp_max)
        except Exception:
            hp_max_i = max(hp_cur_i, _hp_defaults()[1])

        e["hp_current"] = max(0, hp_cur_i)
        e["max_hp"] = max(1, hp_max_i)

        # Mantener compatibilidad con claves antiguas
        e["hp"] = e["hp_current"]
        e["hp_max"] = e["max_hp"]

        # Si no estaba marcado, que siga indicando stub si no tiene source
        e.setdefault("stub", True)

        # aplicar defaults extra si vienen
        if defaults and isinstance(defaults, dict):
            for k, v in defaults.items():
                if v is not None:
                    e[k] = v

        canon_save(canon)
        return True

    dcur, dmax = _hp_defaults()
    stub = {
        "name": name,
        "stub": True,
        "ac": 10,
        "hp_current": dcur,
        "max_hp": dmax,
        "hp": dcur,       # compat
        "hp_max": dmax,   # compat
        "conditions": [],
    }
    if defaults and isinstance(defaults, dict):
        stub.update({k: v for k, v in defaults.items() if v is not None})

    enemies[name] = stub
    canon_save(canon)
    return True

def _ensure_enemy_stub_from_combat(canon: dict, target: str) -> bool:
    """
    Si hay combate activo y el target está en la iniciativa con side != party,
    crea un enemigo stub en canon['enemies'] para permitir ataques/condiciones.

    IMPORTANTE: los stubs ya NO usan HP=999 (eso rompe la consistencia).
    """
    combat = canon.get("combat", {}) or {}
    if not combat.get("active"):
        return False

    order = combat.get("order", []) or []
    t_norm = (target or "").strip().lower()

    # buscar en iniciativa
    for o in order:
        name = str(o.get("name", "")).strip()
        side = str(o.get("side", "")).strip().lower()
        if not name:
            continue
        if name.lower() == t_norm and side != "party":
            canon.setdefault("enemies", {})
            enemies = canon["enemies"]

            # si ya existe, ok
            if name in enemies:
                return True

            enemies[name] = {
                "name": name,
                "stub": True,
                "ac": 10,
                "hp_current": 10,
                "max_hp": 10,
                "hp": 10,        # compat
                "hp_max": 10,    # compat
                "conditions": [],
            }
            canon_save(canon)
            return True

    return False

def _unique_enemy_instance_name(canon: dict, base_name: str) -> str:
    enemies = canon.get("enemies", {}) or {}
    if base_name not in enemies:
        return base_name
    i = 2
    while True:
        candidate = f"{base_name} #{i}"
        if candidate not in enemies:
            return candidate
        i += 1


def _speed_to_int(speed_val: Any, default_speed: int = 30) -> int:
    """
    Tu compendio trae speed a veces como dict:
      {"walk": 10, "swim": 40}
    o a veces como int.
    Aquí normalizamos a un entero usable (preferimos walk).
    """
    if isinstance(speed_val, dict):
        if "walk" in speed_val:
            try:
                return int(speed_val.get("walk") or default_speed)
            except Exception:
                return default_speed
        # si no hay walk, usamos el mayor valor numérico
        vals = []
        for v in speed_val.values():
            try:
                vals.append(int(v))
            except Exception:
                pass
        return max(vals) if vals else default_speed

    try:
        return int(speed_val)
    except Exception:
        return default_speed


def _ensure_enemy_from_bestiary(canon: dict, target_name: str) -> Optional[Tuple[str, dict]]:
    """
    Si target_name no existe en canon["enemies"], intenta crearlo desde dm/bestiary.json
    (estructura compendio: indexes.by_name + monsters{}).
    Devuelve (instance_name, enemy_obj) o None.
    """
    # Si ya existe, no hacemos nada
    fe = _find_enemy(canon, target_name)
    if fe:
        return fe

    mon_def = _get_monster_def_by_name(target_name)
    if not mon_def:
        return None

    enemies = _ensure_dict(canon, "enemies")
    base = (mon_def.get("name") or target_name).strip() or target_name
    instance_name = _unique_enemy_instance_name(canon, base)

    enemies[instance_name] = deepcopy(mon_def)
    enemies[instance_name]["_bestiary_name"] = mon_def.get("name", base)

    # -----------------------------
    # Runtime HP unificado: max_hp + hp_current
    # -----------------------------
    hp_raw = enemies[instance_name].get("hp")
    if isinstance(hp_raw, dict):
        max_hp = int(hp_raw.get("avg", 1) or 1)
    else:
        try:
            max_hp = int(hp_raw or 1)
        except Exception:
            max_hp = 1

    enemies[instance_name]["max_hp"] = max_hp
    enemies[instance_name]["hp_current"] = max_hp

    # Condiciones runtime
    if "conditions" not in enemies[instance_name] or not isinstance(enemies[instance_name]["conditions"], list):
        enemies[instance_name]["conditions"] = []

    # Movimiento runtime + speed normalizada (walk / max)
    sp_raw = enemies[instance_name].get("speed", 30)
    enemies[instance_name]["speed"] = _speed_to_int(sp_raw, default_speed=30)
    if "move_left" not in enemies[instance_name]:
        enemies[instance_name]["move_left"] = 0

    return (instance_name, enemies[instance_name])


def _load_json_file(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def get_bestiary() -> Optional[dict]:
    """
    Devuelve el bestiary como COMPENDIO (indexes.by_name + monsters{}).
    """
    global _BESTIARY_COMP_CACHE
    if _BESTIARY_COMP_CACHE is None:
        _BESTIARY_COMP_CACHE = _load_json_file(BESTIARY_PATH)
    return _BESTIARY_COMP_CACHE


def reset_compendium_caches() -> None:
    global _BESTIARY_COMP_CACHE, _BESTIARY_FLAT_CACHE, _ITEMS_CACHE, _SPELLS_CACHE
    _BESTIARY_COMP_CACHE = None
    _BESTIARY_FLAT_CACHE = None
    _ITEMS_CACHE = None
    _SPELLS_CACHE = None


def get_items_compendium() -> Optional[dict]:
    global _ITEMS_CACHE
    if _ITEMS_CACHE is None:
        _ITEMS_CACHE = _load_json_file(ITEMS_PATH)
    return _ITEMS_CACHE


def get_spells_compendium() -> Optional[dict]:
    global _SPELLS_CACHE
    if _SPELLS_CACHE is None:
        _SPELLS_CACHE = _load_json_file(SPELLS_PATH)
    return _SPELLS_CACHE


def get_modifiers_compendium() -> list:
    global _MODIFIERS_CACHE
    if _MODIFIERS_CACHE is not None:
        return _MODIFIERS_CACHE

    chosen = None
    for p in MODIFIERS_PATHS:
        try:
            if isinstance(p, Path) and p.exists():
                chosen = p
                break
        except Exception:
            pass

    if not chosen:
        _MODIFIERS_CACHE = []
        return _MODIFIERS_CACHE

    data = _load_json_file(chosen)
    _MODIFIERS_CACHE = data if isinstance(data, list) else []
    return _MODIFIERS_CACHE


def _ctx_has_feature(ctx: dict, needle: str) -> bool:
    needle = _norm(needle)
    feats = ctx.get("actor_features_lc", [])
    return any(needle in f for f in feats)

_MODE_ALIASES = {
    # sharpshooter
    "sharpshooter": "sharpshooter",
    "sharp_shooter": "sharpshooter",
    "sharp shooter": "sharpshooter",
    "tirador de elite": "sharpshooter",
    "tirador de élite": "sharpshooter",

    # great weapon master
    "gwm": "gwm",
    "greatweaponmaster": "gwm",
    "great weapon master": "gwm",
    "gran maestro de armas": "gwm",
    "maestro de armas a dos manos": "gwm",

    # sneak attack
    "sneak_attack": "sneak_attack",
    "sneak attack": "sneak_attack",
    "ataque furtivo": "sneak_attack",

    # dread ambusher
    "dread_ambusher": "dread_ambusher",
    "dread ambusher": "dread_ambusher",
    "gloomstalker": "dread_ambusher",
    "gloom stalker": "dread_ambusher",
}

def _canon_mode_token(s: str) -> str:
    s0 = _norm(s)
    if not s0:
        return ""
    # normaliza separadores para capturar "great-weapon-master", etc.
    s1 = re.sub(r"[\s\-]+", " ", s0).strip()
    if s1 in _MODE_ALIASES:
        return _MODE_ALIASES[s1]
    # versión sin espacios/underscore
    s2 = re.sub(r"[^a-z0-9]+", "", s0)
    if s2 in _MODE_ALIASES:
        return _MODE_ALIASES[s2]
    return s0

_MODE_ALIASES = {
    "sharpshooter": "sharpshooter",
    "sharp shooter": "sharpshooter",
    "sharp_shooter": "sharpshooter",
    "tirador de elite": "sharpshooter",
    "tirador de élite": "sharpshooter",

    "gwm": "gwm",
    "great weapon master": "gwm",
    "greatweaponmaster": "gwm",
    "gran maestro de armas": "gwm",

    "sneak_attack": "sneak_attack",
    "sneak attack": "sneak_attack",
    "ataque furtivo": "sneak_attack",

    "dread_ambusher": "dread_ambusher",
    "dread ambusher": "dread_ambusher",
    "gloom stalker": "dread_ambusher",
    "gloomstalker": "dread_ambusher",
}

def _canon_mode_token(s: str) -> str:
    s0 = _norm(s)
    if not s0:
        return ""
    s1 = re.sub(r"[\s\-]+", " ", s0).strip()
    if s1 in _MODE_ALIASES:
        return _MODE_ALIASES[s1]
    s2 = re.sub(r"[^a-z0-9]+", "", s0)
    if s2 in _MODE_ALIASES:
        return _MODE_ALIASES[s2]
    return s0

def _ctx_modes_include(ctx: dict, mode: str) -> bool:
    want = _canon_mode_token(mode)
    return want in (ctx.get("modes", set()) or set())

def _ctx_weapon_has(ctx: dict, tag: str) -> bool:
    return _norm(tag) in (ctx.get("weapon_tags", set()) or set())


def _ctx_weapon_lacks(ctx: dict, tag: str) -> bool:
    return not _ctx_weapon_has(ctx, tag)


def _mod_if_matches(mod_if: dict, ctx: dict) -> bool:
    if not isinstance(mod_if, dict):
        return True

    # attack_type
    if "attack_type" in mod_if:
        want = mod_if["attack_type"]
        if isinstance(want, list):
            if ctx.get("attack_type") not in [str(x).lower() for x in want]:
                return False
        else:
            if ctx.get("attack_type") != str(want).lower():
                return False

    # actor_feature_contains
    if "actor_feature_contains" in mod_if:
        want = mod_if["actor_feature_contains"]
        if isinstance(want, list):
            if not any(_ctx_has_feature(ctx, x) for x in want):
                return False
        else:
            if not _ctx_has_feature(ctx, want):
                return False

    # mode_includes
    if "mode_includes" in mod_if:
        want = mod_if["mode_includes"]
        if isinstance(want, list):
            if not any(_ctx_modes_include(ctx, x) for x in want):
                return False
        else:
            if not _ctx_modes_include(ctx, want):
                return False

    # weapon_has
    if "weapon_has" in mod_if:
        want = mod_if["weapon_has"]
        if isinstance(want, list):
            if not any(_ctx_weapon_has(ctx, x) for x in want):
                return False
        else:
            if not _ctx_weapon_has(ctx, want):
                return False

    # weapon_lacks
    if "weapon_lacks" in mod_if:
        want = mod_if["weapon_lacks"]
        if isinstance(want, list):
            if not all(_ctx_weapon_lacks(ctx, x) for x in want):
                return False
        else:
            if not _ctx_weapon_lacks(ctx, want):
                return False

    return True


def _apply_attack_modifiers(ctx: dict) -> tuple[int, int, list]:
    """
    Devuelve: (to_hit_delta, damage_bonus_flat, notes[])
    - Primero aplica rules-as-data desde modifiers.json
    - Luego aplica fallback builtin para SS/GWM si no se aplicaron (para evitar “no matchea y se rompe”)
    """
    to_hit = 0
    dmg_bonus = 0
    notes = []

    applied_ids: set[str] = set()

    mods = get_modifiers_compendium() or []
    for m in mods:
        if not isinstance(m, dict):
            continue
        when = (m.get("when") or "").strip().lower()
        if when != "attack":
            continue
        if not _mod_if_matches(m.get("if") or {}, ctx):
            continue

        ap = m.get("apply") or {}
        mid = str(m.get("id") or "").strip()
        if mid:
            applied_ids.add(mid)

        try:
            to_hit += int(ap.get("to_hit", 0) or 0)
        except Exception:
            pass
        try:
            dmg_bonus += int(ap.get("damage_bonus", 0) or 0)
        except Exception:
            pass

        note = ap.get("note") or mid
        if note:
            notes.append(str(note))

    # ----------------------------
    # Fallback builtin (SS / GWM)
    # ----------------------------
    # Sharpshooter: ranged + mode sharpshooter => -5/+10
    if _ctx_modes_include(ctx, "sharpshooter") and ctx.get("attack_type") == "ranged":
        if "sharpshooter_power_shot" not in applied_ids:
            to_hit -= 5
            dmg_bonus += 10
            notes.append("Sharpshooter (-5/+10) [builtin]")

    # GWM: melee + heavy + mode gwm => -5/+10
    if _ctx_modes_include(ctx, "gwm") and ctx.get("attack_type") == "melee":
        if _ctx_weapon_has(ctx, "heavy"):
            if "gwm_power_attack" not in applied_ids:
                to_hit -= 5
                dmg_bonus += 10
                notes.append("GWM (-5/+10) [builtin]")

    return to_hit, dmg_bonus, notes


def _get_spell_def_by_name(name: str) -> Optional[dict]:
    """
    Devuelve el objeto spell (definición) desde dm/spells.json dado un nombre.
    Requiere estructura:
      { "indexes": {"by_name": {...}}, "spells": { "<id>": {...} } }
    """
    comp = get_spells_compendium()
    if not comp:
        return None
    spell_id = _compendium_lookup_by_name(comp, name)
    if not spell_id:
        return None
    return (comp.get("spells") or {}).get(spell_id)


def _compendium_lookup_by_name(comp: dict, name: str) -> Optional[str]:
    """
    Devuelve el ID interno del compendio usando indexes.by_name.
    """
    if not comp or not name:
        return None
    idx = (comp.get("indexes") or {}).get("by_name") or {}
    return idx.get(_norm(name))


def _get_item_def_by_name(name: str) -> Optional[dict]:
    """
    Devuelve el objeto item (definición) desde dm/items.json dado un nombre.
    """
    comp = get_items_compendium()
    if not comp:
        return None
    item_id = _compendium_lookup_by_name(comp, name)
    if not item_id:
        return None
    return (comp.get("items") or {}).get(item_id)


def _get_monster_def_by_name(name: str) -> Optional[dict]:
    """
    (opcional) si quieres mantener simetría con items; si ya tienes esto para bestiary, no lo dupliques.
    """
    comp = get_bestiary()
    if not comp:
        return None
    monster_id = _compendium_lookup_by_name(comp, name)
    if not monster_id:
        return None
    return (comp.get("monsters") or {}).get(monster_id)

# =========================================================
# Bestiary helpers (compendio) — match flexible + normalización
# =========================================================

# Alias ES -> EN mínimo (ampliable). Solo se aplica al primer token.
_MONSTER_FIRSTWORD_ALIASES = {
    "cultista": "cultist",
    "bandido": "bandit",
    "guardia": "guard",
    "capitán": "captain",
    "capitan": "captain",
    "acólito": "acolyte",
    "acolito": "acolyte",
    "lobo": "wolf",
    "oso": "bear",
    "orco": "orc",
    "trol": "troll",
}

def _monster_alias_candidates(name: str) -> list[str]:
    """
    Genera candidatos de nombre para mejorar el match con bestiary (ES->EN básico).
    """
    name = (name or "").strip()
    if not name:
        return []
    parts = name.split()
    if not parts:
        return [name]
    first = parts[0].lower()
    if first in _MONSTER_FIRSTWORD_ALIASES:
        parts2 = [_MONSTER_FIRSTWORD_ALIASES[first]] + parts[1:]
        return [name, " ".join(parts2)]
    return [name]

def _strip_instance_suffix(name: str) -> str:
    """
    Quita sufijos típicos de “instancia”: 'A', '#2', '2', etc.
    Ej: 'Cultist A' -> 'Cultist', 'Goblin #2' -> 'Goblin'
    """
    s = (name or "").strip()
    if not s:
        return s
    # 'X #2' o 'X 2'
    s = re.sub(r"\s+#\d+\s*$", "", s)
    s = re.sub(r"\s+\d+\s*$", "", s)
    # 'X A' / 'X B'
    s = re.sub(r"\s+[A-Z]\s*$", "", s)
    return s.strip()

def _get_monster_def_fuzzy(name: str) -> Optional[dict]:
    """
    Busca un monstruo en el compendio por:
      - match exacto en indexes.by_name
      - match quitando sufijos (A/#2)
      - match por alias ES->EN básico
      - match difuso (difflib) sobre indexes.by_name
    """
    comp = get_bestiary()
    if not comp or not name:
        return None
    idx = (comp.get("indexes") or {}).get("by_name") or {}
    monsters = (comp.get("monsters") or {})

    # 1) exacto / por variantes
    candidates = []
    for base in _monster_alias_candidates(name):
        candidates.append(base)
        candidates.append(_strip_instance_suffix(base))

    for cand in candidates:
        mid = idx.get(_norm(cand))
        if mid and mid in monsters:
            return monsters[mid]

    # 2) difuso
    keys = list(idx.keys())
    for cand in candidates:
        nn = _norm(cand)
        close = difflib.get_close_matches(nn, keys, n=1, cutoff=0.84)
        if close:
            mid = idx.get(close[0])
            if mid and mid in monsters:
                return monsters[mid]

    return None

def _import_monster_as_enemy(mon: dict, instance_name: str) -> dict:
    """
    Convierte un statblock del bestiary a un enemigo “runtime” estable en canon['enemies'].
    Unifica HP en: max_hp + hp_current (+ compat: hp/hp_max).
    """
    # AC
    ac_raw = mon.get("ac", 10)
    ac = 10
    if isinstance(ac_raw, list) and ac_raw:
        first = ac_raw[0]
        if isinstance(first, dict) and "ac" in first:
            ac = int(first.get("ac") or 10)
        else:
            try:
                ac = int(first)
            except Exception:
                ac = 10
    else:
        try:
            ac = int(ac_raw)
        except Exception:
            ac = 10

    # HP
    hp_raw = mon.get("hp", 10)
    max_hp = 10
    if isinstance(hp_raw, dict):
        # tu compendio usa 'avg'; soportamos 'average' también
        try:
            max_hp = int(hp_raw.get("avg") or hp_raw.get("average") or 10)
        except Exception:
            max_hp = 10
    elif isinstance(hp_raw, str):
        # formatos tipo "45 (6d8+18)"
        m = re.match(r"^\s*(\d+)", hp_raw)
        if m:
            max_hp = int(m.group(1))
        else:
            max_hp = 10
    else:
        try:
            max_hp = int(hp_raw)
        except Exception:
            max_hp = 10

    max_hp = max(1, int(max_hp))

    entry = {
        "name": instance_name,
        "stub": False,
        "source": "bestiary",
        "_bestiary_name": str(mon.get("name") or instance_name),
        "ac": ac,

        # runtime HP unificado
        "max_hp": max_hp,
        "hp_current": max_hp,

        # compat keys (hay funciones que aún leen estas)
        "hp": max_hp,
        "hp_max": max_hp,

        "conditions": [],
        "speed": _speed_to_int(mon.get("speed", 30), default_speed=30),
        "move_left": 0,

        # guardar referencia mínima para auditoría / futuros tools
        "raw": mon,
    }

    # CR si existe
    if "cr" in mon:
        cr_val = _parse_cr(mon.get("cr"))
        if cr_val is not None:
            entry["cr"] = cr_val

    # Abilities si existen como keys
    abil = {}
    for k in ["str", "dex", "con", "int", "wis", "cha"]:
        if k in mon:
            try:
                abil[k] = int(mon.get(k))
            except Exception:
                pass
    if abil:
        entry["abilities"] = abil

    return entry

def _has_condition(obj: dict, name: str) -> bool:
    name = (name or "").upper()
    for c in obj.get("conditions", []) or []:
        if str(c.get("name", "")).upper() == name:
            return True
    return False


def _ensure_conditions(obj: dict) -> list:
    return _ensure_list(obj, "conditions")

# =========================================================
# XP System (híbrido: progreso + bichos) + escalado por nº PJs
# =========================================================

# Tabla XP por CR (D&D 5e)
CR_XP = {
    0: 10,
    0.125: 25,   # 1/8
    0.25: 50,    # 1/4
    0.5: 100,    # 1/2
    1: 200,
    2: 450,
    3: 700,
    4: 1100,
    5: 1800,
    6: 2300,
    7: 2900,
    8: 3900,
    9: 5000,
    10: 5900,
    11: 7200,
    12: 8400,
    13: 10000,
    14: 11500,
    15: 13000,
    16: 15000,
    17: 18000,
    18: 20000,
    19: 22000,
    20: 25000,
    21: 33000,
    22: 41000,
    23: 50000,
    24: 62000,
    25: 75000,
    26: 90000,
    27: 105000,
    28: 120000,
    29: 135000,
    30: 155000,
}

PROGRESS_XP_PER_PC = {
    "minor": 25,
    "standard": 75,
    "major": 200,
}

def _party_size(canon: dict) -> int:
    members = canon.get("party", {}).get("members", [])
    if isinstance(members, list) and members:
        return len(members)
    # fallback si alguien lo guarda explícito
    try:
        return int(canon.get("party", {}).get("size", 0) or 0)
    except Exception:
        return 0

def _ensure_xp_struct(canon: dict) -> dict:
    xp = canon.get("xp")
    if not isinstance(xp, dict):
        xp = {}
        canon["xp"] = xp
    if "total" not in xp:
        xp["total"] = 0
    if "log" not in xp or not isinstance(xp["log"], list):
        xp["log"] = []
    if "by_character" not in xp or not isinstance(xp["by_character"], dict):
        xp["by_character"] = {}
    return xp

def _add_xp_to_party(canon: dict, amount_total: int, reason: str, meta: Optional[dict] = None) -> dict:
    """
    amount_total: XP total del grupo (antes de repartir) o ya repartido (según como lo uses).
    Aquí lo tratamos como XP DEL GRUPO y opcionalmente lo repartimos a by_character.
    """
    xp = _ensure_xp_struct(canon)
    amount_total = int(amount_total or 0)
    if amount_total <= 0:
        return {"ok": False, "error": "amount_total debe ser > 0"}

    xp["total"] = int(xp.get("total", 0) or 0) + amount_total

    # Reparto por personaje (solo informativo; el total de grupo sigue siendo el “source of truth”)
    members = canon.get("party", {}).get("members", [])
    n = _party_size(canon)
    if n <= 0:
        n = 1
    per_pc = amount_total // n

    if isinstance(members, list):
        for m in members:
            nm = (m.get("name") or "").strip()
            if not nm:
                continue
            xp["by_character"][nm] = int(xp["by_character"].get(nm, 0) or 0) + per_pc

    entry = {
        "amount_total": amount_total,
        "per_pc": per_pc,
        "party_size": n,
        "reason": reason,
    }
    if meta and isinstance(meta, dict):
        entry["meta"] = meta

    xp["log"].append(entry)
    return {"ok": True, **entry, "xp_total_now": xp["total"]}

def _parse_cr(cr_val: Any) -> Optional[float]:
    """
    Acepta:
      - 10
      - "10"
      - "1/2"
      - "1/4"
      - "1/8"
      - 0.5
    """
    if cr_val is None:
        return None
    if isinstance(cr_val, (int, float)):
        return float(cr_val)
    s = str(cr_val).strip()
    if not s:
        return None
    if "/" in s:
        try:
            a, b = s.split("/", 1)
            return float(a) / float(b)
        except Exception:
            return None
    try:
        return float(s)
    except Exception:
        return None

def _xp_for_cr(cr: Any) -> int:
    f = _parse_cr(cr)
    if f is None:
        return 0
    # Normaliza fracciones típicas
    if abs(f - 0.125) < 1e-6:
        f = 0.125
    if abs(f - 0.25) < 1e-6:
        f = 0.25
    if abs(f - 0.5) < 1e-6:
        f = 0.5
    return int(CR_XP.get(f, 0))

def tool_xp_status() -> str:
    canon = canon_load()
    xp = _ensure_xp_struct(canon)
    n = _party_size(canon) or 1
    out = {
        "party_size": n,
        "xp_total": int(xp.get("total", 0) or 0),
        "per_pc_estimate": int(xp.get("total", 0) or 0) // n,
        "by_character": xp.get("by_character", {}),
        "log_tail": (xp.get("log", []) or [])[-10:],
    }
    return json.dumps(out, ensure_ascii=False, indent=2)

def tool_xp_progress(kind: str, reason: str = "") -> str:
    """
    kind: minor | standard | major
    """
    kind = (kind or "").strip().lower()
    if kind not in PROGRESS_XP_PER_PC:
        return "Error: kind debe ser minor | standard | major"

    canon = canon_load()
    n = _party_size(canon)
    if n <= 0:
        return "Error: no hay party.members en canon; no puedo calcular party_size."

    per_pc = PROGRESS_XP_PER_PC[kind]
    total = per_pc * n

    res = _add_xp_to_party(
        canon,
        amount_total=total,
        reason=f"PROGRESS:{kind} {reason}".strip(),
        meta={"kind": kind, "per_pc": per_pc},
    )
    canon_save(canon)
    return json.dumps(res, ensure_ascii=False, indent=2)

def tool_xp_kill(enemy: str) -> str:
    """
    Otorga XP por un enemigo muerto basándose en su CR.
    Busca en canon.enemies[enemy]; si existe campo cr -> usa tabla.
    """
    canon = canon_load()
    fe = _find_enemy(canon, enemy)
    if not fe:
        return f"Error: enemigo '{enemy}' no existe en canon.enemies"

    name, obj = fe
    cr = obj.get("cr")
    xp_val = _xp_for_cr(cr)
    if xp_val <= 0:
        return f"Error: enemigo '{name}' no tiene CR válido (cr={cr})."

    n = _party_size(canon) or 1
    res = _add_xp_to_party(
        canon,
        amount_total=xp_val,
        reason=f"KILL:{name}",
        meta={"enemy": name, "cr": cr, "xp": xp_val, "per_pc_estimate": xp_val // n},
    )
    canon_save(canon)
    return json.dumps(res, ensure_ascii=False, indent=2)

def tool_level_status() -> str:
    """
    Devuelve nivel estimado del grupo (por XP) usando XP total / nº PJs.
    También devuelve el umbral del siguiente nivel.
    """
    canon = canon_load()
    n = _party_size(canon) or 1
    xp_per_pc = _party_xp_per_pc(canon)
    lvl = _level_for_xp(xp_per_pc)
    nxt = _next_level_threshold(lvl)

    out = {
        "party_size": n,
        "xp_total_group": int(_ensure_xp_struct(canon).get("total", 0) or 0),
        "xp_per_pc": xp_per_pc,
        "level_by_xp": lvl,
        "next_level": (lvl + 1) if lvl < 20 else None,
        "next_level_threshold_xp_per_pc": nxt,
        "xp_needed_per_pc": (max(0, nxt - xp_per_pc) if nxt is not None else 0),
    }
    return json.dumps(out, ensure_ascii=False, indent=2)

def tool_level_check_up() -> str:
    """
    Compara el nivel "current_level" guardado en canon (si existe) con el nivel por XP.
    Si hay subida, lo registra y actualiza canon.party.level (opcional).
    """
    canon = canon_load()
    party = canon.get("party", {}) if isinstance(canon.get("party", {}), dict) else {}
    current_level = party.get("level")

    # Si no hay nivel guardado, inferimos el nivel mínimo observado en members (si existe),
    # y si no, usamos el nivel por XP.
    if current_level is None:
        members = party.get("members", [])
        if isinstance(members, list) and members:
            levels = []
            for m in members:
                try:
                    levels.append(int(m.get("level", 0) or 0))
                except Exception:
                    pass
            current_level = max(levels) if levels else None

    xp_level = _level_for_xp(_party_xp_per_pc(canon))
    if current_level is None:
        current_level = xp_level

    try:
        current_level = int(current_level)
    except Exception:
        current_level = xp_level

    up = xp_level > current_level

    # (Opcional) actualizar canon.party.level si sube
    if up:
        party = _ensure_dict(canon, "party")
        party["level"] = xp_level

        # También puedes sincronizar en cada member (solo si quieres)
        members = party.get("members", [])
        if isinstance(members, list):
            for m in members:
                if isinstance(m, dict):
                    m["level"] = xp_level

        # Log dentro del XP log para auditoría
        xp = _ensure_xp_struct(canon)
        xp["log"].append({
            "event": "LEVEL_UP",
            "from": current_level,
            "to": xp_level,
            "xp_per_pc": _party_xp_per_pc(canon),
        })

        canon_save(canon)

    out = {
        "current_level_saved": current_level,
        "level_by_xp": xp_level,
        "level_up": up,
        "new_level": xp_level if up else None
    }
    return json.dumps(out, ensure_ascii=False, indent=2)

def tool_level_up_announce() -> str:
    """
    Si hay subida de nivel por XP, actualiza canon.party.level (y members.level) y devuelve
    un mensaje listo para el DM: "Subís a nivel X".
    Si no hay subida, devuelve "Aún no subís de nivel."
    """
    # Reutilizamos la lógica existente
    result = json.loads(tool_level_check_up())

    if result.get("level_up"):
        lvl = result.get("level_by_xp")
        return f"✅ Subís a nivel {lvl}."
    return "Aún no subís de nivel."

# =========================================================
# Level by XP (umbrales por PJ, D&D 5e)
# =========================================================

# Umbrales de XP por PJ para alcanzar ese nivel (PHB 5e).
# Nota: nivel 1 empieza en 0 XP.
LEVEL_XP_THRESHOLDS = {
    1: 0,
    2: 300,
    3: 900,
    4: 2700,
    5: 6500,
    6: 14000,
    7: 23000,
    8: 34000,
    9: 48000,
    10: 64000,
    11: 85000,
    12: 100000,
    13: 120000,
    14: 140000,
    15: 165000,
    16: 195000,
    17: 225000,
    18: 265000,
    19: 305000,
    20: 355000,
}

def _level_for_xp(xp_per_pc: int) -> int:
    """
    Devuelve el nivel (1-20) correspondiente a xp_per_pc según umbrales.
    """
    try:
        x = int(xp_per_pc or 0)
    except Exception:
        x = 0

    lvl = 1
    for L in range(1, 21):
        if x >= LEVEL_XP_THRESHOLDS[L]:
            lvl = L
    return lvl

def _next_level_threshold(level: int) -> Optional[int]:
    if level >= 20:
        return None
    return LEVEL_XP_THRESHOLDS.get(level + 1)

def _party_xp_per_pc(canon: dict) -> int:
    xp = _ensure_xp_struct(canon)
    total = int(xp.get("total", 0) or 0)
    n = _party_size(canon) or 1
    return total // n

# =========================================================
# Spells (Conjuros)
# =========================================================
def tool_spell_info(name: str) -> str:
    """
    Devuelve la definición completa de un spell por nombre (JSON string) desde dm/spells.json.
    Si no hay match exacto en el índice, intenta fallback por substring.
    """
    sp = _get_spell_def_by_name(name)
    if not sp:
        # fallback: intenta encontrar el primer match por substring
        comp = get_spells_compendium() or {}
        spells = (comp.get("spells") or {})
        q = _norm(name or "")
        best = None
        for _id, cand in spells.items():
            if not isinstance(cand, dict):
                continue
            nm = cand.get("name", "")
            if nm and q and q in _norm(nm):
                best = cand
                break
        if not best:
            return f"Error: spell '{name}' no encontrado en dm/spells.json"
        sp = best

    return json.dumps(sp, ensure_ascii=False, indent=2)


def tool_spell_search(query: str = "", limit: int = 10) -> str:
    """
    Busca spells por substring en nombre (fuzzy simple) y devuelve lista de nombres.
    """
    comp = get_spells_compendium() or {}
    spells = (comp.get("spells") or {})

    q = _norm(query or "")
    try:
        limit = int(limit or 10)
    except Exception:
        limit = 10
    limit = max(1, min(limit, 50))

    hits: List[str] = []
    for _id, sp in spells.items():
        if not isinstance(sp, dict):
            continue
        nm = sp.get("name", "")
        if not nm:
            continue
        if not q or q in _norm(nm):
            hits.append(nm)

    hits = sorted(set(hits))[:limit]
    return json.dumps({"query": query, "count": len(hits), "results": hits}, ensure_ascii=False, indent=2)

# =========================================================
# Distancias (bandas)
# =========================================================
def _band_rank(b: str) -> int:
    return {"melee": 0, "short": 1, "medium": 2, "long": 3, "extreme": 4}.get((b or "medium").lower(), 2)

def _get_positions(canon: dict) -> dict:
    return _ensure_dict(canon, "positions")

def tool_set_range(target: str, band: str) -> str:
    band = (band or "").strip().lower()
    if band not in {"melee", "short", "medium", "long", "extreme"}:
        return "band debe ser: melee | short | medium | long | extreme"

    canon = canon_load()
    found = _get_target_container(canon, target)
    cname = found[2] if found else target  # nombre canónico si existe

    pos = _get_positions(canon)
    pos[cname] = band
    canon_save(canon)
    return f"OK. {cname} ahora está en rango '{band}'."

def _canonical_name(canon: dict, name: str) -> str:
    found = _get_target_container(canon, name)
    return found[2] if found else name

def tool_get_range(a: str, b: str) -> str:
    canon = canon_load()
    pos = canon.get("positions", {}) or {}

    ca = _canonical_name(canon, a)
    cb = _canonical_name(canon, b)

    ra = pos.get(ca, "medium")
    rb = pos.get(cb, "medium")

    if ra == "melee" and rb == "melee":
        return "melee"
    rank = max(_band_rank(ra), _band_rank(rb))
    inv = {0: "melee", 1: "short", 2: "medium", 3: "long", 4: "extreme"}
    return inv.get(rank, "medium")

def tool_approach(actor: str) -> str:
    canon = canon_load()
    pos = _get_positions(canon)
    cur = pos.get(actor, "medium")
    pos[actor] = "melee"
    canon_save(canon)
    return f"{actor} se acerca: {cur} → melee"

def tool_retreat(actor: str) -> str:
    canon = canon_load()
    pos = _get_positions(canon)
    cur = pos.get(actor, "melee")
    pos[actor] = "short" if cur == "melee" else "medium"
    canon_save(canon)
    return f"{actor} se aleja: {cur} → {pos[actor]}"


# =========================================================
# Tools: escena
# =========================================================
def tool_start_scene(location: str, hook: str) -> str:
    # 1) Persistencia por sesión (AgentState)
    st = _get_state()
    if st is not None:
        st.scene["location"] = location
        st.scene["current_scene"] = hook
        # no machacamos recap/open_threads si ya existen

    # 2) Compatibilidad hacia atrás (canon.session global)
    canon = canon_load()
    session = _ensure_dict(canon, "session")
    session["location"] = location
    session["current_scene"] = hook
    canon_save(canon)

    return f"Escena iniciada. Lugar: {location}. Gancho: {hook}"


def tool_scene_status() -> str:
    st = _get_state()

    # Preferimos estado por sesión si existe
    if st is not None and isinstance(st.scene, dict):
        out = {
            "location": st.scene.get("location", ""),
            "current_scene": st.scene.get("current_scene", ""),
            "recap": st.scene.get("recap", ""),
            "open_threads": st.scene.get("open_threads", []),
            "flags": st.flags,
            "module_progress": st.module_progress,
        }
        return json.dumps(out, ensure_ascii=False, indent=2)

    # Fallback a canon.session (compat)
    canon = canon_load()
    session = canon.get("session", {}) or {}
    out = {
        "location": session.get("location", ""),
        "current_scene": session.get("current_scene", ""),
        "recap": session.get("recap", ""),
        "open_threads": session.get("open_threads", []),
    }
    return json.dumps(out, ensure_ascii=False, indent=2)


# =========================================================
# Tools: dados / checks
# =========================================================
def tool_roll(expr: str) -> str:
    expr = (expr or "").strip().lower().replace(" ", "")
    m = re.fullmatch(r"(\d*)d(\d+)([+-]\d+)?", expr)
    if not m:
        return "Formato inválido. Usa: 2d6+3, 1d20-1, d8, etc."
    n_str, sides_str, mod_str = m.groups()
    n = int(n_str) if n_str else 1
    sides = int(sides_str)
    mod = int(mod_str) if mod_str else 0
    if n < 1 or n > 200:
        return "Número de dados fuera de rango (1–200)."
    if sides < 2 or sides > 1000:
        return "Caras fuera de rango (2–1000)."
    rolls = [random.randint(1, sides) for _ in range(n)]
    total = sum(rolls) + mod
    return f"{expr} -> tiradas={rolls}, modificador={mod}, total={total}"

def tool_check(skill: str, dc: int, bonus: int = 0, mode: str = "normal") -> str:
    mode = (mode or "normal").lower()
    if mode not in {"normal", "adv", "dis"}:
        return "Error: mode debe ser 'normal', 'adv' o 'dis'."
    if not isinstance(dc, int) or dc < 1 or dc > 40:
        return "Error: dc fuera de rango (1–40)."
    r1 = random.randint(1, 20)
    r2 = random.randint(1, 20)
    if mode == "normal":
        chosen = r1
        rolls = [r1]
    elif mode == "adv":
        chosen = max(r1, r2)
        rolls = [r1, r2]
    else:
        chosen = min(r1, r2)
        rolls = [r1, r2]
    total = chosen + int(bonus)
    success = total >= dc
    return (
        f"CHECK {skill} | DC {dc} | bonus {bonus} | mode {mode}\n"
        f"tiradas={rolls} -> elegido={chosen} -> total={total}\n"
        f"resultado={'ÉXITO' if success else 'FALLO'}"
    )

# ----------------------------
# Skill check automático (abilities + prof_bonus + skill_proficiencies)
# ----------------------------
SKILL_TO_ABILITY = {
    # STR
    "athletics": "str",
    # DEX
    "acrobatics": "dex",
    "sleight of hand": "dex",
    "sleight_of_hand": "dex",
    "stealth": "dex",
    # INT
    "arcana": "int",
    "history": "int",
    "investigation": "int",
    "nature": "int",
    "religion": "int",
    # WIS
    "animal handling": "wis",
    "animal_handling": "wis",
    "insight": "wis",
    "medicine": "wis",
    "perception": "wis",
    "survival": "wis",
    # CHA
    "deception": "cha",
    "intimidation": "cha",
    "performance": "cha",
    "persuasion": "cha",
}

def _ability_mod(score: int) -> int:
    try:
        s = int(score)
    except Exception:
        s = 10
    return (s - 10) // 2

def tool_skill_check(actor: str, skill: str, dc: int, mode: str = "normal", extra_bonus: int = 0) -> str:
    """
    Skill check automático para PJ:
      bonus = mod(ability asociada) + (prof_bonus si competente) + extra_bonus
    Devuelve SIEMPRE desglose + la tirada (vía tool_check).
    """
    canon = canon_load()
    m = _find_party_member(canon, actor)
    if not m:
        return f"Error: '{actor}' no es un PJ (party member)."

    # Normaliza skill
    sk_raw = (skill or "").strip().lower().replace("-", " ")
    sk_raw = " ".join(sk_raw.split())
    sk_key = sk_raw.replace(" ", "_")
    sk_space = sk_raw

    ability_key = SKILL_TO_ABILITY.get(sk_raw) or SKILL_TO_ABILITY.get(sk_key) or SKILL_TO_ABILITY.get(sk_space)
    if not ability_key:
        return (
            f"Error: habilidad '{skill}' no reconocida.\n"
            f"Usa check(skill, dc, bonus, mode) o añade el mapeo en SKILL_TO_ABILITY."
        )

    abilities = m.get("abilities", {}) if isinstance(m.get("abilities", {}), dict) else {}
    score = abilities.get(ability_key, 10)
    mod = _ability_mod(score)

    pb = int(m.get("prof_bonus", 0) or 0)

    profs = m.get("skill_proficiencies", [])
    if not isinstance(profs, list):
        profs = []
    profs_norm = set(" ".join(str(x).strip().lower().replace("-", " ").split()) for x in profs)

    proficient = (sk_raw in profs_norm) or (sk_space in profs_norm) or (sk_key in profs_norm)
    prof_part = pb if proficient else 0
    bonus = mod + prof_part + int(extra_bonus)

    breakdown = (
        f"SKILL_CHECK {m.get('name')} -> {skill} ({ability_key.upper()})\n"
        f"score {ability_key}={score} mod={mod} | prof_bonus={pb} | proficient={proficient} (+{prof_part}) | extra={int(extra_bonus)}\n"
        f"TOTAL BONUS = {bonus}\n"
    )

    # Tirada con tu tool_check para que incluya las tiradas
    label = f"{skill} ({ability_key.upper()}){' [PROF]' if proficient else ''}"
    roll_out = tool_check(skill=label, dc=int(dc), bonus=int(bonus), mode=mode)

    return breakdown + roll_out

# =========================================================
# Tools: condiciones / HP
# =========================================================
def tool_apply_condition(target: str, condition: str, rounds: int = 1) -> str:
    canon = canon_load()
    found = _get_target_container(canon, target)
    # Auto-stub si no se encuentra (evita bloqueos)
    if not found:
        if _ensure_enemy_stub_anywhere(canon, target):
            canon = canon_load()
            found = _get_target_container(canon, target)
    if not found:
        return f"No encuentro el objetivo '{target}'."
    _, obj, cname = found
    conds = _ensure_conditions(obj)

    name = (condition or "").strip().upper()
    rounds = int(rounds)
    if not name:
        return "condition no puede estar vacío."
    if rounds < 1 or rounds > 99:
        return "rounds fuera de rango (1–99)."

    for c in conds:
        if str(c.get("name", "")).upper() == name:
            c["remaining"] = max(int(c.get("remaining", 0)), rounds)
            canon_save(canon)
            return f"OK. {cname} mantiene {name} ({c['remaining']} rondas)."

    conds.append({"name": name, "remaining": rounds})
    canon_save(canon)
    return f"OK. {cname} gana {name} ({rounds} rondas)."

def tool_remove_condition(target: str, condition: str) -> str:
    canon = canon_load()
    found = _get_target_container(canon, target)
    if not found:
        return f"No encuentro el objetivo '{target}'."
    _, obj, cname = found
    name = (condition or "").strip().upper()
    if not name:
        return "condition no puede estar vacío."
    before = len(obj.get("conditions", []) or [])
    obj["conditions"] = [c for c in (obj.get("conditions", []) or []) if str(c.get("name", "")).upper() != name]
    canon_save(canon)
    removed = before - len(obj["conditions"])
    return f"OK. Quitado {name} de {cname}." if removed else f"{cname} no tenía {name}."

def tool_register_enemies(names_json: str) -> str:
    """
    Registra enemigos en canon['enemies'] sin necesidad de iniciar combate.

    Entrada (JSON string) aceptada:
    - ["Cultist A", "Cultist B"]
    - [{"name":"Cultist A","ac":12,"hp":27}, {"name":"Encapuchado Oscuro","ac":14}]
    - {"enemies":[...]}  (lista como arriba)

    NUEVO:
    - Si el nombre (o template) coincide con bestiary, importa AC/HP y marca stub=False por defecto.

    Devuelve un resumen de creados/actualizados/ignorados.
    """
    canon = canon_load()

    try:
        data = json.loads(names_json) if isinstance(names_json, str) else names_json
    except Exception as e:
        return f"Error: names_json no es JSON válido: {e}"

    # Normalizar a lista
    if isinstance(data, dict) and "enemies" in data:
        items = data.get("enemies", [])
    else:
        items = data

    if not isinstance(items, list):
        return "Error: names_json debe ser una lista o un objeto {enemies:[...] }."

    canon.setdefault("enemies", {})
    enemies = canon["enemies"]

    created, updated, ignored = [], [], []

    def _as_int(v, default):
        try:
            if v is None:
                return default
            if isinstance(v, bool):
                return int(v)
            return int(v)
        except Exception:
            return default

    def _as_float(v, default):
        try:
            if v is None:
                return default
            return float(v)
        except Exception:
            return default

    def _as_list(v):
        if v is None:
            return []
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            s = v.strip()
            return [s] if s else []
        return [v]

    def _as_dict(v):
        return v if isinstance(v, dict) else {}

    for it in items:
        if isinstance(it, str):
            name = it.strip()
            payload = {}
        elif isinstance(it, dict):
            name = str(it.get("name", "")).strip()
            payload = it
        else:
            ignored.append(str(it))
            continue

        if not name:
            ignored.append(str(it))
            continue

        # No crear enemigo si es PJ/party
        if _find_party_member(canon, name):
            ignored.append(f"{name} (es party)")
            continue

        template = str(payload.get("template", "") or "").strip()

        # Defaults (si hay bestiary, mejor)
        mon_def = _get_monster_def_fuzzy(template) if template else None
        if not mon_def:
            mon_def = _get_monster_def_fuzzy(name)

        if mon_def:
            base_entry = _import_monster_as_enemy(mon_def, name)
            default_ac = int(base_entry.get("ac", 10))
            default_max_hp = int(base_entry.get("max_hp", 10))
            default_stub = False
            default_source = "bestiary"
        else:
            default_ac = 10
            default_max_hp = 10
            default_stub = True
            default_source = "generated"

        # Aplicar overrides del payload si vienen
        ac = _as_int(payload.get("ac"), default_ac)
        max_hp = _as_int(payload.get("hp"), default_max_hp)
        max_hp = _as_int(payload.get("hp_max"), max_hp)
        max_hp = _as_int(payload.get("max_hp"), max_hp)
        hp_current = _as_int(payload.get("hp_current"), max_hp)

        stub = bool(payload.get("stub", default_stub))
        source_s = str(payload.get("source", default_source)).strip() if payload.get("source", default_source) is not None else default_source

        # Campos “ricos” opcionales
        role = payload.get("role")
        level = payload.get("level")
        cr = payload.get("cr")
        abilities = payload.get("abilities")
        skills = payload.get("skills")
        saves = payload.get("saves")
        attacks = payload.get("attacks")
        spells = payload.get("spells")
        features = payload.get("features")
        tags = payload.get("tags")
        notes = payload.get("notes")

        # Normalizar tipos
        level_i = _as_int(level, None) if level is not None else None
        cr_f = _as_float(cr, None) if cr is not None else None
        abilities_d = _as_dict(abilities)
        skills_d = skills if isinstance(skills, dict) else (_as_list(skills) if skills is not None else None)
        saves_d = saves if isinstance(saves, dict) else (_as_list(saves) if saves is not None else None)
        attacks_l = _as_list(attacks) if attacks is not None else None
        spells_l = _as_list(spells) if spells is not None else None
        features_l = _as_list(features) if features is not None else None
        tags_l = _as_list(tags) if tags is not None else None
        notes_s = str(notes).strip() if notes is not None else None
        role_s = str(role).strip() if role is not None else None

        if name not in enemies:
            # base from bestiary if exists (pero dejamos que payload sobrescriba abajo)
            entry = {}
            if mon_def:
                entry.update(_import_monster_as_enemy(mon_def, name))

            entry.update({
                "name": name,
                "stub": stub,
                "ac": ac,
                "max_hp": max(1, int(max_hp)),
                "hp_current": max(0, int(hp_current)),
                "hp": max(0, int(hp_current)),          # compat
                "hp_max": max(1, int(max_hp)),          # compat
                "conditions": [],
                "source": source_s,
            })

            # Añadir “ricos” si vienen
            if role_s is not None:
                entry["role"] = role_s
            if level_i is not None:
                entry["level"] = level_i
            if cr_f is not None:
                entry["cr"] = cr_f
            if abilities_d:
                entry["abilities"] = abilities_d
            if skills_d is not None:
                entry["skills"] = skills_d
            if saves_d is not None:
                entry["saves"] = saves_d
            if attacks_l is not None:
                entry["attacks"] = attacks_l
            if spells_l is not None:
                entry["spells"] = spells_l
            if features_l is not None:
                entry["features"] = features_l
            if tags_l is not None:
                entry["tags"] = tags_l
            if notes_s is not None:
                entry["notes"] = notes_s

            enemies[name] = entry
            created.append(name)

        else:
            eobj = enemies[name]
            eobj.setdefault("name", name)
            eobj.setdefault("conditions", [])

            # sobrescribir solo si viene en payload (o si es stub y hay bestiary)
            if mon_def and eobj.get("stub") and not payload.get("stub", None):
                # “enriquece” un stub con bestiary si no te lo han bloqueado explícitamente
                eobj.update(_import_monster_as_enemy(mon_def, name))

            if "ac" in payload:
                eobj["ac"] = ac
            if any(k in payload for k in ["hp", "hp_max", "max_hp", "hp_current"]):
                eobj["max_hp"] = max(1, int(max_hp))
                eobj["hp_current"] = max(0, int(hp_current))
                eobj["hp"] = eobj["hp_current"]
                eobj["hp_max"] = eobj["max_hp"]

            if "stub" in payload:
                eobj["stub"] = stub
            if "source" in payload:
                eobj["source"] = source_s

            if role_s is not None and "role" in payload:
                eobj["role"] = role_s
            if level is not None and "level" in payload:
                eobj["level"] = level_i
            if cr is not None and "cr" in payload:
                eobj["cr"] = cr_f
            if abilities is not None and "abilities" in payload:
                eobj["abilities"] = abilities_d
            if skills is not None and "skills" in payload:
                eobj["skills"] = skills_d
            if saves is not None and "saves" in payload:
                eobj["saves"] = saves_d
            if attacks is not None and "attacks" in payload:
                eobj["attacks"] = attacks_l
            if spells is not None and "spells" in payload:
                eobj["spells"] = spells_l
            if features is not None and "features" in payload:
                eobj["features"] = features_l
            if tags is not None and "tags" in payload:
                eobj["tags"] = tags_l
            if notes is not None and "notes" in payload:
                eobj["notes"] = notes_s

            updated.append(name)

    canon_save(canon)

    return (
        "ENEMIES REGISTERED\n"
        f"created: {created}\n"
        f"updated: {updated}\n"
        f"ignored: {ignored}\n"
        f"total_enemies: {len(enemies)}"
    )

def tool_spawn_encounter(encounter_json: str) -> str:
    """
    Crea enemigos con stats jugables.
    - Si encuentra coincidencia en bestiary.json (compendio: indexes.by_name + monsters{}), importa stats.
    - Si no, genera stats por role + cr/level.

    Entrada JSON (string):
    {
      "enemies": [
        {"name":"Cultist A","template":"cultist","cr":0.25},
        {"name":"Cultist B","template":"cultist","cr":0.25},
        {"name":"Encapuchado Oscuro","role":"caster","cr":3, "spells":["Hold Person","Magic Missile"]}
      ]
    }
    """
    try:
        data = json.loads(encounter_json) if isinstance(encounter_json, str) else encounter_json
    except Exception as e:
        return f"Error: encounter_json no es JSON válido: {e}"

    items = data.get("enemies", data if isinstance(data, list) else None)
    if not isinstance(items, list):
        return "Error: encounter_json debe ser lista o {enemies:[...] }"

    canon = canon_load()
    canon.setdefault("enemies", {})
    enemies = canon["enemies"]

    def _as_int(v, default):
        try:
            if v is None:
                return default
            return int(v)
        except Exception:
            return default

    def _hp_for(cr: float, role: str) -> int:
        base = {
            0: 5, 0.125: 9, 0.25: 13, 0.5: 22, 1: 33, 2: 52, 3: 72, 4: 88, 5: 110,
            6: 130, 7: 150, 8: 175, 9: 200, 10: 225
        }
        keys = sorted(base.keys())
        nearest = min(keys, key=lambda k: abs(k - cr))
        hp = base[nearest]
        role = (role or "").lower()
        if role in {"minion"}:
            hp = max(1, int(hp * 0.5))
        elif role in {"brute"}:
            hp = int(hp * 1.35)
        elif role in {"boss"}:
            hp = int(hp * 1.8)
        return int(max(1, hp))

    def _ac_for(cr: float, role: str) -> int:
        ac = 11 + int(cr // 2)
        role = (role or "").lower()
        if role in {"skirmisher"}:
            ac += 1
        if role in {"boss"}:
            ac += 1
        return int(min(20, max(10, ac)))

    def _atk_bonus_for(cr: float) -> int:
        return int(3 + int(max(0, cr - 1) // 2) * 2)

    def _save_dc_for(cr: float) -> int:
        return int(11 + int(max(0, cr) // 2))

    created, updated, ignored = [], [], []

    for it in items:
        if isinstance(it, str):
            name = it.strip()
            payload = {}
        elif isinstance(it, dict):
            name = str(it.get("name", "")).strip()
            payload = it
        else:
            ignored.append(str(it))
            continue

        if not name:
            ignored.append(str(it))
            continue

        # no crear si es party
        if _find_party_member(canon, name):
            ignored.append(f"{name} (es party)")
            continue

        template = str(payload.get("template", "") or "").strip()
        role = str(payload.get("role", "") or "").strip().lower() or "skirmisher"

        # 1) bestiary match (template primero, luego name)
        mon_def = _get_monster_def_fuzzy(template) if template else None
        if not mon_def:
            mon_def = _get_monster_def_fuzzy(name)

        if mon_def:
            entry = _import_monster_as_enemy(mon_def, name)
        else:
            # 2) generate
            cr = payload.get("cr")
            level = payload.get("level")
            try:
                cr_val = float(cr) if cr is not None else max(0.25, float(level or 1) / 2.0)
            except Exception:
                cr_val = 0.25

            ac = _as_int(payload.get("ac"), _ac_for(cr_val, role))
            max_hp = _as_int(payload.get("hp"), _hp_for(cr_val, role))
            max_hp = _as_int(payload.get("hp_max"), max_hp)

            entry = {
                "name": name,
                "stub": False,
                "role": role,
                "cr": cr_val,
                "level": level,
                "ac": ac,

                # runtime HP unificado
                "max_hp": int(max_hp),
                "hp_current": int(max_hp),

                # compat keys
                "hp": int(max_hp),
                "hp_max": int(max_hp),

                "conditions": [],
                "atk_bonus": _atk_bonus_for(cr_val),
                "save_dc": _save_dc_for(cr_val),
                "spells": payload.get("spells", []) or [],
                "features": payload.get("features", []) or [],
                "source": "generated",
                "speed": _speed_to_int(payload.get("speed", 30), default_speed=30),
                "move_left": 0,
            }

        # permitir overrides explícitos del payload (sin romper runtime hp)
        if isinstance(payload, dict):
            if "ac" in payload:
                entry["ac"] = _as_int(payload.get("ac"), entry.get("ac", 10))
            # si te pasan hp_current/max_hp explícitos
            if "max_hp" in payload or "hp_max" in payload or "hp" in payload or "hp_current" in payload:
                max_hp = _as_int(payload.get("max_hp"), _as_int(payload.get("hp_max"), _as_int(payload.get("hp"), entry.get("max_hp", 10))))
                hp_cur = _as_int(payload.get("hp_current"), max_hp)
                entry["max_hp"] = max(1, int(max_hp))
                entry["hp_current"] = max(0, int(hp_cur))
                entry["hp"] = entry["hp_current"]
                entry["hp_max"] = entry["max_hp"]

        if name in enemies:
            enemies[name].update(entry)
            updated.append(name)
        else:
            enemies[name] = entry
            created.append(name)

    canon_save(canon)
    return (
        "SPAWN_ENCOUNTER OK\n"
        f"created={created}\n"
        f"updated={updated}\n"
        f"ignored={ignored}\n"
        f"total_enemies={len(enemies)}"
    )

def tool_target_status(target: str) -> str:
    canon = canon_load()
    found = _get_target_container(canon, target)
    
    if not found:
        if _ensure_enemy_stub_anywhere(canon, target):
            canon = canon_load()
            found = _get_target_container(canon, target)

    if not found:
        return f"No encuentro el objetivo '{target}'."
    kind, obj, cname = found
    conds = obj.get("conditions", []) or []

    if kind == "party":
        hp = obj.get("hp", 0)
        max_hp = obj.get("max_hp", hp)
        ac = obj.get("ac", 10)
        stable = obj.get("stable", True)
        header = f"{cname} (PJ) HP {hp}/{max_hp} CA {ac} estable={stable}"
    else:
        if "hp_current" in obj and "max_hp" in obj:
            header = f"{cname} (ENEMIGO) HP {obj.get('hp_current','?')}/{obj.get('max_hp','?')} CA {obj.get('ac','?')}"
        else:
            header = f"{cname} (ENEMIGO) HP {obj.get('hp','?')} CA {obj.get('ac','?')}"

    if not conds:
        return header + "\nCondiciones: (ninguna)"
    pretty = ", ".join([f"{c['name']}({c.get('remaining','?')})" for c in conds])
    return header + f"\nCondiciones: {pretty}"

def tool_conditions_status() -> str:
    canon = canon_load()
    lines = []
    for m in canon.get("party", {}).get("members", []):
        conds = m.get("conditions", []) or []
        if conds:
            pretty = ", ".join([f"{c['name']}({c.get('remaining','?')})" for c in conds])
            lines.append(f"PJ {m.get('name')}: {pretty}")
    for name, e in (canon.get("enemies", {}) or {}).items():
        conds = e.get("conditions", []) or []
        if conds:
            pretty = ", ".join([f"{c['name']}({c.get('remaining','?')})" for c in conds])
            lines.append(f"EN {name}: {pretty}")
    return "\n".join(lines) if lines else "No hay condiciones activas."

def tool_heal(target: str, amount: int) -> str:
    canon = canon_load()
    m = _find_party_member(canon, target)
    if not m:
        return f"Objetivo '{target}' no es un PJ."
    amount = int(amount)
    if amount <= 0:
        return "amount debe ser > 0"
    hp = int(m.get("hp", 0))
    max_hp = int(m.get("max_hp", hp))
    new_hp = min(max_hp, hp + amount)
    m["hp"] = new_hp
    if new_hp > 0:
        m["stable"] = True
    canon_save(canon)
    return f"Curación: {m.get('name')} {hp} → {new_hp} (max {max_hp})"

def tool_damage(target: str, amount: int) -> str:
    canon = canon_load()
    amount = int(amount)
    if amount <= 0:
        return "amount debe ser > 0"

    # 1) Party
    m = _find_party_member(canon, target)
    if m:
        hp = int(m.get("hp", 0))
        new_hp = max(0, hp - amount)
        m["hp"] = new_hp
        if new_hp == 0:
            m["stable"] = False
        canon_save(canon)
        status = "INCONSCIENTE (inestable)" if new_hp == 0 else "OK"
        return f"Daño: {m.get('name')} {hp} → {new_hp} ({status})"

    # 2) Enemy
    fe = _find_enemy(canon, target)
    if not fe:
        # intenta resolver por contenedor (strip comillas, fuzzy, bestiary autospawn, etc.)
        found = _get_target_container(canon, target)
        if found and found[0] == "enemy":
            enemy_name = found[2]
            enemy_obj = found[1]
        else:
            return f"Objetivo '{target}' no existe como PJ ni como enemigo."
    else:
        enemy_name, enemy_obj = fe

    # Runtime HP unificado: hp_current/max_hp si existen, si no hp/hp_max
    if "hp_current" in enemy_obj:
        hp = int(enemy_obj.get("hp_current", 0) or 0)
        new_hp = max(0, hp - amount)
        enemy_obj["hp_current"] = new_hp
        max_hp = int(enemy_obj.get("max_hp", hp) or hp)
    else:
        hp = int(enemy_obj.get("hp", 0) or 0)
        new_hp = max(0, hp - amount)
        enemy_obj["hp"] = new_hp
        max_hp = int(enemy_obj.get("hp_max", hp) or hp)

    killed = (new_hp <= 0)

    # si muere en combate, quitar de orden
    combat = canon.get("combat", {}) or {}
    if killed and combat.get("active"):
        order = combat.get("order", []) or []
        combat["order"] = [o for o in order if _norm(o.get("name","")) != _norm(enemy_name)]
        canon["combat"] = combat

    canon_save(canon)
    return f"Daño: {enemy_name} {hp}/{max_hp} → {new_hp}/{max_hp}" + (" (DERROTADO)" if killed else "")

def tool_execute_actions(script: str, default_target: str = "") -> str:
    canon = canon_load()
    party = (canon.get("party") or {})
    members = party.get("members") or []
    enemies = canon.get("enemies", {}) or {}

    def _norm(s: str) -> str:
        return (s or "").strip().lower()

    def _first_alive_enemy() -> str:
        for name, obj in enemies.items():
            hp = obj.get("hp_current", obj.get("hp", 0))
            try:
                if int(hp) > 0:
                    return name
            except Exception:
                continue
        return ""

    def _find_member(name: str) -> dict:
        nn = _norm(name)
        # exact match
        for m in members:
            if _norm(m.get("name", "")) == nn:
                return m

        # fallback: match por primer token ("Kaelen" vs "Kaelen el Sombrío")
        nn_first = nn.split()[0] if nn.split() else nn
        for m in members:
            mm = _norm(m.get("name", ""))
            if mm.startswith(nn_first):
                return m

        # fuzzy match (typos tipo Kaedrel->Kaelen)
        names = [m.get("name", "") for m in members if m.get("name")]
        lowered = {n.lower(): n for n in names}
        candidates = difflib.get_close_matches(nn, list(lowered.keys()), n=1, cutoff=0.72)
        if candidates:
            best = lowered[candidates[0]]
            for m in members:
                if m.get("name") == best:
                    return m

        return {}

    def _ability_mod(score) -> int:
        try:
            s = int(score)
        except Exception:
            s = 10
        return (s - 10) // 2

    def _equipped(member: dict) -> list:
        inv = member.get("inventory") or []
        return [it for it in inv if isinstance(it, dict) and it.get("equipped")]

    def _get_prof(member: dict) -> int:
        try:
            return int(member.get("prof_bonus", 0) or 0)
        except Exception:
            return 0

    def _weapon_bonus(item: dict) -> int:
        try:
            return int(item.get("bonus", 0) or 0)
        except Exception:
            return 0

    def _is_focus_cd_attack(item: dict) -> bool:
        n = _norm(item.get("name", ""))
        return ("cd/attack" in n) or ("cd/ataque" in n) or ("spell attack" in n)

    # ---- NUEVO: leer Sneak Attack del canon (p.ej. "Sneak Attack (3d6)") ----
    def _rogue_level(member: dict) -> int:
        lvl = 0
        for c in (member.get("classes") or []):
            if _norm(c.get("name", "")) == "rogue":
                try:
                    lvl += int(c.get("level", 0) or 0)
                except Exception:
                    pass
        return int(lvl)
    
    def _sneak_dice_from_level(rogue_lvl: int) -> str:
        # tabla 5e
        table = {
            1: "1d6", 2: "1d6",
            3: "2d6", 4: "2d6",
            5: "3d6", 6: "3d6",
            7: "4d6", 8: "4d6",
            9: "5d6", 10: "5d6",
            11: "6d6", 12: "6d6",
            13: "7d6", 14: "7d6",
            15: "8d6", 16: "8d6",
            17: "9d6", 18: "9d6",
            19: "10d6", 20: "10d6",
        }
        return table.get(max(1, min(20, int(rogue_lvl))), "1d6")
    
    def _get_sneak_dice(member: dict) -> str:
        feats = member.get("features") or []
        for f in feats:
            s = str(f)
            m = re.search(r"Sneak\s*Attack\s*\((\d+d\d+)\)", s, re.IGNORECASE)
            if m:
                return m.group(1)
        # fallback razonable
        rl = _rogue_level(member)
        if rl > 0:
            return _sneak_dice_from_level(rl)
        return "1d6"

    # ---- NUEVO: aplicar daño directo a enemigo (porque no hay tool_damage_enemy) ----
    def _apply_enemy_damage(enemy_name: str, amount: int) -> str:
        c = canon_load()
        e = (c.get("enemies", {}) or {}).get(enemy_name)
        if not e:
            return f"(WARN) No existe enemigo '{enemy_name}' para aplicar Sneak Attack."

        amount = int(amount)
        if amount <= 0:
            return "(WARN) Sneak Attack amount inválido."

        if "hp_current" in e:
            hp0 = int(e.get("hp_current", 0) or 0)
            hp1 = max(0, hp0 - amount)
            e["hp_current"] = hp1
        else:
            hp0 = int(e.get("hp", 0) or 0)
            hp1 = max(0, hp0 - amount)
            e["hp"] = hp1

        killed = (hp1 <= 0)

        # XP si mata (igual que tool_attack)
        if killed:
            cr = e.get("cr")
            xp_val = _xp_for_cr(cr)
            if xp_val > 0:
                _add_xp_to_party(c, xp_val, reason=f"KILL:{enemy_name} (Sneak Attack)", meta={"cr": cr})

        canon_save(c)
        if killed:
            return f"SNEAK ATTACK: daño {amount}. HP {enemy_name}: {hp0} → {hp1}. (MUERTE)"
        return f"SNEAK ATTACK: daño {amount}. HP {enemy_name}: {hp0} → {hp1}."

    # ---- NUEVO: parse de arma forzada: Actor (xxx) ----
    def _parse_forced_actor(actor_token: str) -> tuple[str, str]:
        m = re.match(r"^(.+?)\s*\((.+?)\)\s*$", actor_token.strip())
        if not m:
            return actor_token.strip(), ""
        return m.group(1).strip(), _norm(m.group(2))

    def _pick_weapon(member: dict, attack_type: str, forced: str = "") -> dict:
        eq = _equipped(member)
        if not eq:
            return {}

        def nm(it): return _norm(it.get("name", ""))

        if forced:
            forced_map = {
                "arco": ["arco", "bow", "longbow", "shortbow"],
                "bow": ["arco", "bow", "longbow", "shortbow"],
                "estoque": ["estoque", "rapier"],
                "rapier": ["estoque", "rapier"],
                "alabarda": ["alabarda", "halberd"],
                "halberd": ["alabarda", "halberd"],
                "martillo": ["martillo", "warhammer"],
                "warhammer": ["martillo", "warhammer"],
                "bastón": ["bastón", "staff", "quarterstaff"],
                "staff": ["bastón", "staff", "quarterstaff"],
                "melee": ["alabarda", "halberd", "estoque", "rapier", "martillo", "warhammer", "bastón", "staff"],
                "ranged": ["arco", "bow", "ballesta", "crossbow"],
            }
            keys = forced_map.get(forced, [forced])
            for it in eq:
                n = nm(it)
                if any(k in n for k in keys):
                    return it

        if attack_type == "ranged":
            for it in eq:
                n = nm(it)
                if "arco" in n or "bow" in n or "ballesta" in n or "crossbow" in n:
                    return it
            return eq[0]

        for it in eq:
            n = nm(it)
            if "alabarda" in n or "halberd" in n or "glaive" in n or "guja" in n:
                return it
        for it in eq:
            n = nm(it)
            if "estoque" in n or "rapier" in n:
                return it
        for it in eq:
            n = nm(it)
            if "martillo" in n or "warhammer" in n or "maza" in n or "mace" in n:
                return it
        for it in eq:
            n = nm(it)
            if "bastón" in n or "staff" in n or "quarterstaff" in n:
                return it

        return eq[0]

    def _weapon_damage_die(item: dict, attack_type: str) -> tuple[str, bool, bool]:
        name = _norm(item.get("name", ""))

        if "arco largo" in name or "longbow" in name:
            return ("1d8", False, False)
        if "arco corto" in name or "shortbow" in name:
            return ("1d6", False, False)
        if "ballesta" in name or "crossbow" in name:
            return ("1d8", False, False)

        if "alabarda" in name or "halberd" in name:
            return ("1d10", False, True)
        if "guja" in name or "glaive" in name:
            return ("1d10", False, True)
        if "estoque" in name or "rapier" in name:
            return ("1d8", True, False)
        if "espada corta" in name or "shortsword" in name:
            return ("1d6", True, False)
        if "martillo de guerra" in name or "warhammer" in name:
            return ("1d8", False, False)
        if "bastón" in name or "staff" in name or "quarterstaff" in name:
            return ("1d6", False, False)

        return ("1d8", False, False)

    def _attack_stat_mod(member: dict, finesse: bool, attack_type: str) -> int:
        ab = member.get("abilities", {}) or {}
        str_mod = _ability_mod(ab.get("str", 10))
        dex_mod = _ability_mod(ab.get("dex", 10))
        if attack_type == "ranged":
            return dex_mod
        return max(dex_mod, str_mod) if finesse else str_mod

    def _rogue_level(member: dict) -> int:
        lvl = 0
        for c in (member.get("classes") or []):
            if _norm(c.get("name", "")) == "rogue":
                try:
                    lvl += int(c.get("level", 0) or 0)
                except Exception:
                    pass
        return int(lvl)
    
    def _spell_attack_bonus(member: dict) -> int:
        classes = member.get("classes") or []
        main = ""
        main_lvl = -1
        for c in classes:
            try:
                lvl = int(c.get("level", 0) or 0)
            except Exception:
                lvl = 0
            if lvl > main_lvl:
                main_lvl = lvl
                main = _norm(c.get("name", ""))
    
        ab = member.get("abilities", {}) or {}
        if main in {"wizard", "artificer"}:
            stat_mod = _ability_mod(ab.get("int", 10))
        elif main in {"cleric", "druid", "ranger"}:
            stat_mod = _ability_mod(ab.get("wis", 10))
        elif main in {"paladin", "sorcerer", "warlock", "bard"}:
            stat_mod = _ability_mod(ab.get("cha", 10))
        else:
            stat_mod = _ability_mod(ab.get("int", 10))
    
        focus_bonus = 0
        for it in _equipped(member):
            if _is_focus_cd_attack(it):
                focus_bonus = max(focus_bonus, _weapon_bonus(it))
    
        return _get_prof(member) + stat_mod + focus_bonus

    def _fire_bolt_damage(member: dict) -> str:
        try:
            lvl = int(member.get("total_level", 1) or 1)
        except Exception:
            lvl = 1
        if lvl >= 17:
            return "4d10"
        if lvl >= 11:
            return "3d10"
        if lvl >= 5:
            return "2d10"
        return "1d10"

    tgt = default_target.strip() or _first_alive_enemy()
    if not tgt:
        return "No hay enemigos vivos registrados para ejecutar acciones."

        # -------- PRE-FLIGHT: asegurar enemigos mencionados en el script --------
    def _extract_enemy_names(text: str) -> list[str]:
        """
        Detecta patrones tipo "Bandido A", "Cultista B", "Encapuchado Oscuro", etc.
        OJO: es heurístico; por eso filtramos contra party y enemigos ya registrados.
        """
        hits = set()

        # 1) cosas entre comillas "..."
        for q in re.findall(r'"([^"]+)"', text):
            q = q.strip()
            if len(q) >= 3:
                hits.add(q)

        # 2) patrón tipo "Bandido A" / "Cultista B"
        for m in re.findall(r'\b([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)\s+([A-Z])\b', text):
            hits.add(f"{m[0]} {m[1]}")

        # 3) patrón tipo "Encapuchado Oscuro" (dos palabras capitalizadas)
        for m in re.findall(r'\b([A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑáéíóúñ]+)\s+([A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑáéíóúñ]+)\b', text):
            cand = f"{m[0]} {m[1]}".strip()
            # filtra falsos positivos típicos
            if cand.lower() not in {"gran arma", "arma grande", "great weapon", "sharp shooter"}:
                hits.add(cand)

        return sorted(hits)

    mentioned = _extract_enemy_names(script)
    to_spawn = [n for n in mentioned if n not in enemies and not _find_member(n)]
    if to_spawn:
        # En vez de stubs, intentamos spawn real (bestiary si hay match; si no, generado)
        spawn_items = []
        for n in to_spawn:
            # si hay bestiary match directo, spawn con name basta
            mon_def = _get_monster_def_fuzzy(n)
            if mon_def:
                spawn_items.append({"name": n})
            else:
                # fallback: generado “ligero”
                spawn_items.append({"name": n, "role": "skirmisher", "cr": 0.25})

        tool_spawn_encounter(json.dumps({"enemies": spawn_items}, ensure_ascii=False))

        # recarga canon/enemies para que ya existan en esta ejecución
        canon = canon_load()
        enemies = canon.get("enemies", {}) or {}
    # -----------------------------------------------------------------------


    parts = [p.strip() for p in re.split(r"[.\n]+", script) if p.strip()]
    out_lines = ["EXECUTE_ACTIONS"]

    # ---- NUEVO: SA se aplica una vez por actor por llamada a tool_execute_actions ----
    sa_used_by_actor = {}

    for p in parts:
        m = re.match(r"^([A-Za-zÁÉÍÓÚÑáéíóúñ0-9_\-\(\)\s]+)\s+(.*)$", p)
        if not m:
            continue

        actor_token = m.group(1).strip()
        rest_raw = m.group(2).strip()
        rest = rest_raw.lower()

        actor, forced_weapon = _parse_forced_actor(actor_token)
        member = _find_member(actor)

        n = 1
        mn = re.search(r"(\d+)\s+(disparos|ataques|ataque)", rest)
        if mn:
            n = max(1, int(mn.group(1)))

        use_ss = ("sharpshooter" in rest) or ("sharp shooter" in rest)
        use_gwm = ("gwm" in rest) or ("great weapon master" in rest) or ("great weapon mastery" in rest)
        use_pam = ("pam" in rest) or ("polearm mastery" in rest) or ("maestro de armas de asta" in rest)

        # ---- NUEVO: detecta Sneak Attack en la línea ----
        wants_sa = ("sneak attack" in rest) or ("ataque furtivo" in rest)

        attack_type = "ranged" if ("disparo" in rest or "ranged" in rest or use_ss) else "melee"
        if "mele" in rest or "melee" in rest:
            attack_type = "melee"

        if "fire bolt" in rest:
            if not member:
                out_lines.append(f"\n[{actor}] Fire Bolt → {tgt}")
                out_lines.append(tool_attack(attacker=actor, target=tgt, bonus=0, damage="1d10", attack_type="ranged"))
                continue

            atk_bonus = _spell_attack_bonus(member)
            dmg = _fire_bolt_damage(member)
            out_lines.append(f"\n[{actor}] Fire Bolt → {tgt} (atk+{atk_bonus}, dmg {dmg})")
            out_lines.append(
                tool_attack(
                    attacker=actor,
                    target=tgt,
                    bonus=atk_bonus,
                    damage=dmg,
                    attack_type="ranged",
                    auto_power=False
                )
            )
            continue

        # ataques con arma
        if not member:
            out_lines.append(f"\n[{actor}] x{n} {attack_type} → {tgt}")
            for _ in range(n):
                out_lines.append(
                    tool_attack(
                        attacker=actor,
                        target=tgt,
                        bonus=0,
                        damage="1d8",
                        attack_type=attack_type,
                        power_shot=use_ss,
                        power_attack=use_gwm,
                        auto_power=True
                    )
                )
            continue

        weapon = _pick_weapon(member, attack_type, forced=forced_weapon)
        die, finesse, heavy = _weapon_damage_die(weapon, attack_type)
        w_bonus = _weapon_bonus(weapon)

        stat_mod = _attack_stat_mod(member, finesse=finesse, attack_type=attack_type)
        prof = _get_prof(member)

        atk_bonus = prof + stat_mod + w_bonus
        dmg_mod = stat_mod + w_bonus
        dmg = f"{die}+{dmg_mod}" if dmg_mod != 0 else die

        out_lines.append(
            f"\n[{actor}] x{n} {attack_type} → {tgt} (atk+{atk_bonus}, dmg {dmg})"
            + (f" [weapon:{weapon.get('name','?')}]" if weapon else "")
            + (" (SS)" if use_ss else "")
            + (" (GWM)" if use_gwm else "")
            + (" (PAM)" if use_pam else "")
            + (" (SA)" if wants_sa else "")
        )

        # ataques principales
        for _ in range(n):
            res = tool_attack(
                attacker=actor,
                target=tgt,
                bonus=atk_bonus,
                damage=dmg,
                attack_type=attack_type,
                power_shot=use_ss,
                power_attack=use_gwm,
                auto_power=True
            )
            out_lines.append(res)

            # ---- NUEVO: aplica Sneak Attack al primer IMPACTO (una vez por actor) ----
            if wants_sa and not sa_used_by_actor.get(actor, False):
                # hit si contiene "Impacto" y no contiene "Resultado: FALLO"
                if ("Impacto" in res) and ("Resultado: FALLO" not in res):
                    sa_dice = _get_sneak_dice(member)  # e.g. "3d6"
                    roll_txt = tool_roll(sa_dice)
                    mtotal = re.search(r"total\s*=\s*(\d+)", roll_txt)
                    sa_amt = int(mtotal.group(1)) if mtotal else 0

                    out_lines.append(f"[{actor}] Sneak Attack extra roll: {roll_txt}")
                    out_lines.append(_apply_enemy_damage(tgt, sa_amt))

                    sa_used_by_actor[actor] = True

        # PAM bonus attack 1d4
        if use_pam and attack_type == "melee":
            wname = _norm(weapon.get("name", "")) if weapon else ""
            pam_ok = ("alabarda" in wname) or ("halberd" in wname) or ("guja" in wname) or ("glaive" in wname)
            if pam_ok:
                pam_dmg = f"1d4+{dmg_mod}" if dmg_mod != 0 else "1d4"
                out_lines.append(f"[{actor}] PAM bonus attack → {tgt} (atk+{atk_bonus}, dmg {pam_dmg})")
                out_lines.append(
                    tool_attack(
                        attacker=actor,
                        target=tgt,
                        bonus=atk_bonus,
                        damage=pam_dmg,
                        attack_type="melee",
                        auto_power=False
                    )
                )

    return "\n".join(out_lines)

def _equipped_items(member: dict) -> list:
    inv = member.get("inventory") or []
    return [it for it in inv if isinstance(it, dict) and it.get("equipped")]


def _weapon_bonus(item: dict) -> int:
    try:
        return int(item.get("bonus", 0) or 0)
    except Exception:
        return 0


def _weapon_profile_from_item(item: dict, attack_type: str) -> dict:
    """
    Devuelve: {"die": "1d8", "tags": set(...), "finesse": bool, "heavy": bool}
    """
    name = _norm(item.get("name", ""))
    tags = set()
    finesse = False
    heavy = False
    two_handed = False

    if attack_type == "ranged":
        tags.add("ranged")
    else:
        tags.add("melee")

    # Ranged
    if "arco largo" in name or "longbow" in name:
        two_handed = True
        return {"die": "1d8", "tags": tags | {"two_handed"}, "finesse": False, "heavy": False}
    if "arco corto" in name or "shortbow" in name:
        two_handed = True
        return {"die": "1d6", "tags": tags | {"two_handed"}, "finesse": False, "heavy": False}
    if "ballesta" in name or "crossbow" in name:
        two_handed = True
        return {"die": "1d8", "tags": tags | {"two_handed"}, "finesse": False, "heavy": False}

    # Melee (heavy/two-handed)
    if "alabarda" in name or "halberd" in name:
        heavy = True
        two_handed = True
        return {"die": "1d10", "tags": tags | {"heavy", "two_handed"}, "finesse": False, "heavy": True}
    if "guja" in name or "glaive" in name:
        heavy = True
        two_handed = True
        return {"die": "1d10", "tags": tags | {"heavy", "two_handed"}, "finesse": False, "heavy": True}

    # Finesse
    if "estoque" in name or "rapier" in name:
        finesse = True
        return {"die": "1d8", "tags": tags | {"finesse"}, "finesse": True, "heavy": False}
    if "espada corta" in name or "shortsword" in name:
        finesse = True
        return {"die": "1d6", "tags": tags | {"finesse"}, "finesse": True, "heavy": False}

    # Other melee
    if "martillo de guerra" in name or "warhammer" in name:
        return {"die": "1d8", "tags": tags, "finesse": False, "heavy": False}
    if "maza" in name or "mace" in name:
        return {"die": "1d6", "tags": tags, "finesse": False, "heavy": False}
    if "bastón" in name or "staff" in name or "quarterstaff" in name:
        return {"die": "1d6", "tags": tags, "finesse": False, "heavy": False}

    # Heavy melee típicas
    if "mandoble" in name or "greatsword" in name:
        heavy = True
        two_handed = True
        return {"die": "2d6", "tags": tags | {"heavy", "two_handed"}, "finesse": False, "heavy": True}

    if "gran hacha" in name or "greataxe" in name:
        heavy = True
        two_handed = True
        return {"die": "1d12", "tags": tags | {"heavy", "two_handed"}, "finesse": False, "heavy": True}

    if "maza" in name and ("dos manos" in name or "2h" in name or "maul" in name):
        heavy = True
        two_handed = True
        return {"die": "2d6", "tags": tags | {"heavy", "two_handed"}, "finesse": False, "heavy": True}

    # Si el item declara explícitamente tags, respétalos (si tu inventario los trae)
    raw_tags = item.get("tags")
    if isinstance(raw_tags, list):
        for t in raw_tags:
            tt = _norm(str(t))
            if tt:
                tags.add(tt)

    # fallback estable
    return {"die": "1d8", "tags": tags, "finesse": False, "heavy": False}


def _attack_stat_mod(member: dict, finesse: bool, attack_type: str) -> int:
    ab = member.get("abilities", {}) or {}
    str_mod = _ability_mod(ab.get("str", 10))
    dex_mod = _ability_mod(ab.get("dex", 10))
    if attack_type == "ranged":
        return dex_mod
    return max(dex_mod, str_mod) if finesse else str_mod


def _parse_forced_actor(actor_token: str) -> tuple[str, str]:
    """
    Permite: "Kaelen (rapier)" o "Myrmyr (bow)"
    Devuelve: ("Kaelen", "rapier")
    """
    m = re.match(r"^(.+?)\s*\((.+?)\)\s*$", (actor_token or "").strip())
    if not m:
        return (actor_token or "").strip(), ""
    return m.group(1).strip(), _norm(m.group(2))


def _pick_weapon(member: dict, attack_type: str, forced: str = "") -> dict:
    eq = _equipped_items(member)
    if not eq:
        return {}

    def nm(it): return _norm(it.get("name", ""))

    if forced:
        forced_map = {
            "arco": ["arco", "bow", "longbow", "shortbow"],
            "bow": ["arco", "bow", "longbow", "shortbow"],
            "estoque": ["estoque", "rapier"],
            "rapier": ["estoque", "rapier"],
            "alabarda": ["alabarda", "halberd"],
            "halberd": ["alabarda", "halberd"],
            "martillo": ["martillo", "warhammer"],
            "warhammer": ["martillo", "warhammer"],
            "bastón": ["bastón", "staff", "quarterstaff"],
            "staff": ["bastón", "staff", "quarterstaff"],
            "melee": ["alabarda", "halberd", "estoque", "rapier", "martillo", "warhammer", "bastón", "staff"],
            "ranged": ["arco", "bow", "ballesta", "crossbow"],
        }
        keys = forced_map.get(forced, [forced])
        for it in eq:
            n = nm(it)
            if any(k in n for k in keys):
                return it

    if attack_type == "ranged":
        for it in eq:
            n = nm(it)
            if "arco" in n or "bow" in n or "ballesta" in n or "crossbow" in n:
                return it
        return eq[0]

    # melee preference: heavy reach > finesse > others
    for it in eq:
        n = nm(it)
        if "alabarda" in n or "halberd" in n or "glaive" in n or "guja" in n:
            return it
    for it in eq:
        n = nm(it)
        if "estoque" in n or "rapier" in n or "espada corta" in n or "shortsword" in n:
            return it
    return eq[0]


def _auto_roll_mode_from_target(target_obj: dict, attack_type: str) -> tuple[str, list]:
    conds = _ensure_conditions(target_obj)
    cond_names = {str(c.get("name", "")).strip().lower() for c in conds}
    notes = []
    mode = "normal"
    if "paralyzed" in cond_names:
        mode = "adv"
        notes.append("AUTO: target PARALYZED => ADV")
    if "prone" in cond_names:
        if attack_type == "melee":
            mode = "adv"
            notes.append("AUTO: target PRONE + melee => ADV")
        else:
            mode = "dis"
            notes.append("AUTO: target PRONE + ranged => DIS")
    return mode, notes

def _combat_round_num(canon: dict) -> int:
    combat = canon.get("combat", {}) or {}
    for k in ("round", "round_num", "round_number"):
        if k in combat:
            try:
                return int(combat.get(k) or 1)
            except Exception:
                pass
    return 1

def _combat_once_map(canon: dict, key: str) -> dict:
    combat = canon.setdefault("combat", {})
    flags = combat.setdefault("once_per_combat", {})
    return flags.setdefault(key, {})

_DICE_RE = re.compile(r"^\s*(\d*)d(\d+)\s*([+-]\s*\d+)?\s*$", re.IGNORECASE)

def _roll_damage_expr(expr: str, crit: bool = False) -> dict:
    """
    expr: "1d8+6"
    crit: duplica SOLO los dados, no el modificador flat.
    """
    raw = (expr or "").strip().lower().replace(" ", "")
    m = _DICE_RE.match(raw)
    if not m:
        return {"expr": expr, "ok": False, "total": 0, "text": f"{expr} -> (FORMATO INVÁLIDO)"}

    n_str, sides_str, mod_str = m.groups()
    n = int(n_str) if n_str else 1
    sides = int(sides_str)
    mod = int(mod_str.replace(" ", "")) if mod_str else 0

    rolls1 = [random.randint(1, sides) for _ in range(n)]
    rolls2 = [random.randint(1, sides) for _ in range(n)] if crit else []
    total = sum(rolls1) + sum(rolls2) + mod

    if crit:
        text_out = f"{expr} -> rolls={rolls1}, CRIT rolls={rolls2}, mod={mod}, total={total}"
    else:
        text_out = f"{expr} -> rolls={rolls1}, mod={mod}, total={total}"

    return {"expr": expr, "ok": True, "total": total, "text": text_out}

def _add_flat_to_damage_expr(expr: str, flat: int) -> str:
    raw = (expr or "").strip().lower().replace(" ", "")
    m = _DICE_RE.match(raw)
    if not m:
        return expr
    n_str, sides_str, mod_str = m.groups()
    n = int(n_str) if n_str else 1
    mod = int(mod_str.replace(" ", "")) if mod_str else 0
    mod2 = mod + int(flat or 0)
    base = f"{n}d{sides_str}"
    if mod2 == 0:
        return base
    sign = "+" if mod2 > 0 else ""
    return f"{base}{sign}{mod2}"

def _format_damage_expr(die: str, mod: int) -> str:
    die = (die or "1d8").strip()
    if mod == 0:
        return die
    sign = "+" if mod > 0 else ""
    return f"{die}{sign}{mod}"


def _get_sneak_dice(member: dict) -> str:
    feats = member.get("features") or []
    for f in feats:
        s = str(f)
        m = re.search(r"Sneak\s*Attack\s*\((\d+d\d+)\)", s, re.IGNORECASE)
        if m:
            return m.group(1)
    # fallback razonable
    return "3d6"


def _is_focus_cd_attack(item: dict) -> bool:
    n = _norm(item.get("name", ""))
    return ("cd/attack" in n) or ("cd/ataque" in n) or ("spell attack" in n)


def _spell_attack_bonus(member: dict) -> int:
    classes = member.get("classes") or []
    main = ""
    main_lvl = -1
    for c in classes:
        try:
            lvl = int(c.get("level", 0) or 0)
        except Exception:
            lvl = 0
        if lvl > main_lvl:
            main_lvl = lvl
            main = _norm(c.get("name", ""))

    casting = {
        "wizard": "int",
        "sorcerer": "cha",
        "warlock": "cha",
        "bard": "cha",
        "cleric": "wis",
        "druid": "wis",
        "paladin": "cha",
        "ranger": "wis",
        "artificer": "int",
    }
    ab_key = casting.get(main, "int")
    ab = member.get("abilities", {}) or {}
    mod = _ability_mod(ab.get(ab_key, 10))
    prof = int(member.get("prof_bonus", 0) or 0)

    # foco CD/Attack equipado: suma su bonus si lo tiene
    focus_bonus = 0
    for it in _equipped_items(member):
        if _is_focus_cd_attack(it):
            focus_bonus = max(focus_bonus, _weapon_bonus(it))
    return prof + mod + focus_bonus


def _fire_bolt_damage(member: dict) -> str:
    lvl = int(member.get("total_level", 1) or 1)
    if lvl >= 17:
        return "4d10"
    if lvl >= 11:
        return "3d10"
    if lvl >= 5:
        return "2d10"
    return "1d10"

def _infer_sneak_dice(actor_obj: dict) -> str:
    """
    Devuelve Nd6 para Sneak Attack:
    1) intenta leer 'Sneak Attack (Xd6)' en features
    2) si no, calcula por nivel de Rogue en actor_obj["classes"]
    3) fallback: 1d6
    """
    feats = actor_obj.get("features") or []
    txt = " ".join(str(x) for x in feats).lower()
    m = re.search(r"sneak\s*attack\s*\((\d+d6)\)", txt)
    if m:
        return m.group(1)

    # cálculo por nivel de pícaro en classes[]
    rl = 0
    for c in (actor_obj.get("classes") or []):
        if _norm(c.get("name", "")) == "rogue":
            try:
                rl += int(c.get("level", 0) or 0)
            except Exception:
                pass

    # progresión 5e: 1-2=1d6, 3-4=2d6, 5-6=3d6, etc.
    if rl > 0:
        dice = max(1, (rl + 1) // 2)
        return f"{dice}d6"

    return "1d6"

def tool_resolve_actions(plan_json: str) -> str:
    """
    Ejecuta un Action Plan JSON (sin parser frágil).
    plan_json ejemplo:
    {
      "actions":[
        {"type":"attack","actor":"Myrmyr Lash","target":"Bandit A","count":3,"attack_type":"ranged","weapon_hint":"bow","modes":["sharpshooter"]},
        {"type":"attack","actor":"Kaelen el Sombrío","target":"Bandit A","count":2,"attack_type":"melee","weapon_hint":"rapier","modes":["sneak_attack"]},
        {"type":"spell_attack","actor":"Trip","target":"Bandit A","spell":"Fire Bolt"}
      ]
    }
    """
    try:
        plan = json.loads(plan_json or "{}")
    except Exception as e:
        return f"ERROR: plan_json no es JSON válido: {e}"

    actions = plan.get("actions")
    if not isinstance(actions, list) or not actions:
        return "ERROR: plan_json debe incluir {'actions':[...]}"

    canon = canon_load()
    party = canon.get("party", {}) or {}
    members = party.get("members") or []

    # index rápido de PJ
    members_by_lc = {_norm(m.get("name", "")): m for m in members if m.get("name")}
    out = []
    sneak_used_by = set()  # 1/turn dentro de esta resolución

    def _ensure_target_exists(name: str):
        """
        Asegura que el target exista como party/enemy.
        - intenta spawn_encounter con bestiary (si hay match)
        - si no, crea stub razonable (AC 10 / HP 10), NUNCA 999
        """
        name = (name or "").strip()
        if not name:
            return

        c = canon_load()
        if _get_target_container(c, name):
            return

        # Intento: spawn real (bestiary/generado)
        try:
            payload = {"enemies": [{"name": name}]}
            tool_spawn_encounter(json.dumps(payload, ensure_ascii=False))
        except Exception:
            pass

        c2 = canon_load()
        if _get_target_container(c2, name):
            return

        # Último recurso: stub razonable
        try:
            _ensure_enemy_stub_anywhere(c2, name, defaults={"ac": 10, "max_hp": 10, "hp_current": 10})
            canon_save(c2)
        except Exception:
            pass

        # si ya existe como party/enemy, ok
        c = canon_load()
        if _get_target_container(c, name):
            return

        # si no existe, intenta autospawn desde bestiary (si hay helper disponible)
        try:
            # intenta usar spawn_encounter si existe en tu agent.py
            if "tool_spawn_encounter" in globals():
                tool_spawn_encounter(name, 1)
        except Exception:
            pass

        # si sigue sin existir, crea stub como enemigo (último recurso)
        c2 = canon_load()
        if not _get_target_container(c2, name):
            try:
                _ensure_enemy_stub_anywhere(c2, name, defaults={"ac": 10, "max_hp": 999, "hp_current": 999})
                canon_save(c2)
            except Exception:
                pass

    for i, a in enumerate(actions, start=1):
        if not isinstance(a, dict):
            out.append(f"[{i}] ERROR: acción no es objeto JSON.")
            continue

        typ = _norm(a.get("type", ""))
        actor_raw = a.get("actor", "") or ""
        target = a.get("target", "") or ""

        actor, forced = _parse_forced_actor(actor_raw)

        if not typ:
            out.append(f"[{i}] ERROR: falta 'type'.")
            continue

        # acciones simples
        if typ == "move":
            feet = int(a.get("feet", 0) or 0)
            out.append(f"[{i}] MOVE: {actor} {feet} ft")
            out.append(tool_move(actor, feet))
            continue

        if typ == "apply_condition":
            cond = a.get("condition", "")
            rounds = int(a.get("rounds", 1) or 1)
            tgt = target or actor
            _ensure_target_exists(tgt)
            out.append(f"[{i}] APPLY_CONDITION: {tgt} {cond} ({rounds})")
            out.append(tool_apply_condition(tgt, cond, rounds))
            continue

        if typ == "remove_condition":
            cond = a.get("condition", "")
            tgt = target or actor
            _ensure_target_exists(tgt)
            out.append(f"[{i}] REMOVE_CONDITION: {tgt} {cond}")
            out.append(tool_remove_condition(tgt, cond))
            continue

        if typ == "damage":
            amt = int(a.get("amount", 0) or 0)
            tgt = target or actor
            _ensure_target_exists(tgt)
            out.append(f"[{i}] DAMAGE: {tgt} {amt}")
            out.append(tool_damage(tgt, amt))
            continue

        if typ == "heal":
            amt = int(a.get("amount", 0) or 0)
            tgt = target or actor
            out.append(f"[{i}] HEAL: {tgt} {amt}")
            out.append(tool_heal(tgt, amt))
            continue

        # asegurar target si hace falta
        if target:
            _ensure_target_exists(target)

        # ATTACK / SPELL_ATTACK
        if typ in {"attack", "spell_attack"}:
            # localizar actor
            canon_now = canon_load()
            found_a = _get_target_container(canon_now, actor)
            if not found_a:
                out.append(f"[{i}] ERROR: no encuentro actor '{actor}'.")
                continue
            a_side, actor_obj, actor_name = found_a

            # localizar target (permite party o enemy)
            found_t = _get_target_container(canon_now, target) if target else None
            if not found_t:
                out.append(f"[{i}] ERROR: no encuentro target '{target}'.")
                continue
            t_side, target_obj, target_name = found_t

            count = int(a.get("count", 1) or 1)
            count = max(1, min(20, count))

            attack_type = _norm(a.get("attack_type", "")) or ("ranged" if typ == "spell_attack" else "melee")
            if attack_type not in {"melee", "ranged"}:
                attack_type = "melee"

            roll_mode = _norm(a.get("roll_mode", "auto"))  # auto|normal|adv|dis
            if roll_mode not in {"auto", "normal", "adv", "dis"}:
                roll_mode = "auto"

            modes = a.get("modes") or []
            if isinstance(modes, str):
                modes = [modes]
            modes = {_canon_mode_token(x) for x in modes if str(x).strip()}
            modes = {m for m in modes if m}

            # fallback: si el plan no puso attack_type pero sí modo, infiere
            if not _norm(a.get("attack_type", "")):
                if "sharpshooter" in modes:
                    attack_type = "ranged"
                elif "gwm" in modes:
                    attack_type = "melee"

            crit_range = int(a.get("crit_range", 20) or 20)
            crit_range = max(2, min(20, crit_range))

            # build actor features lc
            actor_features_lc = []
            if a_side == "party":
                actor_features_lc = [_norm(x) for x in (actor_obj.get("features") or [])]
            else:
                actor_features_lc = [_norm(x) for x in (actor_obj.get("features") or [])] if isinstance(actor_obj.get("features"), list) else []

            # base bonus/damage
            weapon_tags = set()
            weapon_item = {}
            die = a.get("damage_die")  # override opcional
            base_bonus = None
            base_damage_expr = None

            if typ == "spell_attack":
                if a_side != "party":
                    out.append(f"[{i}] ERROR: spell_attack requiere actor PJ (por ahora).")
                    continue
                atk_bonus = _spell_attack_bonus(actor_obj)
                dmg_die = "1d10"
                spell = _norm(a.get("spell", ""))
                if "firebolt" in spell or "fire bolt" in spell:
                    dmg_die = _fire_bolt_damage(actor_obj)
                # sin mod de habilidad por defecto en cantrip tipo Fire Bolt
                base_bonus = atk_bonus
                base_damage_expr = dmg_die
                weapon_tags = {"spell_attack", "ranged"}
            else:
                if a_side == "party":
                    weapon_hint = _norm(a.get("weapon_hint", "")) or forced
                    weapon_item = _pick_weapon(actor_obj, attack_type, weapon_hint)
                    prof = int(actor_obj.get("prof_bonus", 0) or 0)
                    wb = _weapon_bonus(weapon_item) if weapon_item else 0

                    prof_w = _weapon_profile_from_item(weapon_item or {}, attack_type)
                    die = die or prof_w["die"]
                    weapon_tags = set(prof_w["tags"] or set())
                    finesse = bool(prof_w.get("finesse"))
                    stat_mod = _attack_stat_mod(actor_obj, finesse, attack_type)

                    base_bonus = prof + stat_mod + wb
                    base_damage_expr = _format_damage_expr(die, stat_mod + wb)
                else:
                    # Enemy: por ahora requiere que el plan provea bonus/damage o usa defaults.
                    base_bonus = int(a.get("bonus", 0) or actor_obj.get("atk_bonus", 0) or 0)
                    die = die or str(a.get("damage", "1d8"))
                    base_damage_expr = str(a.get("damage", die))
                    weapon_tags = {"enemy"}

            # Heurística: si es melee y el dado base es 1d12 o 2d6 0 1d10, trátalo como arma heavy (para GWM)
            if attack_type == "melee":
                d0 = (die or "").strip().lower().replace(" ", "")
                if d0 in {"1d12", "2d6"}:
                    weapon_tags.add("heavy")
                    weapon_tags.add("two_handed")

            # ejecutar N ataques
            out.append(f"[{i}] {typ.upper()}: {actor_name} -> {target_name} x{count} ({attack_type})")
            # Dread Ambusher (Gloom Stalker): 1/combate en round 1, añade 1d8 al primer HIT de esta acción
            dread_pending = False
            if typ == "attack" and a_side == "party":
                has_dread = (
                    any("dread ambusher" in f or "gloom stalker" in f for f in actor_features_lc)
                    or ("dread_ambusher" in modes)
                )
                if has_dread:
                    ctmp = canon_load()
                    if (ctmp.get("combat", {}) or {}).get("active") and _combat_round_num(ctmp) == 1:
                        used_map = _combat_once_map(ctmp, "dread_ambusher_used")
                        if not used_map.get(_norm(actor_name)):
                            dread_pending = True
            for k in range(1, count + 1):
                # modo adv/dis
                auto_notes = []
                if roll_mode == "auto":
                    rm, auto_notes = _auto_roll_mode_from_target(target_obj, attack_type)
                    mode_use = rm
                else:
                    mode_use = roll_mode

                # contexto para modifiers.json
                ctx = {
                    "attack_type": attack_type,
                    "modes": modes,
                    "weapon_tags": weapon_tags,
                    "actor_features_lc": actor_features_lc,
                }

                to_hit_delta, dmg_bonus_flat, mod_notes = _apply_attack_modifiers(ctx)

                bonus = int(base_bonus or 0) + int(to_hit_delta or 0)
                dmg_expr = _add_flat_to_damage_expr(base_damage_expr, int(dmg_bonus_flat or 0))

                # tirada ataque
                r1 = random.randint(1, 20)
                r2 = random.randint(1, 20) if mode_use in {"adv", "dis"} else None
                chosen = r1 if r2 is None else (max(r1, r2) if mode_use == "adv" else min(r1, r2))
                total = chosen + bonus

                ac = int(target_obj.get("ac", 10) or 10)
                hit = total >= ac
                is_crit = chosen >= crit_range

                out.append(f"  - [{k}/{count}] d20: {r1}" + (f", {r2} -> {chosen}" if r2 is not None else f" -> {chosen}") + f" +{bonus} = {total} vs AC {ac} => " + ("HIT" if hit else "MISS"))

                notes_all = []
                notes_all.extend(auto_notes)
                notes_all.extend(mod_notes)
                if notes_all:
                    out.append("    NOTES: " + " | ".join(notes_all))

                if not hit:
                    continue

                # daño + aplicar
                dmg_roll = _roll_damage_expr(dmg_expr, crit=is_crit)
                out.append("    DMG: " + dmg_roll["text"])
                out.append("    " + tool_damage(target_name, int(dmg_roll["total"])))
                
                # Extra Dread Ambusher: solo si este ataque HIT y está pendiente
                if dread_pending and hit:
                    # en crítico duplica los dados extra también
                    extra = "2d8" if is_crit else "1d8"
                    ex_roll = _roll_damage_expr(extra, crit=False)
                    out.append("    DREAD AMBUSHER: " + ex_roll["text"])
                    out.append("    " + tool_damage(target_name, int(ex_roll["total"])))
                    dread_pending = False

                    # marcar usado 1/combate
                    ctmp2 = canon_load()
                    used_map2 = _combat_once_map(ctmp2, "dread_ambusher_used")
                    used_map2[_norm(actor_name)] = True
                    canon_save(ctmp2)

                # Sneak Attack (1/turn dentro de este resolve)
                if hit and ("sneak_attack" in modes) and (a_side == "party") and (_norm(actor_name) not in sneak_used_by):
                    sneak_dice = a.get("sneak_dice") or _infer_sneak_dice(actor_obj)
                    sneak_dice = str(sneak_dice or "").strip().lower().replace(" ", "")

                    if not _DICE_RE.match(sneak_dice):
                        sneak_dice = "1d6"

                    # si el ataque fue crítico, _roll_damage_expr(..., crit=True) ya duplica dados
                    sa_roll = _roll_damage_expr(sneak_dice, crit=is_crit)
                    out.append("    SNEAK: " + sa_roll["text"])
                    out.append("    " + tool_damage(target_name, int(sa_roll["total"])))

                    sneak_used_by.add(_norm(actor_name))

            continue

        out.append(f"[{i}] ERROR: type '{typ}' no soportado todavía.")

    return "\n".join(out)

def tool_stabilize(target: str) -> str:
    canon = canon_load()
    m = _find_party_member(canon, target)
    if not m:
        return f"Objetivo '{target}' no es un PJ."
    hp = int(m.get("hp", 0))
    if hp > 0:
        return f"{m.get('name')} no está a 0 HP; no necesita estabilización."
    m["stable"] = True
    canon_save(canon)
    return f"{m.get('name')} queda ESTABLE a 0 HP (inconsciente)."

def tool_rest(kind: str = "short") -> str:
    kind = (kind or "short").strip().lower()
    if kind not in {"short", "long"}:
        return "kind debe ser 'short' o 'long'"
    canon = canon_load()
    members = canon.get("party", {}).get("members", [])
    if not members:
        return "No hay miembros del grupo."
    lines = [f"Descanso: {kind}"]
    for m in members:
        name = m.get("name", "?")
        hp = int(m.get("hp", 0))
        max_hp = int(m.get("max_hp", hp))
        if kind == "long":
            m["hp"] = max_hp
            m["stable"] = True
            lines.append(f"- {name}: {hp} → {max_hp}")
        else:
            heal_amt = max(1, max_hp // 4)
            new_hp = min(max_hp, hp + heal_amt)
            m["hp"] = new_hp
            if new_hp > 0:
                m["stable"] = True
            lines.append(f"- {name}: {hp} → {new_hp} (+{heal_amt})")
    canon_save(canon)
    return "\n".join(lines)


# =========================================================
# Tools: movimiento
# =========================================================
def _get_speed(obj: dict, default_speed: int = 30) -> int:
    try:
        return int(obj.get("speed", default_speed))
    except Exception:
        return default_speed

def _set_move_left(obj: dict, value: int) -> None:
    obj["move_left"] = max(0, int(value))

def _get_move_left(obj: dict) -> int:
    try:
        return int(obj.get("move_left", 0))
    except Exception:
        return 0

def tool_start_turn() -> str:
    canon = canon_load()
    combat = canon.get("combat", {}) or {}
    if not combat.get("active"):
        return "No hay combate activo."
    idx = int(combat.get("turn_index", 0))
    order = combat.get("order", []) or []
    if not order or idx >= len(order):
        return "Orden de combate inválido."
    actor_name = order[idx]["name"]
    found = _get_target_container(canon, actor_name)
    if not found:
        return f"No encuentro al actor actual '{actor_name}'."
    _, obj, cname = found
    speed = _get_speed(obj, 30)
    _set_move_left(obj, speed)
    canon_save(canon)
    return f"Inicio de turno: {cname}. Movimiento disponible: {speed} ft."

def tool_move(target: str, feet: int) -> str:
    canon = canon_load()
    found = _get_target_container(canon, target)
    if not found:
        return f"No encuentro el objetivo '{target}'."
    _, obj, cname = found
    feet = int(feet)
    if feet <= 0:
        return "feet debe ser > 0"
    if _has_condition(obj, "RESTRAINED"):
        return f"{cname} está RESTRAINED y no puede moverse."
    move_left = _get_move_left(obj)
    if move_left <= 0:
        return f"{cname} no tiene movimiento disponible (move_left=0)."
    if feet > move_left:
        return f"{cname} solo tiene {move_left} ft de movimiento; no puede mover {feet}."
    _set_move_left(obj, move_left - feet)
    canon_save(canon)
    return f"{cname} se mueve {feet} ft. Movimiento restante: {_get_move_left(obj)} ft."

def tool_stand_up(target: str) -> str:
    canon = canon_load()
    found = _get_target_container(canon, target)
    if not found:
        return f"No encuentro el objetivo '{target}'."
    _, obj, cname = found
    if not _has_condition(obj, "PRONE"):
        return f"{cname} no está PRONE."
    if _has_condition(obj, "RESTRAINED"):
        return f"{cname} está RESTRAINED y no puede levantarse."
    speed = _get_speed(obj, 30)
    cost = max(5, speed // 2)
    move_left = _get_move_left(obj)
    if move_left < cost:
        return f"{cname} necesita {cost} ft para levantarse, pero solo tiene {move_left}."
    _set_move_left(obj, move_left - cost)
    obj["conditions"] = [c for c in (obj.get("conditions", []) or []) if str(c.get("name","")).upper() != "PRONE"]
    canon_save(canon)
    return f"{cname} se levanta (gasta {cost} ft). Movimiento restante: {_get_move_left(obj)} ft."


# =========================================================
# Tools: combate (iniciativa + turnos + tick condiciones)
# =========================================================
def tool_start_combat(combatants_json: str) -> str:
    try:
        combatants = json.loads(combatants_json)
        if not isinstance(combatants, list) or not combatants:
            return "Error: combatants_json debe ser una lista JSON no vacía."
    except Exception as e:
        return f"Error: JSON inválido: {e}"

    order = []
    for c in combatants:
        if not isinstance(c, dict) or "name" not in c:
            return "Error: cada combatiente debe ser objeto con al menos 'name'."
        name = str(c.get("name"))
        side = str(c.get("side", "unknown"))
        init_bonus = int(c.get("init_bonus", 0))
        roll = random.randint(1, 20)
        init_total = roll + init_bonus
        order.append({
            "name": name,
            "side": side,
            "init_bonus": init_bonus,
            "init_roll": roll,
            "init_total": init_total
        })

    order.sort(key=lambda x: (x["init_total"], x["init_roll"], x["name"]), reverse=True)

    canon = canon_load()
    canon["combat"] = {
        "active": True,
        "round": 1,
        "turn_index": 0,
        "order": order,
        "bonus_attack_available": False,
        "bonus_attack_owner": None,
        "pending_reaction": None,
    }
    # --- REGISTRO AUTOMÁTICO DE ENEMIGOS EN CANON ---
    canon.setdefault("enemies", {})
    enemies = canon["enemies"]

    for c in combatants:
        name = str(c.get("name", "")).strip()
        side = str(c.get("side", "unknown")).strip().lower()
        if not name or side == "party":
            continue

        # permite que el JSON de combatants incluya ac/hp si quieres
        defaults = {
            "ac": c.get("ac", 10),
            "hp": c.get("hp", 999),
            "hp_max": c.get("hp_max", c.get("hp", 999)),
            "conditions": [],
            "stub": bool(c.get("stub", True)),
        }

        # crea o asegura mínimos
        if name not in enemies:
            enemies[name] = {
                "name": name,
                "ac": int(defaults["ac"]) if str(defaults["ac"]).isdigit() else 10,
                "hp": int(defaults["hp"]) if str(defaults["hp"]).isdigit() else 999,
                "hp_max": int(defaults["hp_max"]) if str(defaults["hp_max"]).isdigit() else 999,
                "conditions": [],
                "stub": defaults["stub"],
            }
        else:
            enemies[name].setdefault("name", name)
            enemies[name].setdefault("conditions", [])
            enemies[name].setdefault("ac", 10)
            enemies[name].setdefault("hp", 999)
            enemies[name].setdefault("hp_max", enemies[name].get("hp", 999))

    canon_save(canon)

    current = order[0]
    pretty = "\n".join([f"{i+1}. {o['name']} ({o['side']}) init={o['init_total']} [{o['init_roll']}+{o['init_bonus']}]" for i, o in enumerate(order)])
    return (
        "Combate iniciado.\n"
        f"Ronda: 1\n"
        f"Turno actual: {current['name']} ({current['side']})\n"
        "Orden de iniciativa:\n"
        f"{pretty}"
    )

def tool_start_combat_quick(enemies_json: str) -> str:
    """
    Inicia combate usando los PJs del canon + una lista de enemigos en JSON.

    enemies_json ejemplo:
    [
      {"name":"Acolyte", "init_bonus":2},
      {"name":"Aboleth", "init_bonus":0}
    ]

    Si un enemigo no existe en canon["enemies"], intentamos auto-crearlo desde dm/bestiary.json
    (vía _ensure_enemy_from_bestiary). Si tampoco existe en bestiary, metemos defaults.
    """
    try:
        enemies = json.loads(enemies_json)
        if not isinstance(enemies, list) or not enemies:
            return "Error: enemies_json debe ser una lista JSON no vacía."
    except Exception as e:
        return f"Error: JSON inválido: {e}"

    canon = canon_load()

    # Asegurar que el contenedor de enemigos es un dict
    enemies_db = canon.get("enemies", {})
    if not isinstance(enemies_db, dict):
        enemies_db = {}
        canon["enemies"] = enemies_db

    # Asegurar que los enemigos existen en canon["enemies"]
    for e in enemies:
        if not isinstance(e, dict):
            continue
        name = str(e.get("name", "")).strip()
        if not name:
            continue

        # 1) Si ya existe, ok
        if name in enemies_db:
            continue

        # 2) Si no existe, intenta auto-crear desde bestiary (por nombre)
        spawned = _ensure_enemy_from_bestiary(canon, name)
        if spawned:
            inst_name, _obj = spawned
            # Si el bestiary te crea con nombre distinto (por sufijo #2, etc),
            # y el usuario te pidió un "name" concreto, reflejamos el nombre en la lista
            # para que iniciativa use el nombre real en canon.
            if inst_name != name:
                e["name"] = inst_name
            continue

        # 3) Fallback defaults (si no está en bestiary)
        enemies_db[name] = {"speed": 30, "ac": 13, "hp": 11, "hp_current": 11, "conditions": []}

    canon["enemies"] = enemies_db
    canon_save(canon)

    members = canon.get("party", {}).get("members", [])
    if not members:
        return "Error: no hay miembros del grupo en el canon."

    combatants = []
    for m in members:
        combatants.append({
            "name": m.get("name"),
            "side": "party",
            "init_bonus": int(m.get("init_bonus", 0)),
        })

    for e in enemies:
        if not isinstance(e, dict):
            continue
        combatants.append({
            "name": e.get("name"),
            "side": "enemy",
            "init_bonus": int(e.get("init_bonus", 0)),
        })

    return tool_start_combat(json.dumps(combatants, ensure_ascii=False))


def tool_combat_status() -> str:
    canon = canon_load()
    combat = canon.get("combat", {}) or {}
    if not combat.get("active"):
        return "No hay combate activo."
    order = combat.get("order", []) or []
    idx = int(combat.get("turn_index", 0))
    rnd = int(combat.get("round", 1))
    current = order[idx] if order else {"name": "?", "side": "?"}
    pretty = "\n".join([f"{i+1}. {o['name']} ({o['side']}) init={o['init_total']}" for i, o in enumerate(order)])
    return f"Ronda: {rnd}\nTurno actual: {current['name']} ({current['side']})\nOrden:\n{pretty}"

def _tick_conditions_for(canon: dict, name: str) -> None:
    found = _get_target_container(canon, name)
    if not found:
        return
    _, obj, _ = found
    conds = _ensure_conditions(obj)

    new_conds = []
    for c in conds:
        rem = int(c.get("remaining", 0)) - 1
        if rem > 0:
            new_conds.append({"name": c.get("name", ""), "remaining": rem})
    obj["conditions"] = new_conds


def tool_next_turn() -> str:
    canon = canon_load()
    combat = canon.get("combat", {}) or {}
    if not combat.get("active"):
        return "No hay combate activo."
    order = combat.get("order", []) or []
    if not order:
        return "Combate activo pero sin orden de iniciativa."

    idx = int(combat.get("turn_index", 0))
    rnd = int(combat.get("round", 1))

    current_actor = order[idx]["name"]
    _tick_conditions_for(canon, current_actor)  # ✅ ya no guarda aparte

    idx += 1
    if idx >= len(order):
        idx = 0
        rnd += 1

    combat["turn_index"] = idx
    combat["round"] = rnd
    canon["combat"] = combat

    # ✅ un solo guardado al final
    canon_save(canon)

    start_msg = tool_start_turn()
    current = order[idx]
    return f"{start_msg}\nRonda: {rnd}\nTurno: {current['name']} ({current['side']})"

def tool_end_combat() -> str:
    canon = canon_load()
    canon["combat"] = {"active": False}
    canon_save(canon)
    return "Combate finalizado."


# =========================================================
# Tools: ataque (simple, pero estable) + bonus attack flag
# =========================================================
def _avg_damage(expr: str) -> float:
    expr = (expr or "").strip().lower().replace(" ", "")
    m = re.fullmatch(r"(\d*)d(\d+)([+-]\d+)?", expr)
    if not m:
        return 0.0
    n_str, sides_str, mod_str = m.groups()
    n = int(n_str) if n_str else 1
    sides = int(sides_str)
    mod = int(mod_str) if mod_str else 0
    return n * (1 + sides) / 2.0 + mod

def _hit_prob(target_ac: int, atk_bonus: int, mode: str) -> float:
    mode = (mode or "normal").lower()
    roll_needed = target_ac - atk_bonus
    if roll_needed <= 1:
        p = 1.0
    elif roll_needed > 20:
        p = 0.0
    else:
        p = (21 - roll_needed) / 20.0
    if mode == "adv":
        return 1.0 - (1.0 - p) ** 2
    if mode == "dis":
        return p ** 2
    return p

def _cover_bonus(cover: str) -> int:
    return {"none": 0, "half": 2, "three-quarters": 5, "total": 999}.get((cover or "none").lower(), 0)

def tool_attack(
    attacker: str,
    target: str,
    bonus: int,
    damage: str = "1d8",
    attack_type: str = "melee",
    power_shot: bool = False,
    power_attack: bool = False,
    auto_power: bool = True,
    crit_range: int = 20,
    mode: str = "auto"  # auto|normal|adv|dis
) -> str:
    attack_type = (attack_type or "melee").lower()
    if attack_type not in {"melee", "ranged"}:
        return "attack_type debe ser 'melee' o 'ranged'"

    mode = (mode or "auto").lower()
    if mode not in {"auto", "normal", "adv", "dis"}:
        return "mode debe ser 'auto', 'normal', 'adv' o 'dis'"

    canon = canon_load()
    found_a = _get_target_container(canon, attacker)
    found_t = _get_target_container(canon, target)

    # si target no existe pero está en combate, crea stub y reintenta
    if not found_t:
        _ensure_enemy_stub_from_combat(canon, target)
        canon = canon_load()
        found_t = _get_target_container(canon, target)

    if not found_a:
        return f"No encuentro atacante '{attacker}'."
    if not found_t or found_t[0] != "enemy":
        return f"Objetivo '{target}' no existe como enemigo en canon['enemies']."

    _, attacker_obj, attacker_name = found_a
    _, target_obj, target_name = found_t

    # condiciones del objetivo
    conds = _ensure_conditions(target_obj)
    cond_names = {str(c.get("name", "")).lower() for c in conds}

    # calcular adv/dis automático si mode=auto
    auto_mode = "normal"
    notes = []

    if mode == "auto":
        # PARALYZED => ventaja
        if "paralyzed" in cond_names:
            auto_mode = "adv"
            notes.append("AUTO: target PARALYZED => ADV")
        # PRONE => melee adv, ranged dis
        if "prone" in cond_names:
            if attack_type == "melee":
                auto_mode = "adv"
                notes.append("AUTO: target PRONE + melee => ADV")
            else:
                auto_mode = "dis"
                notes.append("AUTO: target PRONE + ranged => DIS")
        mode_use = auto_mode
    else:
        mode_use = mode

    # tirar d20 (visible)
    r1 = random.randint(1, 20)
    r2 = random.randint(1, 20) if mode_use in {"adv", "dis"} else None

    chosen = r1
    if r2 is not None:
        chosen = max(r1, r2) if mode_use == "adv" else min(r1, r2)

    total = chosen + int(bonus)

    # AC (si es stub, será 10 por defecto)
    ac = int(target_obj.get("ac", 10))

    # crit check
    is_crit = chosen >= int(crit_range)

    # PARALYZED: si impacta en melee normalmente es crit automático a 5 ft; no tenemos distancia,
    # pero por defecto, si attack_type==melee y target PARALYZED, marcamos "crit_on_hit" informativo.
    crit_on_hit = False
    if attack_type == "melee" and "paralyzed" in cond_names:
        crit_on_hit = True
        notes.append("PARALYZED: si estás a 5 ft, el golpe es CRIT al impactar (regla 5e).")

    hit = total >= ac

    # daño (visible) -> reutiliza tool_roll para mostrar tiradas
    dmg_out = tool_roll(damage)
    # si crítico (o crit_on_hit y hit), duplicar dados: forma simple -> volver a tirar el mismo daño una vez más
    if hit and (is_crit or crit_on_hit):
        dmg_out_crit = tool_roll(damage)
        dmg_out = f"{dmg_out}\nCRIT extra:\n{dmg_out_crit}"

    # construir salida
    lines = []
    lines.append(f"ATAQUE: {attacker_name} -> {target_name} ({attack_type})")
    lines.append(f"MODE: {mode_use}")
    if r2 is None:
        lines.append(f"d20: {r1} + bonus {bonus} = {total}")
    else:
        lines.append(f"d20: {r1}, {r2} -> elegido {chosen} + bonus {bonus} = {total}")

    lines.append(f"AC objetivo: {ac}" + (" (STUB)" if target_obj.get("stub") else ""))
    lines.append("RESULTADO: IMPACTA" if hit else "RESULTADO: FALLA")

    if notes:
        lines.append("NOTAS: " + " | ".join(notes))

    if hit:
        lines.append("DAÑO:\n" + str(dmg_out))

    return "\n".join(lines)

def tool_bonus_attack(attacker: str, target: str, bonus: int, damage: str = "1d8") -> str:
    canon = canon_load()
    combat = canon.get("combat", {}) or {}
    if not combat.get("active"):
        return "No hay combate activo."
    if not combat.get("bonus_attack_available"):
        return "No hay ataque extra disponible."
    owner = combat.get("bonus_attack_owner")
    if _norm(owner) != _norm(attacker):
        return f"El ataque extra disponible es de {owner}, no de {attacker}."

    combat["bonus_attack_available"] = False
    combat["bonus_attack_owner"] = None
    canon["combat"] = combat
    canon_save(canon)

    result = tool_attack(
        attacker=attacker,
        target=target,
        bonus=bonus,
        damage=damage,
        attack_type="melee",
        power_attack=False,
        auto_power=True,
    )
    return "BONUS ATTACK\n" + result


# =========================================================
# Tools registry
# =========================================================
TOOLS: Dict[str, Callable[..., str]] = {
    "canon_get": tool_canon_get,
    "canon_patch": tool_canon_patch,
    "memory_get": tool_memory_get,
    "memory_set": tool_memory_set,
    "log_event": tool_log_event,
    "give_item": tool_give_item,
    "add_loot": tool_add_loot,

    "spell_info": tool_spell_info,
    "spell_search": tool_spell_search,

    "module_load": tool_module_load,
    "module_query": tool_module_query,
    "module_quote": tool_module_quote,
    "module_set_progress": tool_module_set_progress,
    "update_recap": tool_update_recap,

    "start_scene": tool_start_scene,
    "scene_status": tool_scene_status,

    "roll": tool_roll,
    "check": tool_check,
    "skill_check": tool_skill_check,

    "start_turn": tool_start_turn,
    "move": tool_move,
    "stand_up": tool_stand_up,
    "set_range": tool_set_range,
    "get_range": tool_get_range,
    "approach": tool_approach,
    "retreat": tool_retreat,

    "apply_condition": tool_apply_condition,
    "remove_condition": tool_remove_condition,
    "register_enemies": tool_register_enemies,
    "spawn_encounter": tool_spawn_encounter,
    "target_status": tool_target_status,
    "conditions_status": tool_conditions_status,
    "heal": tool_heal,
    "damage": tool_damage,
    "execute_actions": tool_execute_actions,
    "resolve_actions": tool_resolve_actions,
    "stabilize": tool_stabilize,
    "rest": tool_rest,

    "start_combat": tool_start_combat,
    "start_combat_quick": tool_start_combat_quick,
    "combat_status": tool_combat_status,
    "next_turn": tool_next_turn,
    "end_combat": tool_end_combat,

    "attack": tool_attack,
    "bonus_attack": tool_bonus_attack,

    "xp_status": tool_xp_status,
    "xp_progress": tool_xp_progress,
    "xp_kill": tool_xp_kill,
    "level_status": tool_level_status,
    "level_check_up": tool_level_check_up,
    "level_up_announce": tool_level_up_announce,
}

# =========================================================
# Tool schemas
# =========================================================
def _schema(name: str, desc: str, props: dict, required: list) -> dict:
    return {
        "type": "function",
        "name": name,
        "description": desc,
        "parameters": {"type": "object", "properties": props, "required": required},
    }

TOOL_SCHEMAS = [
    _schema("canon_get", "Lee una clave del canon (estado) y devuelve JSON string.", {"key": {"type": "string"}}, ["key"]),
    _schema("canon_patch", "Aplica un patch JSON (string) al canon (merge recursivo).", {"json_text": {"type": "string"}}, ["json_text"]),
    _schema("log_event", "Añade un evento breve al log de sesión.", {"text": {"type": "string"}}, ["text"]),
    
    _schema("give_item", "Da un objeto mágico/ítem a un miembro del grupo usando dm/items.json.", {
        "member": {"type": "string"},
        "item": {"type": "string"},
        "qty": {"type": "integer"}
    }, ["member", "item"]),

    _schema("add_loot", "Añade un objeto mágico/ítem al loot común usando dm/items.json.", {
        "item": {"type": "string"},
        "qty": {"type": "integer"}
    }, ["item"]),

    _schema("spell_info", "Devuelve la definición completa de un spell desde dm/spells.json.", {
        "name": {"type": "string"}
    }, ["name"]),

    _schema("spell_search", "Busca spells por nombre (substring) en dm/spells.json.", {
        "query": {"type": "string"},
        "limit": {"type": "integer"}
    }, []),

    _schema(
        "resolve_actions",
        "Ejecuta un Action Plan JSON (múltiples acciones encadenadas) sin parsers frágiles.",
        {"plan_json": {"type": "string"}},
        ["plan_json"]
    ),

    _schema("module_load", "Indexa y carga un módulo PDF y lo marca activo en canon.session.module.", {
        "module_id": {"type": "string"},
        "pdf_path": {"type": "string"}
    }, []),

    _schema("module_query", "Busca texto del módulo activo (top_k resultados con id/página/snippet).", {
        "query": {"type": "string"},
        "top_k": {"type": "integer"},
        "page_min": {"type": "integer"},
        "page_max": {"type": "integer"}
    }, ["query"]),

    _schema("module_quote", "Devuelve el texto literal de un chunk por id (para read-aloud/verificación).", {
        "chunk_id": {"type": "string"},
        "max_chars": {"type": "integer"}
    }, ["chunk_id"]),

    _schema("module_set_progress", "Actualiza progreso del módulo en canon.session.module.progress.", {
        "chapter": {"type": "string"},
        "scene": {"type": "string"},
        "add_flags_json": {"type": "string"}
    }, []),

    _schema("update_recap", "Actualiza el resumen persistente de la escena (scene.recap) para continuidad sin releer todo el history.",
        {"text": {"type": "string"}},
        ["text"]
    ),

    _schema("start_scene", "Inicializa escena en canon.session.", {"location": {"type": "string"}, "hook": {"type": "string"}}, ["location", "hook"]),
    _schema("scene_status", "Devuelve estado de escena (JSON).", {}, []),

    _schema("roll", "Tira dados: 2d6+3, 1d20+5, d8, etc.", {"expr": {"type": "string"}}, ["expr"]),
    _schema("check", "Skill check vs DC (normal/adv/dis).", {
        "skill": {"type": "string"},
        "dc": {"type": "integer"},
        "bonus": {"type": "integer"},
        "mode": {"type": "string"}
    }, ["skill", "dc"]),
    _schema("skill_check", "Tirada de habilidad para un PJ calculando bonus automático (mod + PB si competente).", {
        "actor": {"type": "string"},
        "skill": {"type": "string"},
        "dc": {"type": "integer"},
        "mode": {"type": "string"},
        "extra_bonus": {"type": "integer"}
    }, ["actor", "skill", "dc"]),

    _schema("start_combat", "Inicia combate con lista JSON de combatientes.", {"combatants_json": {"type": "string"}}, ["combatants_json"]),
    _schema("start_combat_quick", "Inicia combate usando PJs + enemigos JSON.", {"enemies_json": {"type": "string"}}, ["enemies_json"]),
    _schema("combat_status", "Estado del combate.", {}, []),
    _schema("next_turn", "Avanza turno y hace tick de condiciones.", {}, []),
    _schema("end_combat", "Finaliza combate.", {}, []),

    _schema("start_turn", "Resetea movimiento del actor actual del combate.", {}, []),
    _schema("move", "Mueve un actor X pies (gasta move_left).", {"target": {"type": "string"}, "feet": {"type": "integer"}}, ["target", "feet"]),
    _schema("stand_up", "Levanta a un actor de PRONE gastando movimiento.", {"target": {"type": "string"}}, ["target"]),

    _schema("set_range", "Fija banda de distancia de un objetivo.", {"target": {"type": "string"}, "band": {"type": "string"}}, ["target", "band"]),
    _schema("get_range", "Devuelve banda de distancia entre A y B.", {"a": {"type": "string"}, "b": {"type": "string"}}, ["a", "b"]),
    _schema("approach", "Pone a actor en melee (simplificado).", {"actor": {"type": "string"}}, ["actor"]),
    _schema("retreat", "Aleja a actor a short/medium (simplificado).", {"actor": {"type": "string"}}, ["actor"]),

    _schema("apply_condition", "Aplica condición por X rondas.", {"target": {"type": "string"}, "condition": {"type": "string"}, "rounds": {"type": "integer"}}, ["target", "condition"]),
    _schema("remove_condition", "Quita condición.", {"target": {"type": "string"}, "condition": {"type": "string"}}, ["target", "condition"]),
    _schema("register_enemies", "Registra enemigos en canon['enemies'] a partir de una lista JSON de nombres o de objetos con stats (sin iniciar combate).", {"names_json": {"type": "string", "description": "JSON string: ['Cultista A','Cultista B'] o [{'name':'Cultista A','ac':12,'hp':27}] o {'enemies':[...]}."}}, ["names_json"]),
    _schema("spawn_encounter", "Crea enemigos con stats (bestiary si coincide; si no, genera por role+CR/nivel).", {"encounter_json": {"type":"string"}}, ["encounter_json"]),
    _schema("target_status", "Estado de un objetivo.", {"target": {"type": "string"}}, ["target"]),
    _schema("conditions_status", "Lista condiciones activas.", {}, []),

    _schema("heal", "Cura a un PJ.", {"target": {"type": "string"}, "amount": {"type": "integer"}}, ["target", "amount"]),
    _schema("damage", "Aplica daño a un PJ.", {"target": {"type": "string"}, "amount": {"type": "integer"}}, ["target", "amount"]),
    _schema("execute_actions", "Ejecuta múltiples acciones en cadena (multiataques / disparos / cantrips) a partir de un texto.", {"script": {"type": "string"}, "default_target": {"type": "string"}}, ["script"]),
    _schema("stabilize", "Estabiliza PJ a 0 HP.", {"target": {"type": "string"}}, ["target"]),
    _schema("rest", "Descanso short/long.", {"kind": {"type": "string"}}, []),

    _schema("attack", "Resuelve un ataque contra ENEMIGO.", {
        "attacker": {"type": "string"},
        "target": {"type": "string"},
        "bonus": {"type": "integer"},
        "damage": {"type": "string"},
        "attack_type": {"type": "string"},
        "power_shot": {"type": "boolean"},
        "power_attack": {"type": "boolean"},
        "auto_power": {"type": "boolean"},
        "crit_range": {"type": "integer"}
    }, ["attacker", "target", "bonus"]),
    _schema("bonus_attack", "Ejecuta el ataque extra si está disponible.", {
        "attacker": {"type": "string"},
        "target": {"type": "string"},
        "bonus": {"type": "integer"},
        "damage": {"type": "string"},
    }, ["attacker", "target", "bonus"]),

    _schema("xp_status", "Devuelve el estado de XP del grupo (total, por PJ, últimos logs).", {}, []),

    _schema("xp_progress", "Añade XP por progreso narrativo (minor/standard/major).", {
        "kind": {"type": "string"},
        "reason": {"type": "string"}
    }, ["kind"]),

    _schema("xp_kill", "Añade XP por enemigo muerto usando su CR desde canon.enemies.", {
        "enemy": {"type": "string"}
    }, ["enemy"]),

    _schema("level_status", "Devuelve nivel estimado por XP (y XP necesaria al siguiente).", {}, []),
    _schema("level_check_up", "Comprueba si el grupo sube de nivel según XP y actualiza canon.party.level (y members) si procede.", {}, []),
    _schema("level_up_announce", "Si procede, anuncia la subida de nivel por XP: 'Subís a nivel X'.", {}, []),

    _schema("memory_get", "Lee una clave de memoria.", {"key": {"type": "string"}}, ["key"]),
    _schema("memory_set", "Guarda en memoria (texto).", {"key": {"type": "string"}, "value": {"type": "string"}}, ["key", "value"]),
]

# =========================================================
# Prompt / estilo DM
# =========================================================
def read_text_file(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8")
    except Exception as e:
        return f"[No se pudo leer {path}: {e}]"

DM_STYLE = read_text_file(str(ROOT / "dm" / "style.md"))

SYSTEM = f"""
Eres un Dungeon Master (DM) para D&D 5e.
Prioridad: inmersión + continuidad + reglas consistentes.

# =========================================================
# Planner / Narrator (pipeline: PLAN -> EXECUTE -> NARRATE)
# =========================================================
"""

PLANNER_SYSTEM = """
Eres un planificador táctico para D&D 5e. Tu única salida DEBE ser un JSON válido.
NO escribas texto fuera del JSON.

Objetivo: convertir la intención del usuario en una lista de acciones ejecutables por el motor.

Devuelve un objeto JSON con:
{
  "actions": [
    { ... },
    ...
  ]
}

Acciones soportadas (elige las que apliquen):
- attack:
  {"type":"attack","actor":"<name>","target":"<name>","count":N,
   "attack_type":"melee|ranged","weapon_hint":"bow|rapier|halberd|melee|ranged|... (opcional)",
   "roll_mode":"auto|normal|adv|dis (opcional)",
   "modes":["sharpshooter","gwm","sneak_attack",... (opcional)],
   "crit_range":20 (opcional)}
- spell_attack (por ahora cantrips tipo Fire Bolt):
  {"type":"spell_attack","actor":"<name>","target":"<name>","spell":"Fire Bolt",
   "roll_mode":"auto|normal|adv|dis (opcional)"}
- move:
  {"type":"move","actor":"<name>","feet":30}
- apply_condition / remove_condition:
  {"type":"apply_condition","target":"<name>","condition":"PRONE","rounds":1}
  {"type":"remove_condition","target":"<name>","condition":"PRONE"}
- damage / heal:
  {"type":"damage","target":"<name>","amount":7}
  {"type":"heal","target":"<name>","amount":6}

Reglas:
- Si el usuario describe múltiples acciones encadenadas (varios PJ y/o varios ataques), SIEMPRE genera actions[].
- Usa nombres EXACTOS de actores/targets del snapshot de CANON si aparecen.
- Si el usuario menciona Sharpshooter/GWM/Sneak Attack, añádelo en "modes".
- Si no estás seguro del target, elige el target más probable (no pidas aclaración).
"""

NARRATOR_SYSTEM = """
Eres un Director de Juego de D&D 5e. Vas a narrar el resultado de una resolución YA ejecutada por el motor.

REGLAS DURAS:
- NO inventes tiradas, críticos, totales de daño, ni números (prohibido usar dígitos 0-9).
- NO ofrezcas listas de opciones (“seguimos atacando / negociar / reagruparse…”) salvo que el usuario te pida explícitamente opciones.
- NO cambies quién hace qué: usa EXACTAMENTE los actores y objetivos del PLAN.
- Puedes describir el impacto de forma cualitativa (“impacta”, “falla”, “queda tambaleante”), pero sin cifras.

Tu salida:
- 1-3 párrafos de narración cinematográfica y coherente
- 1 pregunta final abierta: “¿Qué hacéis ahora?”
"""

ESTILO_RULES = """
{DM_STYLE}

REGLAS DE USO:
- Si necesitas estado (HP, CA, condiciones, escena, combate), usa herramientas (canon_get / target_status / combat_status / etc.).
- Para mecánicas repetitivas, usa herramientas (roll/check/attack/move/apply_condition).
- Después de una resolución importante, registra 1–2 líneas con log_event().
- No inventes números si importan: consulta el canon.
- Para tiradas de habilidad de un PJ, usa skill_check(actor, skill, dc, mode) para calcular bonus automáticamente.
- Si el usuario da múltiples acciones encadenadas (varios PJ o varias acciones seguidas), genera un Action Plan JSON y llama UNA sola vez a resolve_actions(plan_json). Evita parsers tipo execute_actions salvo legacy.

MODO ESCENA (CRÍTICO):
- dm_turn NO ES “solo combate”. Si no hay combate activo, SIEMPRE continúas la aventura en modo escena (exploración/social/decisiones).
- Nunca respondas “no hay combate activo” como bloqueo. Si no hay combate, describes la situación actual, propones 2–4 opciones y pides una decisión.
- Si falta contexto de escena, usa scene_status y canon_get para anclar continuidad antes de describir.

- REGLA CRÍTICA DE TRANSPARENCIA:
  Siempre que llames a una tool mecánica (roll/check/skill_check/attack/start_combat/combat_status/target_status),
  en tu siguiente mensaje DEBES incluir una sección "RESOLUCIÓN (mecánica)" y pegar el output de la tool literalmente
  (sin resumirlo). Después ya narras consecuencias.
- Cuando uses una herramienta (roll/check/skill_check/attack), pega el resultado literal en una sección ‘MECÁNICA:’ antes de narrar.
- Para cualquier conjuro (texto, alcance, duración, componentes, etc.), usa spell_info(name). Si dudas del nombre exacto, usa spell_search(query).
- Tras un hito importante, llama a xp_progress(...) y luego a level_check_up() para ver si subimos de nivel.
- Tras añadir XP, llama a level_up_announce() para anunciar la subida de nivel.

- MÓDULO (Fidelidad):
  - Si hay un módulo activo, ANTES de describir una escena nueva o resolver una decisión importante, llama a module_query()
    con palabras clave del momento (lugar, PNJ, objetivo, elemento raro) para anclar la narración al texto real.
  - Usa module_quote(chunk_id) SOLO cuando necesites texto literal (read-aloud) o verificación; evita soltar spoilers.
  - No menciones números de sala/encuentro ni claves internas; adapta la presentación al jugador.
  - Tras hitos grandes, actualiza module_set_progress(chapter, scene, flags) para mantener continuidad.

RECAP (OBLIGATORIO):
- Si has avanzado la escena o cambiado algo relevante, al final del turno llama a update_recap con un resumen de 2–6 líneas:
  lugar actual, qué acaba de pasar, NPCs relevantes, y qué decisiones quedan abiertas.
"""

# =========================================================
# Agente (Chat Completions + function calling)
# =========================================================
from typing import List, Dict, Any, Optional, Callable, Tuple
import json
import time
import inspect

@dataclass
class AgentState:
    # Historial para el LLM (Chat Completions messages)
    history: List[dict] = field(default_factory=list)

    # Estado persistente "ligero" (no depende de releer todo history)
    scene: Dict[str, Any] = field(default_factory=lambda: {
        "location": "",
        "current_scene": "",
        "recap": "",
        "open_threads": [],
    })

    # Flags generales de campaña (decisiones, puertas abiertas, PNJ hostiles, etc.)
    flags: Dict[str, Any] = field(default_factory=lambda: {"debug_mechanics": True})

    # Progreso del módulo (si se usa)
    module_progress: Dict[str, Any] = field(default_factory=lambda: {
        "module_id": "",
        "chapter": "",
        "scene": "",
        "flags": [],
    })

def _system_msg() -> dict:
    return {"role": "system", "content": SYSTEM}

def _user_msg(text: str) -> dict:
    return _user_item(text)

def _user_item(text: str) -> dict:
    return {"role": "user", "content": str(text or "")}

def _assistant_msg(text: str) -> dict:
    return {"role": "assistant", "content": str(text or "")}

def _tool_msg(tool_call_id: str, output: str) -> dict:
    return {"role": "tool", "tool_call_id": tool_call_id, "content": str(output)}

_ACTIVE_STATE: Optional["AgentState"] = None

def _get_state() -> Optional["AgentState"]:
    return _ACTIVE_STATE

def _schemas_to_chat_tools(tool_schemas: list) -> list:
    """
    Convierte tus TOOL_SCHEMAS (formato responses-ish) a formato Chat Completions:
      {"type":"function","function":{"name","description","parameters"}}
    """
    out = []
    for s in tool_schemas or []:
        if not isinstance(s, dict):
            continue
        if s.get("type") != "function":
            continue
        out.append({
            "type": "function",
            "function": {
                "name": s.get("name", ""),
                "description": s.get("description", ""),
                "parameters": (s.get("parameters") or {"type": "object", "properties": {}}),
            }
        })
    return out

CHAT_TOOLS = _schemas_to_chat_tools(TOOL_SCHEMAS)

def _classify_openai_error(e: Exception) -> str:
    status = getattr(e, "status_code", None)
    resp = getattr(e, "response", None)
    if status is None and resp is not None:
        status = getattr(resp, "status_code", None)
    msg = str(e)

    try:
        s = int(status) if status is not None else None
    except Exception:
        s = None

    if s == 401:
        return "AUTH"
    if s == 403:
        return "FORBIDDEN"
    if s == 404:
        return "MODEL_NOT_FOUND"
    if s == 429:
        if "insufficient_quota" in msg:
            return "INSUFFICIENT_QUOTA"
        return "RATE_LIMIT"
    if s is not None and 500 <= s < 600:
        return "SERVER"
    if "insufficient_quota" in msg:
        return "INSUFFICIENT_QUOTA"
    return "OTHER"

def _sleep_backoff(attempt: int) -> None:
    time.sleep(2 * (attempt + 1))

def _call_chat_with_retries(messages: List[dict], *, max_attempts: int = 4):
    last_err: Optional[Exception] = None

    for attempt in range(max_attempts):
        try:
            return client.chat.completions.create(
                model=_require_model(),
                messages=messages,
                tools=CHAT_TOOLS,
                tool_choice="auto",
                temperature=0.2,
            )
        except Exception as e:
            last_err = e
            kind = _classify_openai_error(e)

            if kind == "INSUFFICIENT_QUOTA":
                raise RuntimeError("OPENAI_ERROR:INSUFFICIENT_QUOTA | La API key no tiene cuota/billing activo.") from e
            if kind == "AUTH":
                raise RuntimeError("OPENAI_ERROR:AUTH | API key inválida/no configurada (401).") from e
            if kind == "MODEL_NOT_FOUND":
                raise RuntimeError(f"OPENAI_ERROR:MODEL_NOT_FOUND | Modelo no disponible.") from e

            if kind in {"RATE_LIMIT", "SERVER"} and attempt < (max_attempts - 1):
                _sleep_backoff(attempt)
                continue

            if attempt < (max_attempts - 1):
                _sleep_backoff(attempt)
                continue

    raise RuntimeError(f"OPENAI_ERROR:OTHER | {last_err}") from last_err

def _should_use_action_planner(user_text: str, canon: dict) -> bool:
    """
    Activa planner cuando hay combate / acciones encadenadas.
    Heurística: combate activo, o palabras clave de ataques/casts, o conteos (x2, 3 disparos...).
    """
    combat = canon.get("combat", {}) or {}
    if combat.get("active"):
        return True

    t = _norm(user_text or "")

    # palabras clave comunes
    kw = [
        "ataque", "ataca", "ataques", "disparo", "disparos", "shoot", "attacks",
        "castea", "lanza", "cast", "fire bolt", "sneak", "gwm", "sharpshooter",
        "asalto", "turno", "round", "acciones", "action surge"
    ]
    if any(k in t for k in kw):
        return True

    # patrones de conteo
    if re.search(r"\b\d+\s*(ataques|disparos|attacks|shots)\b", t):
        return True
    if re.search(r"\bx\s*\d+\b", t):  # x2, x3
        return True

    return False


def _canon_snapshot_for_planner(canon: dict, *, max_feats: int = 10) -> dict:
    """
    Snapshot compacto para que el planner use nombres y stats sin tragarse todo el canon.
    """
    snap = {"party": [], "enemies": [], "combat": canon.get("combat", {}) or {}}

    party = (canon.get("party", {}) or {}).get("members", []) or []
    for m in party:
        if not isinstance(m, dict):
            continue
        feats = m.get("features") or []
        feats_lite = [str(x) for x in feats[:max_feats]]
        snap["party"].append({
            "name": m.get("name", ""),
            "ac": m.get("ac", None),
            "hp": m.get("hp", None),
            "max_hp": m.get("max_hp", None),
            "prof_bonus": m.get("prof_bonus", None),
            "abilities": m.get("abilities", {}) or {},
            "features": feats_lite,
            "equipped": [it.get("name") for it in (m.get("inventory") or []) if isinstance(it, dict) and it.get("equipped")],
        })

    enemies = canon.get("enemies", {}) or {}
    for name, e in enemies.items():
        if not isinstance(e, dict):
            continue
        # runtime HP unificado
        hp_cur = e.get("hp_current", e.get("hp", None))
        hp_max = e.get("max_hp", e.get("hp_max", None))
        conds = _ensure_conditions(e)
        snap["enemies"].append({
            "name": name,
            "ac": e.get("ac", None),
            "hp": hp_cur,
            "max_hp": hp_max,
            "conditions": [c.get("name") for c in conds if isinstance(c, dict)],
            "stub": e.get("stub", None),
            "source": e.get("source", None),
        })

    return snap

def _enemy_status_summary(canon: dict, limit: int = 12) -> str:
    enemies = canon.get("enemies", {}) or {}
    lines = []
    for name, e in enemies.items():
        if not isinstance(e, dict):
            continue
        hp = e.get("hp_current", e.get("hp", None))
        mx = e.get("max_hp", e.get("hp_max", None))
        if hp is None or mx is None:
            continue
        try:
            hp_i = int(hp)
            mx_i = int(mx)
        except Exception:
            continue
        conds = [c.get("name") for c in _ensure_conditions(e) if isinstance(c, dict) and c.get("name")]
        cond_txt = f" ({', '.join(conds)})" if conds else ""
        lines.append(f"- {name}: {hp_i}/{mx_i} PV{cond_txt}")
        if len(lines) >= limit:
            break
    return "\n".join(lines).strip()

def _call_planner_json(user_text: str, canon: dict) -> Optional[dict]:
    """
    Llama al modelo para generar SOLO JSON {actions:[...]}.
    """
    snap = _canon_snapshot_for_planner(canon)
    messages = [
        {"role": "system", "content": PLANNER_SYSTEM.strip()},
        {"role": "user", "content": "USER_INPUT:\n" + (user_text or "").strip() + "\n\nCANON_SNAPSHOT:\n" + json.dumps(snap, ensure_ascii=False)}
    ]

    # Intento con response_format si la lib lo soporta; fallback si no.
    try:
        resp = client.chat.completions.create(
            model=_require_model(),
            messages=messages,
            temperature=0.1,
            response_format={"type": "json_object"},
        )
    except TypeError:
        resp = client.chat.completions.create(
            model=_require_model(),
            messages=messages,
            temperature=0.1,
        )

    content = (resp.choices[0].message.content or "").strip()
    if not content:
        return None

    try:
        plan = json.loads(content)
    except Exception:
        return None

    if not isinstance(plan, dict):
        return None
    actions = plan.get("actions")
    if not isinstance(actions, list) or not actions:
        return None

    return plan

_NARRATION_BAD_PATTERNS = [
    r"\bbonus\b",
    r"\bda[nñ]o\b",
    r"\bdc\b",
    r"\bts\b",
    r"\btiro\b",
    r"\btirada\b",
    r"\b1d\d+\b",
    r"\b\d+d\d+\b",
    r"\b\d+\b",
    r"\bcr[ií]tico\b",
    r"resoluci[oó]n",
    r"mec[aá]nica",
    r"interpretaci[oó]n",
]

def _narration_is_clean(text: str) -> bool:
    if not text:
        return False
    t = text.strip().lower()
    for pat in _NARRATION_BAD_PATTERNS:
        if re.search(pat, t, re.IGNORECASE):
            return False
    return True

def _call_narrator_text(user_text: str, canon_after: dict, plan: dict, resolution_text: str) -> str:
    """
    El narrador devuelve SOLO narración (sin mecánica).
    """
    snap = _canon_snapshot_for_planner(canon_after, max_feats=6)
    messages = [
        {"role": "system", "content": NARRATOR_SYSTEM.strip()},
        {"role": "user", "content":
            "Entrada del usuario:\n" + (user_text or "").strip()
            + "\n\nSnapshot post-resolución:\n" + json.dumps(snap, ensure_ascii=False)
            + "\n\nPlan ejecutado:\n" + json.dumps(plan, ensure_ascii=False)
            + "\n\nSalida del motor (no la repitas, solo úsala para narrar):\n" + resolution_text
            + "\n\nDevuelve SOLO narración + pregunta final."
        }
    ]

    # 1er intento
    resp = client.chat.completions.create(
        model=_require_model(),
        messages=messages,
        temperature=0.5,
    )
    txt = (resp.choices[0].message.content or "").strip()

    if _narration_is_clean(txt):
        return txt

    # 2º intento (más estricto)
    messages2 = messages + [{
        "role": "system",
        "content": "SEGUNDO INTENTO: recuerda que está PROHIBIDO incluir números (0-9), totales, críticos explícitos o menús de opciones. Solo narración cualitativa + una pregunta final abierta."
    }]
    resp2 = client.chat.completions.create(
        model=_require_model(),
        messages=messages2,
        temperature=0.2,
    )
    txt2 = (resp2.choices[0].message.content or "").strip()

    # Si aun así no cumple, devolvemos una versión saneada (último recurso)
    if _narration_is_clean(txt2):
        return txt2

    # Sanitizado mínimo: eliminar líneas con dígitos/listas
    lines = []
    for line in (txt2 or txt or "").splitlines():
        if re.search(r"\d", line):
            continue
        if re.search(r"seguir atacando|negociar|reagruparse|decidid", line, re.IGNORECASE):
            continue
        lines.append(line)
    cleaned = "\n".join([l for l in lines if l.strip()]).strip()

    if not cleaned:
        cleaned = "La escena se resuelve en un intercambio rápido y brutal. Los bandoleros vacilan, heridos, pero aún peligrosos.\n\n¿Qué hacéis ahora?"
    return cleaned


def _parse_tool_args(args_raw: Any) -> dict:
    if isinstance(args_raw, dict):
        return args_raw
    try:
        obj = json.loads(args_raw or "{}")
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}

def _filter_args_to_signature(fn: Callable[..., Any], args: dict) -> Tuple[dict, List[str]]:
    dropped: List[str] = []
    try:
        allowed = set(inspect.signature(fn).parameters.keys())
        filtered = {}
        for k, v in (args or {}).items():
            if k in allowed:
                filtered[k] = v
            else:
                dropped.append(k)
        return filtered, dropped
    except Exception:
        return args or {}, dropped

def _normalize_history_for_chat(history: list) -> list:
    """
    Convierte historial viejo (Responses API) a formato Chat Completions.

    Fix crítico:
    - Conserva tool_calls en mensajes del assistant.
    - Elimina mensajes tool "huérfanos" (sin tool_calls previos) para evitar:
      "messages with role 'tool' must be a response to a preceeding message with 'tool_calls'."
    """
    out = []
    pending_tool_ids = set()  # tool_call_ids esperados tras un assistant con tool_calls

    for m in history or []:
        if not isinstance(m, dict):
            continue

        role = m.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            continue

        content = m.get("content")

        # Caso Responses: content = [{"type":"input_text","text":"..."}]
        if isinstance(content, list) and content:
            parts = []
            for c in content:
                if isinstance(c, dict):
                    t = c.get("type")
                    if t in {"input_text", "text", "output_text"}:
                        parts.append(str(c.get("text") or ""))
            content = "\n".join([p for p in parts if p.strip()])

        # Assistant: conservar tool_calls si existen
        if role == "assistant":
            msg = {"role": "assistant", "content": str(content or "")}

            tool_calls = m.get("tool_calls") or []
            if isinstance(tool_calls, list) and tool_calls:
                # Normaliza ids de tool_calls
                ids = set()
                norm_calls = []
                for tc in tool_calls:
                    if hasattr(tc, "model_dump"):
                        tc = tc.model_dump()
                    if isinstance(tc, dict):
                        tc_id = tc.get("id") or tc.get("tool_call_id")
                        if tc_id:
                            ids.add(tc_id)
                        norm_calls.append(tc)
                msg["tool_calls"] = norm_calls
                pending_tool_ids = ids
            else:
                pending_tool_ids = set()

            out.append(msg)
            continue

        # Tool: Chat Completions requiere tool_call_id y debe corresponder a tool_calls previos
        if role == "tool":
            tool_call_id = (
                m.get("tool_call_id")
                or m.get("toolCallId")
                or m.get("call_id")
                or m.get("id")
            )
            if not tool_call_id:
                continue

            # Si no hay tool_calls previos o no coincide, es "huérfano" -> lo descartamos
            if not pending_tool_ids or tool_call_id not in pending_tool_ids:
                continue

            out.append({"role": "tool", "tool_call_id": tool_call_id, "content": str(content or "")})
            pending_tool_ids.discard(tool_call_id)
            continue

        # system/user: normal
        out.append({"role": role, "content": str(content or "")})

    return out

import re
from typing import Tuple

# =========================================================
# Gameplay patch: menos literal (movimiento implícito, NPCs, aliases)
# =========================================================

# Aliases típicos (AD&D/ediciones viejas / nombres comunes)
_SPELL_ALIASES = {
    # classic/legacy -> 5e canonical spell names
    "cure light wounds": "cure wounds",
    "cure serious wounds": "cure wounds",
    "cause light wounds": "inflict wounds",
    "cause serious wounds": "inflict wounds",

    # common casing / spacing variants (safe)
    "magic missile": "magic missile",
    "fireball": "fireball",
    "hold person": "hold person",
    "faerie fire": "faerie fire",
    "lightning bolt": "lightning bolt",
}

def _apply_spell_aliases(text: str) -> Tuple[str, list]:
    """Normaliza nombres de spells usando aliases y devuelve (texto_nuevo, cambios)."""
    if not text:
        return text, []
    changes = []
    out = text

    for src, dst in _SPELL_ALIASES.items():
        pattern = re.compile(rf"\b{re.escape(src)}\b", re.IGNORECASE)
        if pattern.search(out):
            out = pattern.sub(dst, out)
            changes.append((src, dst))

    return out, changes

_TERM_ALIASES = {
    # Feats / shorthand
    "sharp shoot": "sharpshooter",
    "sharp shooter": "sharpshooter",
    "sharpshoot": "sharpshooter",
    "ss": "sharpshooter",

    "great weapon mastery": "great weapon master",
    "great weapon master": "great weapon master",
    "gwm": "great weapon master",

    "pole arm mastery": "polearm master",
    "polearm mastery": "polearm master",
    "polearm master": "polearm master",
    "pam": "polearm master",

    "sentinel": "sentinel",

    # Combat terms
    "sneak attack": "sneak attack",
    "sa": "sneak attack",

    # Spanish -> canonical concept tokens
    "sigilo": "stealth",
    "melé": "melee",
    "mele": "melee",
}

# Nombres que el usuario suele usar como “actor” aunque no esté en party
# (si no existe, lo tratamos como PNJ relevante por defecto)
_DEFAULT_ASSUME_NPC_NAMES = {"eldrin"}

def _apply_term_aliases(text: str) -> Tuple[str, list]:
    """Normaliza términos (feats, shorthand) y devuelve (texto_nuevo, cambios)."""
    if not text:
        return text, []
    changes = []
    out = text

    for src, dst in _TERM_ALIASES.items():
        pattern = re.compile(rf"\b{re.escape(src)}\b", re.IGNORECASE)
        if pattern.search(out):
            out = pattern.sub(dst, out)
            changes.append((src, dst))

    return out, changes

def _inject_intent_hints(text: str) -> str:
    """
    Intents: asunciones razonables para que el DM no sea literal.
    - multiataques (Extra Attack)
    - primer asalto (Gloom Stalker)
    - bonus actions/reactions (GWM/PAM/Sentinel)
    """
    t = text or ""
    tl = t.lower()

    # Detectores básicos
    mentions_attack = any(k in tl for k in ["ataca", "ataque", "golpea", "dispara", "carga", "embiste", "strike", "shoot"])
    wants_melee = any(k in tl for k in ["melee", "cuerpo a cuerpo", "close"])
    mentions_first_round = any(k in tl for k in ["primer asalto", "primer turno", "first round", "round 1", "turno 1"])

    # (A) Ataque melé => moverse y atacar si es posible
    if mentions_attack and (wants_melee or "melee" in tl):
        t += (
            "\n\n[INTENCIÓN IMPLÍCITA: si el atacante no está ya en rango, asume que se mueve lo necesario "
            "para ponerse en rango (usando su movimiento) y luego ejecuta los ataques.]"
        )

    # (B) Rutina completa de ataques (Extra Attack) salvo que el usuario pida explícitamente 1 ataque
    # Nota: no podemos leer niveles/clase aquí; esto es una orden al modelo para que NO sea literal.
    if mentions_attack and not any(k in tl for k in ["solo un ataque", "un único ataque", "only one attack", "1 ataque"]):
        t += (
            "\n\n[INTENCIÓN IMPLÍCITA (MULTIATAQUE): si el actor tiene Extra Attack o ataques múltiples por nivel/rasgo, "
            "ejecuta la rutina completa del turno (p. ej., 2 ataques a nivel 5+ para guerrero/ranger/paladín) "
            "sin pedir confirmación por cada golpe. Solo para si hay una decisión táctica real.]"
        )

    # (C) Gloom Stalker: ataque extra en el primer asalto (si aplica)
    if ("gloom stalker" in tl) or ("dread ambusher" in tl) or mentions_first_round:
        t += (
            "\n\n[INTENCIÓN IMPLÍCITA (GLOOM STALKER): en el primer asalto, si el actor es Gloom Stalker "
            "aplica el ataque adicional de Dread Ambusher (y su daño extra) dentro de la misma secuencia.]"
        )

    # (D) Sharpshooter / GWM: asumir que quiere usar el feat (-5/+10) cuando menciona SS/GWM
    if "sharpshooter" in tl or "great weapon master" in tl:
        t += (
            "\n\n[INTENCIÓN IMPLÍCITA (SS/GWM): si el usuario menciona Sharpshooter o Great Weapon Master, "
            "asume que quiere aplicar -5/+10 cuando sea legal y razonable. Si no procede, dilo sin bloquear.]"
        )

    # (E) GWM bonus attack: al crit o al reducir a 0 PV, recordar bonus action de ataque
    if "great weapon master" in tl:
        t += (
            "\n\n[TRIGGER (GWM): si durante esta secuencia hay un crítico o reduces a 0 PV, recuerda ofrecer/ejecutar "
            "el ataque adicional como bonus action (si la bonus action está libre).]"
        )

    # (F) Polearm Master: reacción al entrar en alcance (normalmente 10 pies) + bonus action butt-end
    if "polearm master" in tl:
        t += (
            "\n\n[TRIGGER (PAM): si un enemigo entra en tu alcance, considera ataque de oportunidad especial (reacción). "
            "En tu turno, recuerda el ataque de bonus action con el extremo del arma si procede.]"
        )

    # (G) Sentinel: OA si intenta salir del alcance (y reduce speed a 0); también castiga disengage según 5e
    if "sentinel" in tl:
        t += (
            "\n\n[TRIGGER (SENTINEL): si el objetivo intenta salir de tu alcance, realiza OA si procede; "
            "al impactar, su velocidad pasa a 0. No bloquees por disengage si Sentinel aplica.]"
        )

    # (H) Sneak Attack: aplicarlo si hay condiciones
    if "sneak attack" in tl:
        t += (
            "\n\n[INTENCIÓN IMPLÍCITA (SNEAK ATTACK): aplica SA si se cumplen condiciones (ventaja, o aliado adyacente, etc.). "
            "Si no se cumplen, dilo y sugiere cómo habilitarlo.]"
        )

    # (I) Stealth: pedir tirada solo cuando haya riesgo real
    if "stealth" in tl or "sigilo" in tl:
        t += (
            "\n\n[INTENCIÓN IMPLÍCITA (STEALTH): si el usuario indica sigilo, asume movimiento cauteloso y pide "
            "Stealth solo cuando haya riesgo real de detección. No bloquees por ello.]"
        )

    # (J) PNJs no registrados (Eldrin, etc.)
    for name in _DEFAULT_ASSUME_NPC_NAMES:
        if re.search(rf"\b{re.escape(name)}\b", tl):
            t += (
                f"\n\n[NOTA: si '{name.title()}' no está en la party/canon, trátalo como PNJ aliado relevante "
                "en la escena (no bloquees). Si necesitas concretar stats/rol, pregunta 1 cosa concreta.]"
            )
            break
    
    # (K) Visibilidad de tiradas: si hay acción con azar, forzar uso de tools y output literal
    if mentions_attack or any(k in tl for k in ["tirada", "d20", "daño", "damage", "salvación", "saving throw", "check"]):
        t += (
            "\n\n[OBLIGATORIO (TIRADAS VISIBLES): resuelve mecánicas usando tools. "
            "Para ataques usa tool_attack (una vez por cada ataque). "
            "Para daño/otros usa tool_roll. Para checks usa tool_check o tool_skill_check. "
            "Después pega SIEMPRE el output literal en la sección RESOLUCIÓN (mecánica). "
            "No narres resultados sin tiradas visibles.]"
        )

    return t

def _preprocess_user_text(user_text: str) -> str:
    """
    Normaliza input del usuario para hacerlo más jugable:
    - aliases de spells
    - aliases de términos (feats/shorthand/idioma)
    - hints de intención (mover y atacar, PNJs no registrados, etc.)
    """
    t = (user_text or "").strip()

    # 1) aliases de spells
    t2, spell_changes = _apply_spell_aliases(t)

    # 2) aliases de términos (feats/shorthand/idioma)
    t3, term_changes = _apply_term_aliases(t2)

    changes = []
    changes.extend(spell_changes)
    changes.extend(term_changes)

    if changes:
        t3 += "\n\n[ALIASES APLICADOS: " + ", ".join([f"'{a}'→'{b}'" for a, b in changes]) + "]"

    # 3) intención implícita
    t3 = _inject_intent_hints(t3)

    return t3

def _expand_letter_groups(text: str) -> List[str]:
    """
    Convierte 'Cultista A/B/C' o 'Cultista A y B' en ['Cultista A','Cultista B','Cultista C'].
    Heurística pensada para IDs de combate.
    """
    out = []
    # Caso: "Base A/B/C"
    for m in re.finditer(r"\b([A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑáéíóúñ\-']+(?:\s+[A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑáéíóúñ\-']+)*?)\s+([A-Z](?:/[A-Z])+)\b", text):
        base = m.group(1).strip()
        letters = m.group(2).split("/")
        for L in letters:
            out.append(f"{base} {L.strip()}")

    # Caso: "Base A y B" / "Base A, B y C"
    for m in re.finditer(r"\b([A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑáéíóúñ\-']+(?:\s+[A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑáéíóúñ\-']+)*?)\s+([A-Z])(?:\s*,\s*([A-Z]))*(?:\s*(?:y|e)\s*([A-Z]))\b", text, re.IGNORECASE):
        base = m.group(1).strip()
        # recolecta letras presentes en grupos 2..4
        letters = [m.group(2), m.group(3), m.group(4)]
        for L in letters:
            if L:
                out.append(f"{base} {L.strip().upper()}")

    return out

def _extract_probable_enemy_names(text: str) -> List[str]:
    """
    Extrae nombres plausibles de enemigos del texto narrativo.
    Regla principal: etiquetas tipo 'X A', 'X B', etc + nombres en comillas.
    Evita falsos positivos lo mejor posible sin NLP pesado.
    """
    if not text:
        return []

    names: Set[str] = set()

    # 1) Expandir "A/B/C" y "A y B"
    for n in _expand_letter_groups(text):
        names.add(n)

    # 2) Capturar explícitos tipo "Cultista A", "Criatura Oscura B" (Base + letra)
    for m in re.finditer(r"\b([A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑáéíóúñ\-']+(?:\s+[A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑáéíóúñ\-']+)*?)\s+([A-Z])\b", text):
        base = m.group(1).strip()
        letter = m.group(2).strip().upper()
        # filtro: base con longitud mínima y no demasiado genérico
        if len(base) >= 4:
            names.add(f"{base} {letter}")

    # 3) Nombres entre comillas “Encapuchado Oscuro” / "Encapuchado Oscuro"
    for m in re.finditer(r"[\"“”‘’']([^\"“”‘’']{3,50})[\"“”‘’']", text):
        candidate = m.group(1).strip()
        # heurística: que tenga al menos una mayúscula inicial
        if re.search(r"\b[A-ZÁÉÍÓÚÑ]", candidate):
            names.add(candidate)

    # 4) Nombres “título” (dos o tres palabras capitalizadas) si aparecen tras palabras gatillo
    # Ej: "aparecen los Cultistas Sangrientos" / "un Encapuchado Oscuro"
    for m in re.finditer(r"\b(?:un|una|unos|unas|los|las)\s+([A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑáéíóúñ\-']+(?:\s+[A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑáéíóúñ\-']+){0,3})\b", text):
        cand = m.group(1).strip()
        # evita cosas típicas de localización (Greyhawk, Puerto de Greyhawk) usando filtros simples
        if any(x in cand.lower() for x in ["greyhawk", "puerto", "taberna"]):
            continue
        # evita frases demasiado largas
        if 3 <= len(cand) <= 40:
            names.add(cand)

    # limpiar y devolver lista estable
    cleaned = []
    for n in names:
        nn = " ".join(n.split()).strip()
        if nn:
            cleaned.append(nn)

    return sorted(cleaned)

def _auto_register_enemies_from_output(output_text: str) -> None:
    """
    Registra silenciosamente enemigos inferidos del texto narrativo.
    No escribe mensajes 'tool' en history (para no violar tool_calls).
    """
    if not output_text:
        return

    canon = canon_load()
    inferred = _extract_probable_enemy_names(output_text)

    if not inferred:
        return

    # registra solo los que NO sean party
    to_add = []
    for name in inferred:
        if _find_party_member(canon, name):
            continue
        to_add.append(name)

    if not to_add:
        return

    canon.setdefault("enemies", {})
    enemies = canon["enemies"]

    changed = False
    for name in to_add:
        if name not in enemies:
            enemies[name] = {
                "name": name,
                "stub": True,
                "ac": 10,
                "hp": 999,
                "hp_max": 999,
                "conditions": [],
            }
            changed = True
        else:
            # asegurar mínimos
            e = enemies[name]
            if "conditions" not in e:
                e["conditions"] = []
                changed = True
            e.setdefault("name", name)
            e.setdefault("ac", 10)
            e.setdefault("hp", 999)
            e.setdefault("hp_max", e.get("hp", 999))

    if changed:
        canon_save(canon)

def run_agent_turn(user_text: str, state: AgentState) -> str:
    """
    Ejecuta un turno del agente.
    A3: activa un puntero global al AgentState actual para que las tools puedan
    persistir scene/location/flags/module_progress en la sesión (no solo en history).
    """
    global _ACTIVE_STATE
    _ACTIVE_STATE = state

    # Contexto de turno para RESOLUCIÓN (mecánica) / LOG
    ctx = TurnContext()

    try:
        state.history = _normalize_history_for_chat(state.history)

        if not state.history or state.history[0].get("role") != "system":
            state.history.insert(0, _system_msg())

        user_text = _preprocess_user_text(user_text)
        state.history.append(_user_msg(user_text))

        # =========================================================
        # PIPELINE: PLAN -> EXECUTE -> NARRATE (para multi-acciones / combate)
        # =========================================================
        canon_now = canon_load()
        if _should_use_action_planner(user_text, canon_now):

            # 1) PLAN (con reintento simple)
            plan = _call_planner_json(user_text, canon_now)
            if not plan:
                plan = _call_planner_json(
                    user_text + "\n\nIMPORTANTE: Devuelve SOLO JSON válido con {\"actions\":[...]} y nombres exactos.",
                    canon_now
                )

            if not plan:
                output = format_turn_output(
                    ctx=ctx,
                    interpretation="No se pudo construir un plan ejecutable.",
                    narration=(
                        "No pude construir un Action Plan JSON válido para ejecutar el motor.\n"
                        "Repite tus acciones en formato breve: Actor -> acción -> objetivo (y conteos), por ejemplo:\n"
                        "“Myrmyr: ataque x3 a Bandit A (sharpshooter). Kaelen: ataque x2 a Bandit A (sneak_attack).”"
                    ),
                    options=[],
                )
                state.history.append(_assistant_msg(output))
                return output

            plan_str = json.dumps(plan, ensure_ascii=False)

            # 2) EXECUTE (motor)
            resolution = tool_resolve_actions(plan_str)

            # Registrar SIEMPRE la mecánica literal del motor
            try:
                ctx.mech(resolution)
            except Exception:
                pass

            # 3) POST STATE
            canon_after = canon_load()

            # 4) NARRATE (solo texto)
            narr = _call_narrator_text(user_text, canon_after, plan, resolution)

            BAD_PATTERNS = [
                r"\b\d+\b",
                r"puntos de daño|daño total|total de",
                r"\bcr[ií]tico\b|\bcrit\b",
                r"\bdc\b|\btirada\b|\bprueba\b|\bsalvaci[oó]n\b",
                r"decidid vuestra acci[oó]n",
                r"seguir atacando|intimidar|negociar|reagruparse|preparar una defensa|usar alguna habilidad",
                r"\bopciones\b",
                r"\bdecidid\b",
            ]

            def _narr_ok(txt: str) -> bool:
                if not txt or not txt.strip():
                    return False
                low = txt.strip().lower()
                for pat in BAD_PATTERNS:
                    if re.search(pat, low):
                        return False
                return True

            if not _narr_ok(narr):
                narr = (
                    "Los eventos se desenvuelven rápidamente. Los enemigos reaccionan a vuestros movimientos, "
                    "el combate se intensifica.\n\n¿Qué hacéis ahora?"
                )

            # ✅ SIEMPRE: salida con formato fijo (incluye RESOLUCIÓN (mecánica))
            output = format_turn_output(
                ctx=ctx,
                interpretation="El motor ejecutó el plan y aplicó los cambios de estado correspondientes.",
                narration=(narr or "").strip() or "(sin narración)",
                options=[
                    "AVANZA: indica tu siguiente acción (intención + prioridades + 'si... entonces...')",
                    "ESTADO",
                    "RESET",
                ],
            )

            # Persistir en history para continuidad
            state.history.append(_assistant_msg(output))

            # (opcional) log a session.md para auditoría
            try:
                tool_log_event("=== TURN ===\nPLAN:\n" + plan_str + "\n\nRESOLUTION:\n" + resolution + "\n\n")
            except Exception:
                pass

            return output

        # =========================================================
        # MODO NORMAL (sin planner): chat directo con function calling
        # =========================================================
        last_narration = ""
        while True:
            response = _call_chat_with_retries(state.history)

            # Procesa respuesta
            msg = response.choices[0].message
            content = (msg.content or "").strip()
            tool_calls = getattr(msg, "tool_calls", None) or []

            if content:
                last_narration = content
                state.history.append(_assistant_msg(content))

            if not tool_calls:
                # ✅ SIEMPRE: salida con formato fijo (aunque no haya tiradas)
                output = format_turn_output(
                    ctx=ctx,
                    interpretation="Turno resuelto.",
                    narration=last_narration or "(sin salida de texto)",
                    options=[
                        "AVANZA: siguiente acción",
                        "ESTADO",
                    ],
                )
                # Nota: no añadimos output al history para no “doblar” el mensaje del assistant.
                # Si quieres consistencia total en history, comenta el append anterior y añade solo este.
                return output

            # Ejecuta tool calls
            tool_results = []
            for tc in tool_calls:
                fn_name = tc.function.name
                args_raw = tc.function.arguments or "{}"
                args = _parse_tool_args(args_raw)

                fn = TOOLS.get(fn_name)
                if not fn:
                    tool_output = f"ERROR: función '{fn_name}' no existe."
                    tool_results.append((tc.id, tool_output))
                    continue

                args_filtered, dropped = _filter_args_to_signature(fn, args)
                try:
                    tool_output = fn(**args_filtered)
                    if dropped:
                        tool_output = f"{tool_output}\n(WARN: args ignorados por no estar en la firma: {dropped})"
                except Exception as e:
                    tool_output = f"ERROR: {e}"

                tool_results.append((tc.id, tool_output))

                # Log mecánica si procede
                if fn_name in {"roll", "check", "skill_check", "attack", "damage", "heal", "apply_condition", "remove_condition"}:
                    try:
                        ctx.mech(f"{fn_name}(...)\n{tool_output}")
                    except Exception:
                        pass

            # Añade tool results al history
            for tool_id, tool_output in tool_results:
                state.history.append(_tool_msg(tool_id, str(tool_output)))

            # Auto-registrar enemigos del último assistant message (si hay)
            if last_narration:
                _auto_register_enemies_from_output(last_narration)

            canon_now = canon_load()
            if canon_now.get("combat", {}).get("active"):
                # Sugerencia: usa combat_status, next_turn, etc.
                pass

    except RuntimeError as e:
        error_msg = str(e)
        if "OPENAI_ERROR" in error_msg:
            output = format_turn_output(
                ctx=ctx,
                interpretation="Error de API.",
                narration=f"⚠️ Error de API OpenAI:\n{error_msg}",
                options=[],
            )
            return output
        raise

    finally:
        _ACTIVE_STATE = None
        # No llames a ctx.dump() si tu TurnContext no lo implementa.
        # Si existe, mantenlo; si no, quítalo para evitar ruido.
        try:
            ctx.dump()
        except Exception:
            pass

def run_turn_wot(session_id: str, user_text: str, session_state: dict) -> str:
    ctx = TurnContext()

    # 1) Tu lógica decide si hay tiradas o no.
    # EJEMPLO: si el jugador dice "registro la sala", haces check percepción CD 15 con bono +6
    if "registro" in user_text.lower():
        ok, total, txt, raw = skill_check(bonus=6, dc=15, mode="normal")
        ctx.mech(f"Percepción (CD 15, bono +6): {txt}")
        if ok:
            narration = "Registras cada rincón con calma operativa. Algo no encaja: una corriente mínima de aire delata una junta oculta."
            interpretation = "Superas la CD: encuentras un indicio útil sin alertar a nadie."
            options = ["Inspeccionar la junta con herramientas", "Marcar el punto y avanzar", "Llamar al Warder para cubrir el pasillo"]
            ctx.log_event("Hallazgo: posible puerta/panel oculto detectado.")
        else:
            narration = "Buscas a conciencia, pero el polvo y la geometría del lugar te engañan: no detectas nada concluyente."
            interpretation = "Fallas la CD: no obtienes información adicional."
            options = ["Repetir con más tiempo (riesgo de retraso)", "Cambiar el ángulo de luz / antorcha", "Avanzar con cautela"]
    else:
        # Turno sin tiradas
        narration = "Anotas el plan y ajustas la formación. Todo queda listo para ejecutar."
        interpretation = "No hay resolución aleatoria en este paso."
        options = ["AVANZA: entramos", "AVANZA: hablamos con el guardia", "ESTADO"]

    # 2) SIEMPRE devuelve el formato fijo con RESOLUCIÓN (mecánica)
    return format_turn_output(
        ctx=ctx,
        interpretation=interpretation,
        narration=narration,
        options=options,
    )

# =========================================================
# CLI (robusto + modo local)
# =========================================================
if __name__ == "__main__":
    state = AgentState()
    print("DM Agent listo. Escribe 'exit' para salir. Usa '/help' para comandos locales.\n")

    def _local_help() -> str:
        return (
            "Comandos locales (sin LLM):\n"
            "  /help\n"
            "  /scene                 -> scene_status()\n"
            "  /combat                -> combat_status()\n"
            "  /status <nombre>       -> target_status(nombre)\n"
            "  /roll <expr>           -> roll(expr)\n"
            "  /check <skill> <dc> <bonus> [normal|adv|dis]\n"
            "  /skill <actor> <skill> <dc> [normal|adv|dis] [extra_bonus]\n"
            "  /spell <nombre>        -> spell_info(nombre)\n"
            "  /spells <query> [n]    -> spell_search(query, n)\n"
            "  /module_load            -> module_load()\n"
            "  /mfind <query>           -> module_query(query, 6)\n"
            "  /mquote <chunk_id>       -> module_quote(chunk_id)\n"
            "  /exit\n"
        )

    while True:
        try:
            text = input("Tú: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSaliendo.\n")
            break

        t = text.lower().strip()
        if t in {"exit", "quit", "salir", ":q", "/exit"}:
            print("Saliendo.\n")
            break

        # ---- modo local (sin modelo) ----
        if t in {"/help", "help"}:
            print(_local_help())
            continue

        if t in {"/scene"}:
            print("\nAgente:\n", tool_scene_status(), "\n")
            continue

        if t in {"/combat"}:
            print("\nAgente:\n", tool_combat_status(), "\n")
            continue

        if t.startswith("/status "):
            name = text.split(" ", 1)[1].strip()
            print("\nAgente:\n", tool_target_status(name), "\n")
            continue

        if t.startswith("/roll "):
            expr = text.split(" ", 1)[1].strip()
            print("\nAgente:\n", tool_roll(expr), "\n")
            continue

        if t.startswith("/spell "):
            name = text.split(" ", 1)[1].strip()
            print("\nAgente:\n", tool_spell_info(name), "\n")
            continue

        if t.startswith("/spells "):
            parts = text.split()
            query = parts[1] if len(parts) >= 2 else ""
            limit = int(parts[2]) if len(parts) >= 3 else 10
            print("\nAgente:\n", tool_spell_search(query, limit), "\n")
            continue

        if t.startswith("/check "):
            parts = text.split()
            # /check stealth 15 5 adv
            if len(parts) < 4:
                print("\nAgente:\n Uso: /check <skill> <dc> <bonus> [normal|adv|dis]\n")
                continue
            skill = parts[1]
            dc = int(parts[2])
            bonus = int(parts[3])
            mode = parts[4] if len(parts) >= 5 else "normal"
            print("\nAgente:\n", tool_check(skill, dc, bonus, mode), "\n")
            continue

        if t.startswith("/skill "):
            parts = text.split()
            # /skill "Myrmyr Lash" stealth 15 adv 0  (sin comillas: usa underscore o escribe sin espacios)
            if len(parts) < 4:
                print("\nAgente:\n Uso: /skill <actor> <skill> <dc> [normal|adv|dis] [extra_bonus]\n")
                continue
            actor = parts[1]
            skill = parts[2]
            dc = int(parts[3])
            mode = parts[4] if len(parts) >= 5 else "normal"
            extra = int(parts[5]) if len(parts) >= 6 else 0
            print("\nAgente:\n", tool_skill_check(actor, skill, dc, mode, extra), "\n")
            continue
        
        if t in {"/module_load"}:
            print("\nAgente:\n", tool_module_load(), "\n")
            continue

        if t.startswith("/mfind "):
            q = text.split(" ", 1)[1].strip()
            print("\nAgente:\n", tool_module_query(q, 6), "\n")
            continue

        if t.startswith("/mquote "):
            cid = text.split(" ", 1)[1].strip()
            print("\nAgente:\n", tool_module_quote(cid), "\n")
            continue

        # ---- modo normal (con LLM) ----
        out = run_agent_turn(text, state)
        print("\nAgente:\n", out, "\n")
