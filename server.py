# server.py
import os
import re
import json
from pathlib import Path
from typing import Dict, Any, Optional, Literal, Tuple

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

# Endpoints mecánicos WoT
from wot_dice import roll_expr, skill_check as _skill_check, attack_roll as _attack_roll

# Formato de salida
from wot_output import TurnContext, format_turn_output

# -----------------------------
# Paths estables + .env
# -----------------------------
ROOT = Path(__file__).resolve().parent
load_dotenv(dotenv_path=ROOT / ".env", override=False)

SESSIONS_DIR = Path(os.getenv("SESSIONS_DIR") or (ROOT / "data" / "sessions"))
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# BOOT logs (solo cosas ya definidas)
# -----------------------------
print("[BOOT] starting server.py")
print("[BOOT] python:", __import__("sys").version)
print("[BOOT] PORT env:", os.getenv("PORT"))
print("[BOOT] DM token set:", bool((os.getenv("DM_API_TOKEN") or "").strip()))
print("[BOOT] sessions dir:", str(SESSIONS_DIR))

# -----------------------------
# PUBLIC paths (exactos) + prefijos (robustos con root_path)
# -----------------------------
PUBLIC_PATHS = {"/", "/favicon.ico"}
PUBLIC_PREFIXES = ("/health", "/config", "/docs", "/openapi.json", "/redoc", "/routes")

def _get_api_token() -> str:
    return (os.getenv("DM_API_TOKEN") or "").strip()

def _auth_ok(request: Request, token: str) -> bool:
    # Authorization: Bearer <token>  OR  X-API-Key: <token>
    auth = request.headers.get("authorization") or ""
    xkey = request.headers.get("x-api-key") or ""
    if auth.lower().startswith("bearer "):
        return auth.split(" ", 1)[1].strip() == token
    if xkey:
        return xkey.strip() == token
    return False

# -----------------------------
# Import tolerante del agente
# -----------------------------
AGENT_IMPORT_ERROR: Optional[str] = None
AgentState = None
run_agent_turn = None

try:
    from agent import AgentState as _AgentState, run_agent_turn as _run_agent_turn
    AgentState = _AgentState
    run_agent_turn = _run_agent_turn
except Exception as e:
    AGENT_IMPORT_ERROR = str(e)

print("[BOOT] agent import error:", AGENT_IMPORT_ERROR)
print("[BOOT] agent available:", bool(AgentState and run_agent_turn))

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="DM Agent API", version="1.2.2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # PROD: restringe a tu dominio
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    path = request.url.path or "/"

    # Soporta root_path (si FastAPI/Proxy lo usa)
    root_path = request.scope.get("root_path", "") or ""
    if root_path and path.startswith(root_path):
        path_no_root = path[len(root_path):] or "/"
    else:
        path_no_root = path

    # Normaliza barra final
    if path_no_root != "/" and path_no_root.endswith("/"):
        path_no_root = path_no_root[:-1]

    # Allowlist
    if path_no_root in PUBLIC_PATHS or path_no_root.startswith(PUBLIC_PREFIXES):
        return await call_next(request)

    token = (os.getenv("DM_API_TOKEN") or "").strip()
    if not token:
        return JSONResponse(status_code=503, content={"detail": "Servidor sin DM_API_TOKEN configurado"})

    auth = request.headers.get("authorization") or ""
    xkey = request.headers.get("x-api-key") or ""

    ok = False
    if auth.lower().startswith("bearer "):
        ok = auth.split(" ", 1)[1].strip() == token
    elif xkey:
        ok = xkey.strip() == token

    if not ok:
        return JSONResponse(status_code=401, content={"detail": "Unauthorized"})

    return await call_next(request)

# -----------------------------
# Modelos
# -----------------------------
Mode = Literal["normal", "adv", "dis"]

class TurnRequest(BaseModel):
    session_id: str
    text: str

class TurnResponse(BaseModel):
    session_id: str
    output: str

class RollReq(BaseModel):
    expr: str
    session_id: Optional[str] = None
    label: Optional[str] = None

class SkillCheckReq(BaseModel):
    bonus: int
    dc: int
    mode: str = Field(default="normal", pattern="^(normal|adv|dis)$")
    session_id: Optional[str] = None
    label: Optional[str] = None

class AttackReq(BaseModel):
    attack_bonus: int
    target_ac: int
    mode: str = Field(default="normal", pattern="^(normal|adv|dis)$")
    session_id: Optional[str] = None
    label: Optional[str] = None
    damage_expr: Optional[str] = None

# -----------------------------
# Helpers sesión
# -----------------------------
def _safe_session_id(session_id: str) -> str:
    safe = "".join(ch for ch in (session_id or "") if ch.isalnum() or ch in ("-", "_")).strip()
    return safe or "default"

def _session_path(session_id: str) -> Path:
    return SESSIONS_DIR / f"{_safe_session_id(session_id)}.json"

def _events_path(session_id: str) -> Path:
    return SESSIONS_DIR / f"{_safe_session_id(session_id)}.events.jsonl"

def _append_session_event(session_id: str, kind: str, payload: Dict[str, Any]) -> None:
    """
    Log ligero en el JSON de sesión: {"events":[...]}
    Best-effort: no bloquea.
    """
    try:
        p = _session_path(session_id)
        data = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        if not isinstance(data, dict):
            data = {}
        events = data.get("events", [])
        if not isinstance(events, list):
            events = []
        events.append({"kind": kind, "payload": payload})
        data["events"] = events
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        return

def load_state(session_id: str):
    if not AgentState:
        return None

    p = _session_path(session_id)
    if not p.exists():
        return AgentState()  # type: ignore

    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            data = {}
    except Exception:
        data = {}

    st = AgentState()  # type: ignore

    hist = data.get("history", [])
    if isinstance(hist, list):
        st.history = hist

    scene = data.get("scene", None)
    if isinstance(scene, dict) and hasattr(st, "scene") and isinstance(getattr(st, "scene", None), dict):
        st.scene.update(scene)  # type: ignore

    flags = data.get("flags", None)
    if isinstance(flags, dict) and hasattr(st, "flags") and isinstance(getattr(st, "flags", None), dict):
        st.flags.update(flags)  # type: ignore

    module_progress = data.get("module_progress", None)
    if isinstance(module_progress, dict) and hasattr(st, "module_progress") and isinstance(getattr(st, "module_progress", None), dict):
        st.module_progress.update(module_progress)  # type: ignore

    return st

def save_state(session_id: str, state) -> None:
    p = _session_path(session_id)
    payload = {
        "history": getattr(state, "history", []),
        "scene": getattr(state, "scene", {}),
        "flags": getattr(state, "flags", {}),
        "module_progress": getattr(state, "module_progress", {}),
        # nota: events se escriben vía _append_session_event, no aquí
    }
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(p)

# -----------------------------
# Runtime checks
# -----------------------------
def _check_runtime_ready() -> Dict[str, Any]:
    if AGENT_IMPORT_ERROR:
        return {"ok": False, "reason": "agent_import_error", "detail": AGENT_IMPORT_ERROR}

    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    model = (os.getenv("OPENAI_MODEL") or "").strip()

    if not api_key:
        return {"ok": False, "reason": "missing_OPENAI_API_KEY", "detail": "Falta OPENAI_API_KEY en entorno/.env"}
    if not model:
        return {"ok": False, "reason": "missing_OPENAI_MODEL", "detail": "Falta OPENAI_MODEL en entorno/.env"}

    return {"ok": True, "model": model}

# -----------------------------
# Endpoints públicos
# -----------------------------
@app.get("/")
def root() -> Dict[str, Any]:
    return {"ok": True, "service": "Althalus DM Agent API"}

@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True, "ready": _check_runtime_ready()}

@app.get("/config")
def config() -> Dict[str, Any]:
    ready = _check_runtime_ready()
    return {
        "ready": ready,
        "openai_api_key_present": bool((os.getenv("OPENAI_API_KEY") or "").strip()),
        "openai_model": (os.getenv("OPENAI_MODEL") or "").strip() or None,
        "dm_api_token_configured": bool(_get_api_token()),
        "sessions_dir": str(SESSIONS_DIR),
        "root": str(ROOT),
    }

@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)

@app.get("/routes")
def routes():
    return sorted(
        [{"path": r.path, "methods": sorted(list(getattr(r, "methods", []) or []))}
         for r in app.routes],
        key=lambda x: x["path"]
    )

# =============================
# Endpoints mecánicos WoT
# =============================
@app.post("/roll", operation_id="roll")
def roll_endpoint(req: RollReq):
    if not req.expr or not req.expr.strip():
        raise HTTPException(status_code=400, detail="expr vacío")

    r = roll_expr(req.expr)
    text = f"{req.label + ': ' if req.label else ''}{r.detail}"

    if req.session_id:
        _append_session_event(req.session_id, "roll", {
            "expr": req.expr, "detail": r.detail, "total": r.total, "label": req.label
        })

    return {"total": r.total, "detail": r.detail, "text": text, "raw": r.raw}

@app.post("/skill_check", operation_id="skill_check")
def skill_check_endpoint(req: SkillCheckReq):
    res = _skill_check(bonus=req.bonus, dc=req.dc, mode=req.mode)

    if isinstance(res, tuple) and len(res) >= 4:
        success = bool(res[0]); total = int(res[1]); txt = str(res[2]); raw = res[3]
    else:
        success, total, txt, raw = False, 0, str(res), {}

    text = (req.label + ": " if req.label else "") + txt

    if req.session_id:
        _append_session_event(req.session_id, "skill_check", {
            "bonus": req.bonus, "dc": req.dc, "mode": req.mode,
            "success": success, "total": total, "label": req.label
        })

    return {"success": success, "total": total, "dc": req.dc, "text": text, "raw": raw}

@app.post("/attack", operation_id="attack")
def attack_endpoint(req: AttackReq):
    hit, crit, total, txt, raw = _attack_roll(
        attack_bonus=req.attack_bonus,
        target_ac=req.target_ac,
        mode=req.mode,
    )

    out: Dict[str, Any] = {
        "hit": hit,
        "crit": crit,
        "total": total,
        "text": (req.label + ": " if req.label else "") + txt,
        "raw": raw,
    }

    if hit and req.damage_expr:
        dmg = roll_expr(req.damage_expr)
        out["damage_total"] = dmg.total
        out["damage_detail"] = dmg.detail
        out["damage_raw"] = dmg.raw

    if req.session_id:
        _append_session_event(req.session_id, "attack", {
            "attack_bonus": req.attack_bonus, "target_ac": req.target_ac,
            "mode": req.mode, "hit": hit, "crit": crit, "total": total,
            "damage_expr": req.damage_expr, "label": req.label,
        })

    return out

# -----------------------------
# Turno DM (operation_id: dm_turn)
# -----------------------------
@app.post("/turn", response_model=TurnResponse, operation_id="dm_turn")
def turn(req: TurnRequest):
    if AGENT_IMPORT_ERROR or not AgentState or not run_agent_turn:
        raise HTTPException(status_code=503, detail=f"Agente no disponible: {AGENT_IMPORT_ERROR}")

    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="text vacío")

    ready = _check_runtime_ready()
    if not ready.get("ok"):
        raise HTTPException(status_code=503, detail=ready)

    state = load_state(req.session_id)
    if state is None:
        raise HTTPException(status_code=503, detail="Agente no inicializado")

    try:
        raw_output = run_agent_turn(req.text, state)
    except RuntimeError as e:
        msg = str(e)
        if "OPENAI_MODEL" in msg or "OPENAI_API_KEY" in msg:
            raise HTTPException(status_code=503, detail=msg)
        raise HTTPException(status_code=500, detail=f"RuntimeError: {msg}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno: {e}")

    try:
        save_state(req.session_id, state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error guardando sesión: {e}")

    text = str(raw_output or "").strip()

    def _has_resolution_block(t: str) -> bool:
        up = t.upper()
        return ("RESOLUCIÓN" in up) and ("MECÁNICA" in up)

    MECH_PAT = re.compile(
        r"("
        r"\bd20\b|"
        r"\bcd\b|"
        r"\btirada\b|"
        r"\bprueba\b|"
        r"\bdañ(?:o|os)\b|"
        r"\bimpacta\b|\bfalla\b|"
        r"\bcr[ií]tico\b|\bcrit\b|"
        r"\bsalvaci[oó]n\b|\bts\b|"
        r"\biniciativa\b|"
        r"\bconcentraci[oó]n\b|"
        r"\bcondici[oó]n\b|"
        r"\bpv\b|\bhp\b|"
        r"\bprone\b|\bparalyzed\b|\bparalizado\b|\bderribado\b|"
        r"\b\d+d\d+\b"
        r")",
        re.IGNORECASE,
    )

    def _split_into_chunks(t: str) -> list[str]:
        if "\n" in t:
            return [ln.strip() for ln in t.splitlines() if ln.strip()]
        parts = re.split(r"(?<=[\.\!\?\u00BF\u00A1])\s+", t.strip())
        return [p.strip() for p in parts if p and p.strip()]

    def _extract_mechanics_and_narration(t: str) -> Tuple[list[str], str]:
        mechanics_lines: list[str] = []
        narration_chunks: list[str] = []

        if "```" in t:
            segments = re.split(r"(```.*?```)", t, flags=re.DOTALL)
            for seg in segments:
                if not seg or not seg.strip():
                    continue
                if seg.startswith("```") and seg.endswith("```"):
                    (mechanics_lines if MECH_PAT.search(seg) else narration_chunks).append(seg.strip())
                else:
                    for ch in _split_into_chunks(seg):
                        (mechanics_lines if MECH_PAT.search(ch) else narration_chunks).append(ch)
        else:
            for ch in _split_into_chunks(t):
                (mechanics_lines if MECH_PAT.search(ch) else narration_chunks).append(ch)

        return mechanics_lines, "\n".join(narration_chunks).strip()

    if _has_resolution_block(text):
        return TurnResponse(session_id=req.session_id, output=text)

    ctx = TurnContext()
    mech_lines, narration_text = _extract_mechanics_and_narration(text)

    if mech_lines:
        for ln in mech_lines[:80]:
            try:
                ctx.mech(ln)
            except Exception:
                pass
        try:
            ctx.log_event("WARN: salida original sin bloque 'RESOLUCIÓN (mecánica)'; mecánica extraída automáticamente.")
        except Exception:
            pass
    else:
        narration_text = narration_text or text
        try:
            ctx.log_event("Salida envuelta en formato fijo (sin mecánica visible).")
        except Exception:
            pass

    wrapped = format_turn_output(
        ctx=ctx,
        interpretation=(
            "Salida envuelta en formato fijo. "
            "Si hay discrepancias, ajusta el agente para emitir 'RESOLUCIÓN (mecánica)' de origen."
        ),
        narration=narration_text if narration_text else "(sin salida de texto)",
        options=[],
    )
    return TurnResponse(session_id=req.session_id, output=wrapped)

# -----------------------------
# Reset sesión
# -----------------------------
@app.post("/session/reset/{session_id}", operation_id="reset_session")
def session_reset_path(session_id: str) -> Dict[str, Any]:
    if not session_id or not session_id.strip():
        raise HTTPException(status_code=422, detail="session_id requerido")

    p = _session_path(session_id)
    if p.exists():
        p.unlink()

    ep = _events_path(session_id)
    if ep.exists():
        try:
            ep.unlink()
        except Exception:
            pass

    return {"ok": True, "session_id": session_id}

# -----------------------------
# Dump sesión
# -----------------------------
@app.get("/session/{session_id}", operation_id="get_session")
def session_dump(session_id: str) -> Dict[str, Any]:
    p = _session_path(session_id)
    if not p.exists():
        return {"ok": True, "session_id": session_id, "history": [], "scene": {}, "flags": {}, "module_progress": {}, "events": []}

    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        data = {}

    return {
        "ok": True,
        "session_id": session_id,
        "history": data.get("history", []) if isinstance(data.get("history", []), list) else [],
        "scene": data.get("scene", {}) if isinstance(data.get("scene", {}), dict) else {},
        "flags": data.get("flags", {}) if isinstance(data.get("flags", {}), dict) else {},
        "module_progress": data.get("module_progress", {}) if isinstance(data.get("module_progress", {}), dict) else {},
        "events": data.get("events", []) if isinstance(data.get("events", []), list) else [],
    }