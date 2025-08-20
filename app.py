# app.py
import io, os, glob, time, pathlib, datetime, json
from typing import Dict, List, Tuple, Optional

import uvicorn
import numpy as np
import requests
from fastapi import FastAPI, UploadFile, File, Form, Query
from pydantic import BaseModel
from PIL import Image
import face_recognition

from dotenv import load_dotenv
load_dotenv()

# SQLAlchemy + estáticos
from sqlalchemy import create_engine, text
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# ===================== ENV =====================
THRESHOLD = float(os.environ.get("THRESHOLD", "0.55"))  # 0.4 estricto, 0.6 tolerante
KNOWN_FACES_DIR = os.environ.get("KNOWN_FACES_DIR", "./known_faces")
CROPS_DIR = os.environ.get("CROPS_DIR", "./events")
SAVE_MATCH_CROP = os.environ.get("SAVE_MATCH_CROP", "true").lower() == "true"

DB_URL = os.environ.get("MYSQL_URL")  # p.ej. mysql+pymysql://user:pass@host:3306/dbname
SQL_ECHO = os.environ.get("SQL_ECHO", "false").lower() == "true"
CORS_ORIGINS = [o.strip() for o in os.environ.get("CORS_ORIGINS", "*").split(",")]

engine = create_engine(
    DB_URL, pool_pre_ping=True, pool_recycle=1800, future=True, echo=SQL_ECHO
) if DB_URL else None

app = FastAPI(title="FaceAuth API", version="1.2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===================== Utils / Log =====================
def console_log(data, label="DEBUG"):
    print(f"[{label}] {json.dumps(data, indent=2, ensure_ascii=False, default=str)}")

def _model_to_dict(m):
    # Compat Pydantic v1/v2
    try:
        return m.model_dump()
    except Exception:
        try:
            return m.dict()
        except Exception:
            return json.loads(json.dumps(m, default=str))

# ===================== DB helpers =====================
def db_exec(sql: str, params: dict | None = None):
    if not engine:
        return
    with engine.begin() as conn:
        conn.execute(text(sql), params or {})

def db_query_one(sql: str, params: dict | None = None) -> Optional[dict]:
    if not engine:
        return None
    with engine.begin() as conn:
        row = conn.execute(text(sql), params or {}).mappings().first()
        return dict(row) if row else None

def db_query_all(sql: str, params: dict | None = None) -> List[dict]:
    if not engine:
        return []
    with engine.begin() as conn:
        rows = conn.execute(text(sql), params or {}).mappings().all()
        return [dict(r) for r in rows]

# ===================== Encodings (filesystem) =====================
class PersonDB:
    def __init__(self, root: str):
        self.root = root
        self.labels: List[str] = []       # nombres
        self.embs: List[np.ndarray] = []  # encodings (128,)
        self.rep_paths: Dict[str, str] = {}  # label -> ruta relativa representativa
        self._load()

    def _load(self):
        self.labels.clear()
        self.embs.clear()
        self.rep_paths.clear()
        base = pathlib.Path(self.root).resolve()
        for person_dir in sorted(glob.glob(os.path.join(self.root, "*"))):
            if not os.path.isdir(person_dir):
                continue
            person = os.path.basename(person_dir)
            imgs = sorted(glob.glob(os.path.join(person_dir, "*.*")))
            for p in imgs:
                try:
                    img = face_recognition.load_image_file(p)
                    locs = face_recognition.face_locations(img, model="hog")
                    if not locs:
                        print(f"[WARN] sin cara en {p}")
                        continue
                    encs = face_recognition.face_encodings(img, known_face_locations=locs)
                    if not encs:
                        continue
                    self.labels.append(person)
                    self.embs.append(encs[0])
                    if person not in self.rep_paths:
                        rel = pathlib.Path(p).resolve().relative_to(base)
                        self.rep_paths[person] = str(rel).replace("\\", "/")
                    print(f"[INFO] + {person} ({os.path.basename(p)})")
                except Exception as e:
                    print(f"[WARN] {p}: {e}")
        print(f"[INFO] Encodings cargados: {len(self.embs)}")

    def match(self, enc: np.ndarray) -> Tuple[Optional[str], float]:
        if not self.embs:
            return None, 999.0
        dists = np.linalg.norm(np.stack(self.embs) - enc, axis=1)
        idx = int(np.argmin(dists))
        return (self.labels[idx], float(dists[idx]))

    def photo_url(self, label: str) -> Optional[str]:
        rel = self.rep_paths.get(label)
        if not rel:
            return None
        return f"/known/{rel}"

DB = PersonDB(root=KNOWN_FACES_DIR)

# ===================== Helpers de imagen =====================
def _bytes_to_rgb(img_bytes: bytes) -> np.ndarray:
    im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return np.array(im)

def _get_image_from_url(url: str, timeout=6) -> np.ndarray:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return _bytes_to_rgb(r.content)

def _encode_first_face(rgb: np.ndarray) -> Tuple[Optional[np.ndarray], int, Optional[Tuple[int,int,int,int]]]:
    locs = face_recognition.face_locations(rgb, model="hog")
    if not locs:
        return None, 0, None
    encs = face_recognition.face_encodings(rgb, known_face_locations=locs)
    if not encs:
        return None, len(locs), None
    return encs[0], len(locs), locs[0]  # (top, right, bottom, left)

def _save_crop(rgb: np.ndarray, box: Tuple[int,int,int,int], cam_id: Optional[str]) -> str:
    top, right, bottom, left = box
    crop = rgb[top:bottom, left:right]
    img = Image.fromarray(crop)
    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    day = datetime.datetime.utcnow().strftime("%Y%m%d")
    folder = pathlib.Path(CROPS_DIR) / day
    folder.mkdir(parents=True, exist_ok=True)
    fname = f"{ts}_{cam_id or 'cam'}.jpg"
    out_path = folder / fname
    img.save(out_path, "JPEG", quality=90)
    rel = pathlib.Path(day) / fname
    return f"/events/{rel.as_posix()}"

# ===================== Modelos I/O =====================
class VerifyIn(BaseModel):
    image_url: str
    cam_id: Optional[str] = None
    faces: Optional[int] = None

class VerifyOut(BaseModel):
    autorizado: bool
    persona: Optional[str] = None
    distancia: Optional[float] = None
    umbral: float
    caras_detectadas: int
    cam_id: Optional[str] = None
    ts: float
    persona_imagen_url: Optional[str] = None
    match_crop_url: Optional[str] = None
    registro_id: Optional[int] = None

class StatsSummary(BaseModel):
    date_from: str
    date_to: str
    cam_id: Optional[str]
    total: int
    aceptados: int
    rechazados: int
    tasa_aceptacion: float

class PersonInfo(BaseModel):
    nombre: str
    cedula: Optional[str] = None
    email: Optional[str] = None
    telefono: Optional[str] = None
    foto_url: Optional[str]
    encodings_count: int
    total_intentos: int
    autorizados: int
    rechazados: int
    tasa_autorizacion: float
    ultimo_acceso: Optional[str]
    primer_acceso: Optional[str]
    fecha_registro: Optional[str] = None

class PersonsResponse(BaseModel):
    total_personas: int
    personas: List[PersonInfo]

class UnauthorizedAttempt(BaseModel):
    id: int
    timestamp: str
    cam_id: Optional[str]
    faces_detected: int
    distance: Optional[float]
    threshold: float
    crop_photo_url: Optional[str]
    person_attempted: Optional[str]  # Si detectó a alguien conocido pero no autorizado
    image_source: Optional[str]

class UnauthorizedResponse(BaseModel):
    total_intentos: int
    intentos_no_autorizados: List[UnauthorizedAttempt]

class VerificationRecord(BaseModel):
    id: int
    timestamp: str
    cam_id: Optional[str]
    image_source: Optional[str]
    faces_detected: int
    authorized: bool
    person_label: Optional[str]
    person_id: Optional[int]
    distance: Optional[float]
    threshold: float
    match_crop_url: Optional[str]

class VerificationsResponse(BaseModel):
    total_registros: int
    registros: List[VerificationRecord]
    
class MonthlyStats(BaseModel):
    año: int
    mes: int
    mes_nombre: str
    total_verificaciones: int
    autorizados: int
    rechazados: int
    tasa_autorizacion: float
    personas_unicas: int
    camaras_activas: int

class MonthlyStatsResponse(BaseModel):
    total_meses: int
    estadisticas_mensuales: List[MonthlyStats]

class PersonRegistration(BaseModel):
    nombre: str
    cedula: str
    email: str
    telefono: str

class PersonRegistrationResponse(BaseModel):
    success: bool
    message: str
    person_id: Optional[int] = None
    nombre: str
    fotos_guardadas: int
    encodings_generados: int
    carpeta_creada: str
    fecha_registro: str

class PersonUpdateRequest(BaseModel):
    cedula: Optional[str] = None
    email: Optional[str] = None
    telefono: Optional[str] = None

class PersonUpdateResponse(BaseModel):
    success: bool
    message: str
    person_id: Optional[int] = None
    nombre: str
    datos_actualizados: bool
    fotos_agregadas: int
    encodings_nuevos: int
    total_encodings: int
    fecha_actualizacion: str

class DailyPersonActivity(BaseModel):
    nombre: str
    cedula: Optional[str] = None
    email: Optional[str] = None
    telefono: Optional[str] = None
    foto_perfil_url: Optional[str] = None
    total_intentos: int
    intentos_autorizados: int
    intentos_rechazados: int
    tasa_autorizacion: float
    primer_intento: str
    ultimo_intento: str
    fotos_intentos: List[str]  # URLs de los crops de cada intento
    detalle_intentos: List[dict]  # Detalle de cada intento

class DailyActivityResponse(BaseModel):
    fecha: str
    total_personas_activas: int
    total_verificaciones: int
    total_autorizadas: int
    total_rechazadas: int
    actividad_por_persona: List[DailyPersonActivity]

# ===================== Startup =====================
@app.on_event("startup")
def _startup():
    # Montar estáticos
    app.mount("/known", StaticFiles(directory=KNOWN_FACES_DIR), name="known")
    pathlib.Path(CROPS_DIR).mkdir(parents=True, exist_ok=True)
    app.mount("/events", StaticFiles(directory=CROPS_DIR), name="events")

    if not engine:
        print("[WARN] MYSQL_URL no configurada; no se registrarán eventos ni stats.")
        return

    ddl_persons = """
    CREATE TABLE IF NOT EXISTS persons (
      id INT AUTO_INCREMENT PRIMARY KEY,
      label VARCHAR(100) NOT NULL UNIQUE,
      image_url VARCHAR(512) NULL,
      cedula VARCHAR(50) NULL,
      email VARCHAR(200) NULL,
      telefono VARCHAR(50) NULL,
      fecha_registro TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      activo TINYINT(1) DEFAULT 1,
      INDEX idx_cedula (cedula),
      INDEX idx_email (email)
    )"""
    ddl_verif = """
    CREATE TABLE IF NOT EXISTS verifications (
      id BIGINT AUTO_INCREMENT PRIMARY KEY,
      ts TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
      cam_id VARCHAR(64) NULL,
      image_source TEXT NULL,
      faces_detected INT NOT NULL,
      authorized TINYINT(1) NOT NULL,
      person_label VARCHAR(100) NULL,
      person_id INT NULL,
      distance DOUBLE NULL,
      threshold DOUBLE NOT NULL,
      match_crop_url VARCHAR(512) NULL,
      INDEX idx_ts (ts),
      INDEX idx_cam (cam_id),
      INDEX idx_auth (authorized),
      INDEX idx_person (person_label),
      CONSTRAINT fk_person_id FOREIGN KEY (person_id) REFERENCES persons(id)
    )"""
    db_exec(ddl_persons)
    db_exec(ddl_verif)

    # seed/actualizar persons desde filesystem
    if DB.labels:
        for label in sorted(set(DB.labels)):
            url = DB.photo_url(label)
            db_exec(
                """INSERT INTO persons(label, image_url)
                   VALUES (:label, :url)
                   ON DUPLICATE KEY UPDATE image_url = VALUES(image_url)""",
                {"label": label, "url": url}
            )

# ===================== Endpoints básicos =====================
@app.get("/health")
def health():
    return {
        "ok": True,
        "encodings": len(DB.embs),
        "threshold": THRESHOLD,
        "mysql": bool(engine)
    }

def _log_and_make_response(*, rgb: Optional[np.ndarray], box: Optional[Tuple[int,int,int,int]],
                           body_image_source: Optional[str], cam_id: Optional[str],
                           name: Optional[str], dist: Optional[float],
                           nfaces: int, ok: bool) -> VerifyOut:
    persona_url = DB.photo_url(name) if (ok and name) else None
    crop_url = None
    if SAVE_MATCH_CROP and rgb is not None and box is not None:
        try:
            crop_url = _save_crop(rgb, box, cam_id)
        except Exception as e:
            print(f"[WARN] al guardar crop: {e}")

    registro_id = None
    if engine:
        try:
            with engine.begin() as conn:
                # Resolver person_id en la MISMA conexión
                person_id = None
                if name:
                    row = conn.execute(
                        text("SELECT id FROM persons WHERE label = :label"),
                        {"label": name}
                    ).mappings().first()
                    person_id = row["id"] if row else None
                    console_log({"person_id_found": person_id}, "PERSON_ID_LOOKUP")

                params = {
                    "cam": cam_id,
                    "src": body_image_source,
                    "faces": int(nfaces),
                    "auth": 1 if ok else 0,
                    "plabel": name,
                    "pid": person_id,
                    "dist": float(dist) if dist is not None else None,
                    "thr": float(THRESHOLD),
                    "crop": crop_url
                }

                res = conn.execute(
                    text("""INSERT INTO verifications
                            (cam_id, image_source, faces_detected, authorized,
                             person_label, person_id, distance, threshold, match_crop_url)
                            VALUES (:cam, :src, :faces, :auth, :plabel, :pid, :dist, :thr, :crop)"""),
                    params
                )

                # Obtener lastrowid en la MISMA conexión
                try:
                    registro_id = res.lastrowid
                except Exception:
                    registro_id = conn.execute(text("SELECT LAST_INSERT_ID()")).scalar()

                console_log({"status": "INSERT SUCCESS", "registro_id": registro_id}, "DB INSERT")
        except Exception as e:
            console_log({"error": str(e)}, "DB INSERT ERROR")
            print(f"[ERROR] Database insert failed: {e}")
    else:
        console_log({"database_connected": False}, "NO DATABASE")

    return VerifyOut(
        autorizado=ok, persona=name if ok else None, distancia=dist,
        umbral=THRESHOLD, caras_detectadas=nfaces, cam_id=cam_id, ts=time.time(),
        persona_imagen_url=persona_url, match_crop_url=crop_url, registro_id=registro_id
    )

@app.post("/verify", response_model=VerifyOut)
def verify(body: VerifyIn):
    console_log(_model_to_dict(body), "VERIFY INPUT")

    rgb = _get_image_from_url(body.image_url)
    enc, nfaces, box = _encode_first_face(rgb)
    if enc is None:
        result = _log_and_make_response(
            rgb=rgb, box=None, body_image_source=body.image_url, cam_id=body.cam_id,
            name=None, dist=None, nfaces=nfaces, ok=False
        )
        console_log(_model_to_dict(result), "VERIFY OUTPUT - NO FACE")
        return result

    name, dist = DB.match(enc)
    ok = (dist <= THRESHOLD)

    console_log({
        "name": name, "distance": dist, "threshold": THRESHOLD,
        "authorized": ok, "faces_detected": nfaces
    }, "MATCH RESULT")

    result = _log_and_make_response(
        rgb=rgb, box=box, body_image_source=body.image_url, cam_id=body.cam_id,
        name=name if ok else None, dist=dist, nfaces=nfaces, ok=ok
    )
    console_log(_model_to_dict(result), "VERIFY OUTPUT - FINAL")
    return result

@app.post("/verify_upload", response_model=VerifyOut)
def verify_upload(file: UploadFile = File(...), cam_id: str = Form(default=None)):
    console_log({"filename": file.filename, "cam_id": cam_id}, "VERIFY_UPLOAD INPUT")

    rgb = _bytes_to_rgb(file.file.read())
    enc, nfaces, box = _encode_first_face(rgb)
    if enc is None:
        result = _log_and_make_response(
            rgb=rgb, box=None, body_image_source="upload", cam_id=cam_id,
            name=None, dist=None, nfaces=nfaces, ok=False
        )
        console_log(_model_to_dict(result), "VERIFY_UPLOAD OUTPUT - NO FACE")
        return result

    name, dist = DB.match(enc)
    ok = (dist <= THRESHOLD)

    console_log({
        "name": name, "distance": dist, "threshold": THRESHOLD,
        "authorized": ok, "faces_detected": nfaces
    }, "UPLOAD MATCH RESULT")

    result = _log_and_make_response(
        rgb=rgb, box=box, body_image_source="upload", cam_id=cam_id,
        name=name if ok else None, dist=dist, nfaces=nfaces, ok=ok
    )
    console_log(_model_to_dict(result), "VERIFY_UPLOAD OUTPUT - FINAL")
    return result

# ===================== Stats =====================
@app.get("/stats/summary", response_model=StatsSummary)
def stats_summary(
    date_from: str = Query(default=None, description="ISO date/time, ej 2025-08-18T00:00:00"),
    date_to: str = Query(default=None, description="ISO date/time"),
    cam_id: Optional[str] = Query(default=None)
):
    if not engine:
        return StatsSummary(date_from=date_from or "", date_to=date_to or "", cam_id=cam_id,
                            total=0, aceptados=0, rechazados=0, tasa_aceptacion=0.0)

    dt_to = datetime.datetime.fromisoformat(date_to) if date_to else datetime.datetime.utcnow()
    dt_from = datetime.datetime.fromisoformat(date_from) if date_from else dt_to - datetime.timedelta(days=1)

    params = {"f": dt_from, "t": dt_to, "cam": cam_id}
    cam_filter = "AND cam_id = :cam" if cam_id else ""
    total = db_query_one(f"SELECT COUNT(*) c FROM verifications WHERE ts BETWEEN :f AND :t {cam_filter}", params)["c"]
    ok = db_query_one(f"SELECT SUM(authorized=1) c FROM verifications WHERE ts BETWEEN :f AND :t {cam_filter}", params)["c"] or 0
    rej = total - ok
    rate = (ok / total) if total else 0.0

    return StatsSummary(
        date_from=dt_from.isoformat(), date_to=dt_to.isoformat(),
        cam_id=cam_id, total=int(total), aceptados=int(ok), rechazados=int(rej),
        tasa_aceptacion=round(rate, 4)
    )

@app.get("/stats/by_person")
def stats_by_person(
    date_from: str = Query(default=None),
    date_to: str = Query(default=None),
    cam_id: Optional[str] = Query(default=None),
    limit: int = 50
):
    if not engine:
        return []
    dt_to = datetime.datetime.fromisoformat(date_to) if date_to else datetime.datetime.utcnow()
    dt_from = datetime.datetime.fromisoformat(date_from) if date_from else dt_to - datetime.timedelta(days=1)
    params = {"f": dt_from, "t": dt_to, "cam": cam_id, "lim": limit}
    cam_filter = "AND cam_id = :cam" if cam_id else ""
    rows = db_query_all(
        f"""SELECT person_label AS persona,
                    SUM(authorized=1) AS aceptados,
                    SUM(authorized=0) AS rechazados,
                    COUNT(*) AS total,
                    ROUND(SUM(authorized=1)/COUNT(*),4) AS tasa_aceptacion
             FROM verifications
             WHERE ts BETWEEN :f AND :t {cam_filter}
             GROUP BY person_label
             ORDER BY total DESC
             LIMIT :lim""", params
    )
    return rows

@app.get("/stats/timeseries")
def stats_timeseries(
    date_from: str = Query(default=None),
    date_to: str = Query(default=None),
    cam_id: Optional[str] = Query(default=None)
):
    if not engine:
        return []
    dt_to = datetime.datetime.fromisoformat(date_to) if date_to else datetime.datetime.utcnow()
    dt_from = datetime.datetime.fromisoformat(date_from) if date_from else dt_to - datetime.timedelta(days=7)
    params = {"f": dt_from, "t": dt_to, "cam": cam_id}
    cam_filter = "AND cam_id = :cam" if cam_id else ""
    rows = db_query_all(
        f"""SELECT DATE(ts) AS fecha,
                    SUM(authorized=1) AS aceptados,
                    SUM(authorized=0) AS rechazados,
                    COUNT(*) AS total
             FROM verifications
             WHERE ts BETWEEN :f AND :t {cam_filter}
             GROUP BY DATE(ts)
             ORDER BY fecha""", params
    )
    return rows

# ===================== Admin / Debug =====================
@app.post("/admin/reload_encodings")
def reload_encodings():
    global DB
    DB._load()
    result = {
        "message": "Encodings recargados exitosamente",
        "total_encodings": len(DB.embs),
        "personas_unicas": len(set(DB.labels)),
        "personas": list(set(DB.labels)),
        "directorio": KNOWN_FACES_DIR,
        "encodings_por_persona": {persona: DB.labels.count(persona) for persona in set(DB.labels)}
    }
    console_log(result, "RELOAD_ENCODINGS")
    return result

@app.get("/debug/db_info")
def debug_db_info():
    result = {
        "encodings_count": len(DB.embs),
        "persons": list(set(DB.labels)),
        "encodings_by_person": {person: DB.labels.count(person) for person in set(DB.labels)},
        "representative_photos": DB.rep_paths,
        "threshold": THRESHOLD,
        "known_faces_dir": KNOWN_FACES_DIR,
        "mysql_connected": bool(engine)
    }
    console_log(result, "DEBUG_INFO")
    return result

@app.get("/debug/which_db")
def which_db():
    if not engine:
        return {"engine": False}
    with engine.begin() as conn:
        row = conn.execute(text("SELECT CURRENT_USER() user, DATABASE() db, @@hostname host")).mappings().first()
        return dict(row)

@app.post("/debug/test_insert")
def debug_test_insert():
    if not engine:
        return {"ok": False, "reason": "no engine"}
    with engine.begin() as conn:
        res = conn.execute(
            text("""INSERT INTO verifications
                    (cam_id, image_source, faces_detected, authorized, threshold)
                    VALUES ('test-cam', 'debug', 0, 0, :thr)"""),
            {"thr": float(THRESHOLD)}
        )
        try:
            last_id = res.lastrowid
        except Exception:
            last_id = conn.execute(text("SELECT LAST_INSERT_ID()")).scalar()
        return {"ok": True, "last_id": last_id}

@app.get("/persons", response_model=PersonsResponse)
def get_all_persons():
    """Obtiene todas las personas registradas con sus estadísticas de acceso"""
    
    # Obtener personas desde el sistema de archivos (conocidas)
    personas_conocidas = set(DB.labels)
    personas_info = []
    
    for persona in personas_conocidas:
        # Información básica desde el sistema de archivos
        foto_url = DB.photo_url(persona)
        encodings_count = DB.labels.count(persona)
        
        # Estadísticas desde la base de datos (si está disponible)
        total_intentos = 0
        autorizados = 0
        rechazados = 0
        ultimo_acceso = None
        primer_acceso = None
        
        if engine:
            # Obtener datos personales desde la BD
            persona_bd = db_query_one("""
                SELECT cedula, email, telefono, image_url, fecha_registro
                FROM persons 
                WHERE label = :persona
            """, {"persona": persona})
            
            # Estadísticas de verificaciones
            stats = db_query_one("""
                SELECT 
                    COUNT(*) as total,
                    SUM(authorized = 1) as auth_count,
                    SUM(authorized = 0) as reject_count,
                    MAX(ts) as ultimo,
                    MIN(ts) as primero
                FROM verifications 
                WHERE person_label = :persona
            """, {"persona": persona})
            
            if stats:
                total_intentos = int(stats["total"] or 0)
                autorizados = int(stats["auth_count"] or 0)
                rechazados = int(stats["reject_count"] or 0)
                ultimo_acceso = stats["ultimo"].isoformat() if stats["ultimo"] else None
                primer_acceso = stats["primero"].isoformat() if stats["primero"] else None
            
            # Usar datos de la BD si están disponibles
            cedula = persona_bd["cedula"] if persona_bd else None
            email = persona_bd["email"] if persona_bd else None
            telefono = persona_bd["telefono"] if persona_bd else None
            fecha_registro = persona_bd["fecha_registro"].isoformat() if persona_bd and persona_bd["fecha_registro"] else None
            
            # Usar foto de BD o del sistema de archivos
            if persona_bd and persona_bd["image_url"]:
                foto_url = persona_bd["image_url"]
        else:
            cedula = email = telefono = fecha_registro = None
        
        tasa_autorizacion = (autorizados / total_intentos) if total_intentos > 0 else 0.0
        
        persona_info = PersonInfo(
            nombre=persona,
            cedula=cedula,
            email=email,
            telefono=telefono,
            foto_url=foto_url,
            encodings_count=encodings_count,
            total_intentos=total_intentos,
            autorizados=autorizados,
            rechazados=rechazados,
            tasa_autorizacion=round(tasa_autorizacion, 4),
            ultimo_acceso=ultimo_acceso,
            primer_acceso=primer_acceso,
            fecha_registro=fecha_registro
        )
        
        personas_info.append(persona_info)
    
    # También incluir personas que aparecen en la BD pero no están en archivos
    if engine:
        personas_solo_bd = db_query_all("""
            SELECT DISTINCT person_label 
            FROM verifications 
            WHERE person_label IS NOT NULL 
            AND person_label NOT IN :conocidas
        """, {"conocidas": tuple(personas_conocidas) if personas_conocidas else ("",)})
        
        for row in personas_solo_bd:
            persona = row["person_label"]
            
            # Obtener datos personales desde la BD
            persona_bd = db_query_one("""
                SELECT cedula, email, telefono, image_url, fecha_registro
                FROM persons 
                WHERE label = :persona
            """, {"persona": persona})
            
            stats = db_query_one("""
                SELECT 
                    COUNT(*) as total,
                    SUM(authorized = 1) as auth_count,
                    SUM(authorized = 0) as reject_count,
                    MAX(ts) as ultimo,
                    MIN(ts) as primero
                FROM verifications 
                WHERE person_label = :persona
            """, {"persona": persona})
            
            total_intentos = int(stats["total"] or 0)
            autorizados = int(stats["auth_count"] or 0)
            rechazados = int(stats["reject_count"] or 0)
            tasa_autorizacion = (autorizados / total_intentos) if total_intentos > 0 else 0.0
            
            # Usar datos de la BD si están disponibles
            cedula = persona_bd["cedula"] if persona_bd else None
            email = persona_bd["email"] if persona_bd else None
            telefono = persona_bd["telefono"] if persona_bd else None
            fecha_registro = persona_bd["fecha_registro"].isoformat() if persona_bd and persona_bd["fecha_registro"] else None
            
            persona_info = PersonInfo(
                nombre=persona,
                cedula=cedula,
                email=email,
                telefono=telefono,
                foto_url=None,  # No tiene foto en archivos
                encodings_count=0,  # No está en archivos
                total_intentos=total_intentos,
                autorizados=autorizados,
                rechazados=rechazados,
                tasa_autorizacion=round(tasa_autorizacion, 4),
                ultimo_acceso=stats["ultimo"].isoformat() if stats["ultimo"] else None,
                primer_acceso=stats["primero"].isoformat() if stats["primero"] else None,
                fecha_registro=fecha_registro
            )
            
            personas_info.append(persona_info)
    
    # Ordenar por nombre
    personas_info.sort(key=lambda p: p.nombre)
    
    result = PersonsResponse(
        total_personas=len(personas_info),
        personas=personas_info
    )
    
    console_log({
        "total_personas": len(personas_info),
        "personas_nombres": [p.nombre for p in personas_info]
    }, "GET_PERSONS")
    
    return result

@app.get("/unauthorized", response_model=UnauthorizedResponse)
def get_unauthorized_attempts(
    limit: int = Query(default=50, description="Número máximo de resultados"),
    date_from: str = Query(default=None, description="Fecha desde (ISO format)"),
    date_to: str = Query(default=None, description="Fecha hasta (ISO format)"),
    cam_id: Optional[str] = Query(default=None, description="Filtrar por cámara específica")
):
    """Obtiene todos los intentos no autorizados con sus fotos"""
    
    if not engine:
        return UnauthorizedResponse(
            total_intentos=0,
            intentos_no_autorizados=[]
        )
    
    # Configurar fechas por defecto (últimos 7 días)
    import datetime
    dt_to = datetime.datetime.fromisoformat(date_to) if date_to else datetime.datetime.utcnow()
    dt_from = datetime.datetime.fromisoformat(date_from) if date_from else dt_to - datetime.timedelta(days=7)
    
    # Construir filtros
    params = {"f": dt_from, "t": dt_to, "limit": limit}
    cam_filter = ""
    
    if cam_id:
        cam_filter = "AND cam_id = :cam_id"
        params["cam_id"] = cam_id
    
    # Query para obtener intentos no autorizados
    query = f"""
        SELECT 
            id,
            ts,
            cam_id,
            faces_detected,
            distance,
            threshold,
            match_crop_url,
            person_label,
            image_source
        FROM verifications 
        WHERE authorized = 0 
        AND ts BETWEEN :f AND :t
        {cam_filter}
        ORDER BY ts DESC 
        LIMIT :limit
    """
    
    rows = db_query_all(query, params)
    
    intentos_no_autorizados = []
    for row in rows:
        intento = UnauthorizedAttempt(
            id=row["id"],
            timestamp=row["ts"].isoformat() if row["ts"] else None,
            cam_id=row["cam_id"],
            faces_detected=row["faces_detected"],
            distance=float(row["distance"]) if row["distance"] else None,
            threshold=float(row["threshold"]),
            crop_photo_url=row["match_crop_url"],
            person_attempted=row["person_label"],  # Puede ser None si no reconoció a nadie
            image_source=row["image_source"]
        )
        intentos_no_autorizados.append(intento)
    
    result = UnauthorizedResponse(
        total_intentos=len(intentos_no_autorizados),
        intentos_no_autorizados=intentos_no_autorizados
    )
    
    console_log({
        "total_unauthorized": len(intentos_no_autorizados),
        "date_range": f"{dt_from.isoformat()} to {dt_to.isoformat()}",
        "cam_filter": cam_id
    }, "GET_UNAUTHORIZED")
    
    return result

@app.get("/unauthorized/photos")
def get_unauthorized_photos(
    date_from: str = Query(default=None),
    date_to: str = Query(default=None)
):
    """Obtiene solo las URLs de las fotos de intentos no autorizados"""
    
    if not engine:
        return {"photos": [], "total": 0}
    
    import datetime
    dt_to = datetime.datetime.fromisoformat(date_to) if date_to else datetime.datetime.utcnow()
    dt_from = datetime.datetime.fromisoformat(date_from) if date_from else dt_to - datetime.timedelta(days=7)
    
    rows = db_query_all("""
        SELECT DISTINCT match_crop_url, ts, cam_id
        FROM verifications 
        WHERE authorized = 0 
        AND match_crop_url IS NOT NULL
        AND ts BETWEEN :f AND :t
        ORDER BY ts DESC
    """, {"f": dt_from, "t": dt_to})
    
    photos = []
    for row in rows:
        photos.append({
            "photo_url": row["match_crop_url"],
            "timestamp": row["ts"].isoformat() if row["ts"] else None,
            "cam_id": row["cam_id"]
        })
    
    return {
        "photos": photos,
        "total": len(photos),
        "date_range": {
            "from": dt_from.isoformat(),
            "to": dt_to.isoformat()
        }
    }

@app.get("/verifications", response_model=VerificationsResponse)
def get_all_verifications(
    limit: int = Query(default=100, description="Número máximo de registros"),
    offset: int = Query(default=0, description="Número de registros a saltar"),
    authorized: Optional[bool] = Query(default=None, description="Filtrar por autorizados (true/false)"),
    person_label: Optional[str] = Query(default=None, description="Filtrar por persona específica"),
    cam_id: Optional[str] = Query(default=None, description="Filtrar por cámara específica"),
    date_from: str = Query(default=None, description="Fecha desde (ISO format)"),
    date_to: str = Query(default=None, description="Fecha hasta (ISO format)")
):
    """Obtiene todos los registros de la tabla verifications con filtros opcionales"""
    
    if not engine:
        return VerificationsResponse(
            total_registros=0,
            registros=[]
        )
    
    # Configurar fechas por defecto (últimos 30 días)
    import datetime
    dt_to = datetime.datetime.fromisoformat(date_to) if date_to else datetime.datetime.utcnow()
    dt_from = datetime.datetime.fromisoformat(date_from) if date_from else dt_to - datetime.timedelta(days=30)
    
    # Construir filtros dinámicos
    filters = ["ts BETWEEN :date_from AND :date_to"]
    params = {
        "date_from": dt_from,
        "date_to": dt_to,
        "limit": limit,
        "offset": offset
    }
    
    if authorized is not None:
        filters.append("authorized = :authorized")
        params["authorized"] = 1 if authorized else 0
    
    if person_label:
        filters.append("person_label = :person_label")
        params["person_label"] = person_label
        
    if cam_id:
        filters.append("cam_id = :cam_id")
        params["cam_id"] = cam_id
    
    where_clause = " AND ".join(filters)
    
    # Query principal
    query = f"""
        SELECT 
            id,
            ts,
            cam_id,
            image_source,
            faces_detected,
            authorized,
            person_label,
            person_id,
            distance,
            threshold,
            match_crop_url
        FROM verifications 
        WHERE {where_clause}
        ORDER BY ts DESC 
        LIMIT :limit OFFSET :offset
    """
    
    rows = db_query_all(query, params)
    
    registros = []
    for row in rows:
        registro = VerificationRecord(
            id=row["id"],
            timestamp=row["ts"].isoformat() if row["ts"] else None,
            cam_id=row["cam_id"],
            image_source=row["image_source"],
            faces_detected=row["faces_detected"],
            authorized=bool(row["authorized"]),
            person_label=row["person_label"],
            person_id=row["person_id"],
            distance=float(row["distance"]) if row["distance"] else None,
            threshold=float(row["threshold"]),
            match_crop_url=row["match_crop_url"]
        )
        registros.append(registro)
    
    result = VerificationsResponse(
        total_registros=len(registros),
        registros=registros
    )
    
    console_log({
        "total_records": len(registros),
        "filters_applied": {
            "authorized": authorized,
            "person_label": person_label,
            "cam_id": cam_id,
            "date_range": f"{dt_from.isoformat()} to {dt_to.isoformat()}"
        }
    }, "GET_VERIFICATIONS")
    
    return result

@app.get("/verifications/monthly", response_model=MonthlyStatsResponse)
def get_monthly_verification_stats(
    year_from: int = Query(default=None, description="Año desde (ej: 2024)"),
    year_to: int = Query(default=None, description="Año hasta (ej: 2025)")
):
    """Obtiene estadísticas de verificaciones agrupadas por mes"""
    
    if not engine:
        return MonthlyStatsResponse(
            total_meses=0,
            estadisticas_mensuales=[]
        )
    
    import datetime
    current_year = datetime.datetime.utcnow().year
    year_from = year_from or (current_year - 1)  # Por defecto último año
    year_to = year_to or current_year
    
    # Query para estadísticas mensuales
    query = """
        SELECT 
            YEAR(ts) as año,
            MONTH(ts) as mes,
            COUNT(*) as total_verificaciones,
            SUM(authorized = 1) as autorizados,
            SUM(authorized = 0) as rechazados,
            COUNT(DISTINCT person_label) as personas_unicas,
            COUNT(DISTINCT cam_id) as camaras_activas
        FROM verifications 
        WHERE YEAR(ts) BETWEEN :year_from AND :year_to
        GROUP BY YEAR(ts), MONTH(ts)
        ORDER BY año DESC, mes DESC
    """
    
    rows = db_query_all(query, {
        "year_from": year_from,
        "year_to": year_to
    })
    
    # Nombres de meses en español
    meses_nombres = {
        1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril",
        5: "Mayo", 6: "Junio", 7: "Julio", 8: "Agosto",
        9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"
    }
    
    estadisticas = []
    for row in rows:
        total = int(row["total_verificaciones"])
        autorizados = int(row["autorizados"] or 0)
        rechazados = int(row["rechazados"] or 0)
        tasa_autorizacion = (autorizados / total) if total > 0 else 0.0
        
        stat = MonthlyStats(
            año=int(row["año"]),
            mes=int(row["mes"]),
            mes_nombre=meses_nombres.get(int(row["mes"]), "Desconocido"),
            total_verificaciones=total,
            autorizados=autorizados,
            rechazados=rechazados,
            tasa_autorizacion=round(tasa_autorizacion, 4),
            personas_unicas=int(row["personas_unicas"] or 0),
            camaras_activas=int(row["camaras_activas"] or 0)
        )
        estadisticas.append(stat)
    
    result = MonthlyStatsResponse(
        total_meses=len(estadisticas),
        estadisticas_mensuales=estadisticas
    )
    
    console_log({
        "total_months": len(estadisticas),
        "year_range": f"{year_from} to {year_to}",
        "months_with_data": [f"{s.año}-{s.mes}" for s in estadisticas[:5]]  # Solo primeros 5
    }, "GET_MONTHLY_STATS")
    
    return result

@app.post("/persons/register", response_model=PersonRegistrationResponse)
async def register_new_person(
    nombre: str = Form(...),
    cedula: str = Form(...),
    email: str = Form(...),
    telefono: str = Form(...),
    fotos: List[UploadFile] = File(...)
):
    """Registra una nueva persona con sus datos y múltiples fotos"""
    
    console_log({
        "nombre": nombre,
        "cedula": cedula,
        "email": email,
        "telefono": telefono,
        "total_fotos": len(fotos)
    }, "PERSON_REGISTRATION_START")
    
    # Validar que el nombre no esté vacío y sea válido para carpeta
    nombre_limpio = "".join(c for c in nombre if c.isalnum() or c in (' ', '-', '_')).strip()
    nombre_carpeta = nombre_limpio.replace(" ", "_")
    
    if not nombre_carpeta:
        return PersonRegistrationResponse(
            success=False,
            message="Nombre inválido para crear carpeta",
            nombre=nombre,
            fotos_guardadas=0,
            encodings_generados=0,
            carpeta_creada="",
            fecha_registro=""
        )
    
    # Crear directorio para la persona
    person_dir = pathlib.Path(KNOWN_FACES_DIR) / nombre_carpeta
    person_dir.mkdir(parents=True, exist_ok=True)
    
    fotos_guardadas = 0
    encodings_generados = 0
    fecha_registro = datetime.datetime.utcnow()
    
    # Guardar fotos y generar encodings
    for i, foto in enumerate(fotos):
        try:
            # Validar que sea una imagen
            if not foto.content_type.startswith('image/'):
                console_log({"archivo": foto.filename, "error": "No es imagen"}, "FOTO_SKIP")
                continue
            
            # Generar nombre único para la foto
            ext = pathlib.Path(foto.filename or "photo.jpg").suffix or ".jpg"
            timestamp = fecha_registro.strftime("%Y%m%d_%H%M%S")
            foto_nombre = f"{nombre_carpeta}_{timestamp}_{i+1:02d}{ext}"
            foto_path = person_dir / foto_nombre
            
            # Guardar archivo físico
            content = await foto.read()
            with open(foto_path, "wb") as f:
                f.write(content)
            
            fotos_guardadas += 1
            
            # Verificar que tenga cara válida para encoding
            try:
                rgb = _bytes_to_rgb(content)
                enc, nfaces, box = _encode_first_face(rgb)
                if enc is not None:
                    encodings_generados += 1
                    console_log({
                        "foto": foto_nombre,
                        "faces_detected": nfaces,
                        "encoding_success": True
                    }, "ENCODING_SUCCESS")
                else:
                    console_log({
                        "foto": foto_nombre,
                        "faces_detected": nfaces,
                        "encoding_success": False
                    }, "ENCODING_FAILED")
            except Exception as e:
                console_log({
                    "foto": foto_nombre,
                    "error": str(e)
                }, "ENCODING_ERROR")
            
        except Exception as e:
            console_log({
                "foto": foto.filename,
                "error": str(e)
            }, "FOTO_ERROR")
            continue
    
    # Recargar encodings del sistema
    global DB
    DB._load()
    
    person_id = None
    
    # Guardar en base de datos
    if engine:
        try:
            # Obtener URL de foto representativa
            foto_url = DB.photo_url(nombre_carpeta) if nombre_carpeta in DB.labels else None
            
            with engine.begin() as conn:
                # Insertar o actualizar persona
                result = conn.execute(
                    text("""INSERT INTO persons (label, cedula, email, telefono, image_url, fecha_registro, activo)
                             VALUES (:nombre, :cedula, :email, :telefono, :foto_url, :fecha, 1)
                             ON DUPLICATE KEY UPDATE 
                                cedula = VALUES(cedula),
                                email = VALUES(email), 
                                telefono = VALUES(telefono),
                                image_url = VALUES(image_url),
                                activo = 1"""),
                    {
                        "nombre": nombre_carpeta,
                        "cedula": cedula,
                        "email": email,
                        "telefono": telefono,
                        "foto_url": foto_url,
                        "fecha": fecha_registro
                    }
                )
                
                # Obtener ID de la persona
                person_row = conn.execute(
                    text("SELECT id FROM persons WHERE label = :nombre"),
                    {"nombre": nombre_carpeta}
                ).mappings().first()
                
                person_id = person_row["id"] if person_row else None
                
                console_log({
                    "person_id": person_id,
                    "database_saved": True
                }, "DB_PERSON_SAVED")
                
        except Exception as e:
            console_log({"error": str(e)}, "DB_PERSON_ERROR")
    
    result = PersonRegistrationResponse(
        success=fotos_guardadas > 0,
        message=f"Persona registrada exitosamente. {fotos_guardadas} fotos guardadas, {encodings_generados} encodings generados.",
        person_id=person_id,
        nombre=nombre,
        fotos_guardadas=fotos_guardadas,
        encodings_generados=encodings_generados,
        carpeta_creada=str(person_dir),
        fecha_registro=fecha_registro.isoformat()
    )
    
    console_log({
        "success": result.success,
        "fotos_guardadas": fotos_guardadas,
        "encodings_generados": encodings_generados,
        "total_encodings_sistema": len(DB.embs)
    }, "PERSON_REGISTRATION_COMPLETE")
    
    return result

@app.put("/persons/{nombre}/update", response_model=PersonUpdateResponse)
async def update_person(
    nombre: str,
    cedula: Optional[str] = Form(default=None),
    email: Optional[str] = Form(default=None),
    telefono: Optional[str] = Form(default=None),
    fotos_nuevas: Optional[List[UploadFile]] = File(default=None)
):
    """Actualiza los datos de una persona existente y/o agrega nuevas fotos"""
    
    console_log({
        "nombre": nombre,
        "cedula": cedula,
        "email": email,
        "telefono": telefono,
        "tiene_fotos_nuevas": bool(fotos_nuevas),
        "total_fotos_nuevas": len(fotos_nuevas) if fotos_nuevas else 0
    }, "PERSON_UPDATE_START")
    
    # Verificar que la persona existe
    nombre_carpeta = nombre.replace(" ", "_")
    person_dir = pathlib.Path(KNOWN_FACES_DIR) / nombre_carpeta
    
    if not person_dir.exists():
        return PersonUpdateResponse(
            success=False,
            message=f"La persona '{nombre}' no existe en el sistema",
            nombre=nombre,
            datos_actualizados=False,
            fotos_agregadas=0,
            encodings_nuevos=0,
            total_encodings=0,
            fecha_actualizacion=""
        )
    
    fecha_actualizacion = datetime.datetime.utcnow()
    person_id = None
    datos_actualizados = False
    fotos_agregadas = 0
    encodings_nuevos = 0
    
    # Actualizar datos personales en la base de datos
    if engine and (cedula or email or telefono):
        try:
            # Construir query dinámico
            campos_actualizar = []
            params = {"nombre": nombre_carpeta}
            
            if cedula is not None:
                campos_actualizar.append("cedula = :cedula")
                params["cedula"] = cedula
                
            if email is not None:
                campos_actualizar.append("email = :email")
                params["email"] = email
                
            if telefono is not None:
                campos_actualizar.append("telefono = :telefono")
                params["telefono"] = telefono
            
            if campos_actualizar:
                query = f"""
                    UPDATE persons 
                    SET {', '.join(campos_actualizar)}, fecha_registro = :fecha
                    WHERE label = :nombre
                """
                params["fecha"] = fecha_actualizacion
                
                with engine.begin() as conn:
                    result = conn.execute(text(query), params)
                    
                    if result.rowcount > 0:
                        datos_actualizados = True
                        console_log({"campos_actualizados": len(campos_actualizar)}, "DB_UPDATE_SUCCESS")
                    
                    # Obtener ID de la persona
                    person_row = conn.execute(
                        text("SELECT id FROM persons WHERE label = :nombre"),
                        {"nombre": nombre_carpeta}
                    ).mappings().first()
                    
                    person_id = person_row["id"] if person_row else None
                    
        except Exception as e:
            console_log({"error": str(e)}, "DB_UPDATE_ERROR")
    
    # Agregar nuevas fotos si se proporcionaron
    if fotos_nuevas:
        global DB
        encodings_antes = len([label for label in DB.labels if label == nombre_carpeta])
        
        for i, foto in enumerate(fotos_nuevas):
            try:
                # Validar que sea una imagen
                if not foto.content_type.startswith('image/'):
                    console_log({"archivo": foto.filename, "error": "No es imagen"}, "FOTO_SKIP")
                    continue
                
                # Generar nombre único para la foto
                ext = pathlib.Path(foto.filename or "photo.jpg").suffix or ".jpg"
                timestamp = fecha_actualizacion.strftime("%Y%m%d_%H%M%S")
                foto_nombre = f"{nombre_carpeta}_{timestamp}_update_{i+1:02d}{ext}"
                foto_path = person_dir / foto_nombre
                
                # Guardar archivo físico
                content = await foto.read()
                with open(foto_path, "wb") as f:
                    f.write(content)
                
                fotos_agregadas += 1
                
                # Verificar que tenga cara válida para encoding
                try:
                    rgb = _bytes_to_rgb(content)
                    enc, nfaces, box = _encode_first_face(rgb)
                    if enc is not None:
                        console_log({
                            "foto": foto_nombre,
                            "faces_detected": nfaces,
                            "encoding_success": True
                        }, "NEW_ENCODING_SUCCESS")
                    else:
                        console_log({
                            "foto": foto_nombre,
                            "faces_detected": nfaces,
                            "encoding_success": False
                        }, "NEW_ENCODING_FAILED")
                except Exception as e:
                    console_log({
                        "foto": foto_nombre,
                        "error": str(e)
                    }, "NEW_ENCODING_ERROR")
                
            except Exception as e:
                console_log({
                    "foto": foto.filename,
                    "error": str(e)
                }, "NUEVA_FOTO_ERROR")
                continue
        
        # Recargar encodings del sistema solo si se agregaron fotos
        if fotos_agregadas > 0:
            encodings_antes = len([enc for i, lbl in enumerate(DB.labels) if lbl == nombre_carpeta])
            DB._load()
            encodings_despues = len([enc for i, lbl in enumerate(DB.labels) if lbl == nombre_carpeta])
            encodings_nuevos = encodings_despues - encodings_antes
    
    # Obtener total actual de encodings para esta persona
    total_encodings = len([label for label in DB.labels if label == nombre_carpeta])
    
    # Actualizar image_url en la base de datos si es necesario
    if engine and fotos_agregadas > 0:
        try:
            foto_url = DB.photo_url(nombre_carpeta) if nombre_carpeta in DB.labels else None
            if foto_url:
                db_exec(
                    "UPDATE persons SET image_url = :foto_url WHERE label = :nombre",
                    {"foto_url": foto_url, "nombre": nombre_carpeta}
                )
        except Exception as e:
            console_log({"error": str(e)}, "IMAGE_URL_UPDATE_ERROR")
    
    success = datos_actualizados or fotos_agregadas > 0
    message_parts = []
    
    if datos_actualizados:
        message_parts.append("datos personales actualizados")
    if fotos_agregadas > 0:
        message_parts.append(f"{fotos_agregadas} fotos agregadas")
        if encodings_nuevos > 0:
            message_parts.append(f"{encodings_nuevos} nuevos encodings generados")
    
    if not message_parts:
        message_parts.append("no se realizaron cambios")
    
    result = PersonUpdateResponse(
        success=success,
        message=f"Persona '{nombre}': " + ", ".join(message_parts) + ".",
        person_id=person_id,
        nombre=nombre,
        datos_actualizados=datos_actualizados,
        fotos_agregadas=fotos_agregadas,
        encodings_nuevos=encodings_nuevos,
        total_encodings=total_encodings,
        fecha_actualizacion=fecha_actualizacion.isoformat()
    )
    
    console_log({
        "success": success,
        "datos_actualizados": datos_actualizados,
        "fotos_agregadas": fotos_agregadas,
        "encodings_nuevos": encodings_nuevos,
        "total_encodings": total_encodings
    }, "PERSON_UPDATE_COMPLETE")
    
    return result

@app.get("/verifications/daily", response_model=DailyActivityResponse)
def get_daily_activity(
    date: str = Query(default=None, description="Fecha específica (YYYY-MM-DD) o automáticamente el último día con actividad")
):
    """Obtiene todas las verificaciones del último día agrupadas por persona con sus datos y fotos"""
    
    if not engine:
        return DailyActivityResponse(
            fecha="",
            total_personas_activas=0,
            total_verificaciones=0,
            total_autorizadas=0,
            total_rechazadas=0,
            actividad_por_persona=[]
        )
    
    import datetime
    
    # Determinar la fecha a consultar
    if date:
        try:
            fecha_consulta = datetime.datetime.strptime(date, "%Y-%m-%d").date()
        except ValueError:
            fecha_consulta = datetime.date.today()
    else:
        # Obtener la fecha más reciente con actividad
        last_date_row = db_query_one("""
            SELECT DATE(ts) as ultima_fecha 
            FROM verifications 
            ORDER BY ts DESC 
            LIMIT 1
        """)
        
        if last_date_row and last_date_row["ultima_fecha"]:
            fecha_consulta = last_date_row["ultima_fecha"]
        else:
            fecha_consulta = datetime.date.today()
    
    console_log({"fecha_consulta": fecha_consulta.isoformat()}, "DAILY_ACTIVITY_START")
    
    # Obtener todas las verificaciones del día
    verificaciones = db_query_all("""
        SELECT 
            id, ts, cam_id, image_source, faces_detected, authorized,
            person_label, person_id, distance, threshold, match_crop_url
        FROM verifications 
        WHERE DATE(ts) = :fecha
        ORDER BY person_label, ts
    """, {"fecha": fecha_consulta})
    
    # Agrupar por persona
    personas_actividad = {}
    total_verificaciones = len(verificaciones)
    total_autorizadas = 0
    total_rechazadas = 0
    
    for verificacion in verificaciones:
        person_label = verificacion["person_label"] or "Persona_Desconocida"
        
        if verificacion["authorized"]:
            total_autorizadas += 1
        else:
            total_rechazadas += 1
        
        if person_label not in personas_actividad:
            personas_actividad[person_label] = {
                "intentos": [],
                "crops": [],
                "total_autorizados": 0,
                "total_rechazados": 0
            }
        
        # Agregar intento
        intento_detalle = {
            "id": verificacion["id"],
            "timestamp": verificacion["ts"].isoformat() if verificacion["ts"] else None,
            "cam_id": verificacion["cam_id"],
            "faces_detected": verificacion["faces_detected"],
            "authorized": bool(verificacion["authorized"]),
            "distance": float(verificacion["distance"]) if verificacion["distance"] else None,
            "threshold": float(verificacion["threshold"]),
            "image_source": verificacion["image_source"]
        }
        
        personas_actividad[person_label]["intentos"].append(intento_detalle)
        
        # Agregar crop URL si existe
        if verificacion["match_crop_url"]:
            personas_actividad[person_label]["crops"].append(verificacion["match_crop_url"])
        
        # Contar autorizados/rechazados
        if verificacion["authorized"]:
            personas_actividad[person_label]["total_autorizados"] += 1
        else:
            personas_actividad[person_label]["total_rechazados"] += 1
    
    # Generar respuesta detallada por persona
    actividad_detallada = []
    
    for person_label, actividad in personas_actividad.items():
        # Obtener datos personales de la base de datos
        persona_datos = db_query_one("""
            SELECT cedula, email, telefono, image_url 
            FROM persons 
            WHERE label = :label
        """, {"label": person_label})
        
        # Obtener foto de perfil desde el sistema de archivos si no hay en BD
        foto_perfil = None
        if persona_datos and persona_datos["image_url"]:
            foto_perfil = persona_datos["image_url"]
        elif person_label in DB.labels:
            foto_perfil = DB.photo_url(person_label)
        
        total_intentos = len(actividad["intentos"])
        intentos_autorizados = actividad["total_autorizados"]
        intentos_rechazados = actividad["total_rechazados"]
        tasa_autorizacion = (intentos_autorizados / total_intentos) if total_intentos > 0 else 0.0
        
        # Obtener primer y último intento
        timestamps = [i["timestamp"] for i in actividad["intentos"] if i["timestamp"]]
        primer_intento = min(timestamps) if timestamps else ""
        ultimo_intento = max(timestamps) if timestamps else ""
        
        persona_actividad = DailyPersonActivity(
            nombre=person_label,
            cedula=persona_datos["cedula"] if persona_datos else None,
            email=persona_datos["email"] if persona_datos else None,
            telefono=persona_datos["telefono"] if persona_datos else None,
            foto_perfil_url=foto_perfil,
            total_intentos=total_intentos,
            intentos_autorizados=intentos_autorizados,
            intentos_rechazados=intentos_rechazados,
            tasa_autorizacion=round(tasa_autorizacion, 4),
            primer_intento=primer_intento,
            ultimo_intento=ultimo_intento,
            fotos_intentos=list(set(actividad["crops"])),  # URLs únicas de crops
            detalle_intentos=actividad["intentos"]
        )
        
        actividad_detallada.append(persona_actividad)
    
    # Ordenar por total de intentos (más activos primero)
    actividad_detallada.sort(key=lambda p: p.total_intentos, reverse=True)
    
    result = DailyActivityResponse(
        fecha=fecha_consulta.isoformat(),
        total_personas_activas=len(actividad_detallada),
        total_verificaciones=total_verificaciones,
        total_autorizadas=total_autorizadas,
        total_rechazadas=total_rechazadas,
        actividad_por_persona=actividad_detallada
    )
    
    console_log({
        "fecha": fecha_consulta.isoformat(),
        "personas_activas": len(actividad_detallada),
        "total_verificaciones": total_verificaciones,
        "personas": [p.nombre for p in actividad_detallada[:5]]  # Solo primeros 5
    }, "DAILY_ACTIVITY_COMPLETE")
    
    return result

# ===================== Main =====================
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", "8000")), reload=False)
