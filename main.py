import os
import re
import json
import time
import uuid
import hmac
import hashlib
import isodate
import redis
import logging
import stripe
import random
#import asyncpg
import asyncio
from collections import defaultdict
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta


import requests
from fastapi import FastAPI, HTTPException, Request, Depends, Header
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, HttpUrl, validator
from prometheus_client import Counter, Histogram, Gauge, generate_latest

from openai import OpenAI
from googleapiclient.discovery import build

def hash_api_key(api_key: str) -> str:
    """Hash API key pour les métriques (privacy)"""
    return hashlib.sha256(api_key.encode()).hexdigest()[:8]
    
# =========================
# 💵 Billing système complet
# =========================

def estimate_cost_usd(tokens_used: int) -> float:
    # Estimation moyenne : $0.0003 / 1K tokens
    return round((tokens_used / 1000.0) * 0.0003, 6)

def create_stripe_usage_record(api_key: str, amount_usd: float, meta: Dict[str, Any]):
    """Crée un usage record Stripe pour facturation"""
    if not STRIPE_API_KEY:
        logger.debug(f"Stripe non configuré, skip billing pour {api_key}")
        return
    
    try:
        # Recherche du customer/subscription par API key
        # Note: En production, mapper api_key -> stripe_customer_id
        customer_id = get_stripe_customer_for_api_key(api_key)
        if not customer_id:
            logger.warning(f"Customer Stripe introuvable pour API key {hash_api_key(api_key)}")
            return
        
        # Créer usage record pour facturation à l'usage
        usage_record = stripe.UsageRecord.create(
            subscription_item='si_xxxxx',  # À mapper depuis customer
            quantity=int(amount_usd * 1000000),  # Micropaiements en micro-USD
            timestamp=int(time.time()),
            action='increment'
        )
        
        logger.info(f"Stripe usage record créé: {usage_record.id} pour {hash_api_key(api_key)}")
        
    except stripe.error.StripeError as e:
        logger.error ("message d'erreur")

# =========================
# 🔑 ENV & Configuration
# =========================

APP_NAME = "Videolyze API"
APP_VERSION = "1.0.0"
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")  # development, staging, production

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# API keys autorisées (séparées par virgules) ex: "key1,key2,key3"
API_KEYS_ALLOWED = {k.strip() for k in os.getenv("VIDEOLYZE_API_KEYS", "").split(",") if k.strip()}

# Stripe configuration (production billing)
STRIPE_API_KEY = os.getenv("STRIPE_API_KEY", "")
if STRIPE_API_KEY:
    stripe.api_key = STRIPE_API_KEY

# PayPal (alternative billing)
PAYPAL_CLIENT_ID = os.getenv("PAYPAL_CLIENT_ID", "")
PAYPAL_CLIENT_SECRET = os.getenv("PAYPAL_CLIENT_SECRET", "")

# Redis configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Postgres configuration for audit
POSTGRES_DSN = os.getenv("POSTGRES_DSN", "")

# Stripe customer mapping depuis variables d'environnement
# Format: "api_key1:customer_id1:subscription_item1,api_key2:customer_id2:subscription_item2"
STRIPE_MAPPING_RAW = os.getenv("STRIPE_API_MAPPING", "")

def parse_stripe_mapping() -> Dict[str, Dict[str, str]]:
    """Parse le mapping Stripe depuis les variables d'environnement"""
    mapping = {}
    if not STRIPE_MAPPING_RAW:
        return mapping
    
    for entry in STRIPE_MAPPING_RAW.split(","):
        parts = entry.strip().split(":")
        if len(parts) == 3:
            api_key, customer_id, subscription_item = parts
            mapping[api_key] = {
                "customer_id": customer_id,
                "subscription_item": subscription_item
            }
    return mapping

API_KEY_TO_STRIPE = parse_stripe_mapping()

# Circuit breaker pour OpenAI
class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func, *args, **kwargs):
        if self.state == "OPEN":
            if time.time() - self.last_failure_time < self.recovery_timeout:
                raise Exception("Circuit breaker is OPEN - OpenAI service unavailable")
            else:
                self.state = "HALF_OPEN"
        
        try:
            result = func(*args, **kwargs)
            self.reset()
            return result
        except Exception as e:
            self.record_failure()
            raise
    
    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.warning(f"Circuit breaker OPEN - OpenAI failures: {self.failure_count}")
    
    def reset(self):
        self.failure_count = 0
        self.state = "CLOSED"

openai_circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)

# CORS configuration based on environment
if ENVIRONMENT == "production":
    ALLOWED_ORIGINS = [origin.strip() for origin in os.getenv("CORS_ALLOWED_ORIGINS", "").split(",") if origin.strip()]
    if not ALLOWED_ORIGINS:
        ALLOWED_ORIGINS = ["https://yourdomain.com"]  # Default secure origin
else:
    ALLOWED_ORIGINS = ["*"]  # Development only

# Logging configuration
logging.basicConfig(
    level=logging.INFO if ENVIRONMENT != "production" else logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("videolyze")

# OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# =========================
# Prometheus Metrics
# =========================

# Compteurs métier
api_calls_total = Counter('videolyze_api_calls_total', 'Total API calls', ['endpoint', 'api_key_hash', 'status'])
tokens_used_total = Counter('videolyze_tokens_used_total', 'Total OpenAI tokens used', ['api_key_hash'])
cache_hits_total = Counter('videolyze_cache_hits_total', 'Cache hit/miss', ['status'])
request_duration = Histogram('videolyze_request_duration_seconds', 'Request duration', ['endpoint'])
openai_errors_total = Counter('videolyze_openai_errors_total', 'OpenAI API errors', ['error_type'])

# Gauges systeme
redis_connected = Gauge('videolyze_redis_connected', 'Redis connection status (1=connected, 0=disconnected)')
active_api_keys = Gauge('videolyze_active_api_keys', 'Number of active API keys')

# =========================
# Postgres connection pour audit (initialisation async)
# =========================

pg_pool = None

async def init_pg_pool():
    global pg_pool
    if not POSTGRES_DSN:
        logger.info("Postgres DSN non configuré, audit désactivé")
        return
    try:
        pg_pool = await asyncpg.create_pool(dsn=POSTGRES_DSN, min_size=1, max_size=5)
        logger.info("Postgres pool initialized for audit")
    except Exception as e:
        logger.error(f"Failed to initialize Postgres pool: {e}")

# Initialisation différée - sera appelée au premier usage
pg_init_task = None

def ensure_pg_initialized():
    global pg_init_task
    if POSTGRES_DSN and pg_init_task is None:
        pg_init_task = asyncio.create_task(init_pg_pool())

# =========================
# 🔐 Stripe customer mapping
# =========================

def get_stripe_customer_for_api_key(api_key: str) -> Optional[Dict[str, str]]:
    """Retourne les infos Stripe (customer_id + subscription_item) pour une API key"""
    return API_KEY_TO_STRIPE.get(api_key)

# =========================
# Retry helper avec backoff
# =========================

def retry_with_backoff(func, max_retries=3, base_delay=2, *args, **kwargs):
    """Retry avec backoff exponentiel + jitter + circuit breaker"""
    def protected_call():
        return openai_circuit_breaker.call(func, *args, **kwargs)
    
    for attempt in range(max_retries):
        try:
            return protected_call()
        except Exception as e:
            if "Circuit breaker is OPEN" in str(e):
                # Circuit breaker ouvert, pas de retry
                logger.error("OpenAI circuit breaker OPEN - skip retries")
                raise
            
            if attempt == max_retries - 1:
                raise
            
            sleep_time = base_delay * (2 ** attempt) + random.uniform(0, 1)
            logger.warning(f"Retry {attempt+1}/{max_retries} après erreur: {e} (sleep={sleep_time:.2f}s)")
            time.sleep(sleep_time)

# =========================
# Audit persistant
# =========================

async def persist_billing_event(api_key: str, amount_usd: float, meta: Dict[str, Any]):
    """Persistance dans Postgres pour audit facturation"""
    if not pg_pool:
        return
    try:
        async with pg_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO billing_audit (api_key_hash, amount_usd, meta, created_at)
                VALUES ($1, $2, $3, NOW())
            """, hash_api_key(api_key), amount_usd, json.dumps(meta))
    except Exception as e:
        logger.error(f"Failed to persist billing event: {e}")

# =========================
# Redis client + Fallback mémoire
# =========================

try:
    redis_client = redis.from_url(REDIS_URL, decode_responses=True, socket_timeout=5, socket_connect_timeout=5)
    # Test connection
    redis_client.ping()
    logger.info(f"Redis connected to {REDIS_URL}")
    redis_connected.set(1)
except Exception as e:
    logger.warning(f"Redis connection failed: {e}")
    redis_client = None
    redis_connected.set(0)

# Fallback mémoire local (limité par IP si Redis down)
memory_cache: Dict[str, Any] = {}
memory_rate_limits: Dict[str, List[datetime]] = {}
MAX_FALLBACK_ENTRIES = 1000  # Limite mémoire globale

def cleanup_memory_fallback():
    """Nettoie le cache mémoire si trop volumineux"""
    if len(memory_cache) > MAX_FALLBACK_ENTRIES:
        # Garde seulement les 500 entrées les plus récentes
        sorted_items = sorted(memory_cache.items(), key=lambda x: x[1].get('timestamp', 0))
        memory_cache.clear()
        for k, v in sorted_items[-500:]:
            memory_cache[k] = v
    
    # Nettoie les rate limits expirés
    now = datetime.utcnow()
    for key in list(memory_rate_limits.keys()):
        memory_rate_limits[key] = [ts for ts in memory_rate_limits[key] if now - ts < timedelta(hours=1)]
        if not memory_rate_limits[key]:
            del memory_rate_limits[key]

# =========================
# ⚙️ App & CORS
# =========================

app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description="Videolyze est un service d'intelligence vidéo (YouTube/TikTok) pour agents. "
                "Endpoints: /analyze/video, /analyze/multi, /badge/product.",
    docs_url="/docs" if ENVIRONMENT != "production" else None,
    redoc_url="/redoc" if ENVIRONMENT != "production" else None
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False if ENVIRONMENT == "production" else True,
    allow_methods=["GET", "POST"] if ENVIRONMENT == "production" else ["*"],
    allow_headers=["Content-Type", "x-api-key"] if ENVIRONMENT == "production" else ["*"],
)

# =========================
# Redis helpers avec fallback mémoire
# =========================

MAX_REQUESTS_PER_HOUR = int(os.getenv("MAX_REQ_PER_HOUR", "200"))

def get_from_cache(key: str) -> Optional[Any]:
    """Récupère une valeur du cache Redis avec fallback mémoire"""
    if redis_client:
        try:
            data = redis_client.get(f"cache:{key}")
            if data:
                cache_hits_total.labels(status="hit").inc()
                return json.loads(data)
        except Exception as e:
            logger.error(f"Redis cache get error: {e}")
    
    # Fallback mémoire
    if key in memory_cache:
        cache_hits_total.labels(status="memory_hit").inc()
        return memory_cache[key]
    
    cache_hits_total.labels(status="miss").inc()
    return None

def set_cache(key: str, value: Any, ttl: int = 3600):
    """Stocke une valeur dans le cache Redis avec fallback mémoire"""
    if redis_client:
        try:
            redis_client.setex(f"cache:{key}", ttl, json.dumps(value, default=str))
            return
        except Exception as e:
            logger.error(f"Redis cache set error: {e}")
    
    # Fallback mémoire avec timestamp pour TTL
    cleanup_memory_fallback()
    memory_cache[key] = {
        "data": value,
        "timestamp": time.time(),
        "expires_at": time.time() + ttl
    }

def check_rate_limit_redis(api_key: str):
    """Vérifie le rate limit avec Redis + fallback mémoire"""
    if redis_client:
        try:
            now = datetime.utcnow()
            window_key = f"rate_limit:{api_key}:{now.hour}"
            
            # Utilise INCR atomique avec EXPIRE
            count = redis_client.incr(window_key)
            if count == 1:
                redis_client.expire(window_key, 3600)  # 1 heure
            
            if count > MAX_REQUESTS_PER_HOUR:
                raise HTTPException(status_code=429, detail="Trop de requêtes. Limite horaire atteinte.")
            return
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Redis rate limit error: {e}")
    
    # Fallback mémoire avec limite par IP
    cleanup_memory_fallback()
    now = datetime.utcnow()
    window_start = now - timedelta(hours=1)
    
    memory_rate_limits.setdefault(api_key, [])
    memory_rate_limits[api_key] = [ts for ts in memory_rate_limits[api_key] if ts > window_start]
    
    if len(memory_rate_limits[api_key]) >= MAX_REQUESTS_PER_HOUR:
        raise HTTPException(status_code=429, detail="Trop de requêtes. Limite horaire atteinte (fallback).")
    
    memory_rate_limits[api_key].append(now)

def incr_usage_redis(api_key: str, tokens: int = 0):
    """Incrémente les compteurs d'usage avec Redis"""
    if redis_client:
        try:
            pipe = redis_client.pipeline()
            pipe.hincrby(f"usage:{api_key}", "calls", 1)
            pipe.hincrby(f"usage:{api_key}", "tokens", max(tokens, 0))
            pipe.expire(f"usage:{api_key}", 86400 * 30)  # 30 jours
            pipe.execute()
        except Exception as e:
            logger.error(f"Redis usage tracking error: {e}")
    
    # Métriques Prometheus
    api_key_hash = hash_api_key(api_key)
    tokens_used_total.labels(api_key_hash=api_key_hash).inc(tokens)

# =========================
# 🧱 Schémas Pydantic
# =========================

class AnalyzeVideoBody(BaseModel):
    url: HttpUrl
    lang: str = Field(default="fr", description="Langue de sortie (ex: fr, en)")
    max_comments: int = Field(default=30, ge=0, le=100)

    @validator('url')
    def validate_url(cls, v):
        url_str = str(v)
        if len(url_str) > 2000:
            raise ValueError("URL trop longue (>2000 caractères)")
        if not re.match(r'https?://', url_str):
            raise ValueError("URL doit commencer par http:// ou https://")
        return v

class AnalyzeMultiBody(BaseModel):
    urls: List[HttpUrl] = Field(min_items=1, max_items=10)
    lang: str = Field(default="fr")
    max_comments: int = Field(default=20, ge=0, le=100)

class BadgeProductBody(BaseModel):
    product_name: str = Field(..., description="Nom court du produit (ex: Dyson V8)")
    video_urls: List[HttpUrl] = Field(min_items=1, max_items=20)
    lang: str = Field(default="fr")
    max_comments: int = Field(default=20, ge=0, le=100)

class SentimentResult(BaseModel):
    label: str
    score: float
    counts: Dict[str, int]

class SummaryResult(BaseModel):
    bullets: List[str]

class FlagsResult(BaseModel):
    items: List[str] = Field(default_factory=list)

class VideoMeta(BaseModel):
    id: str
    title: str
    creator: str
    publishedAt: Optional[str] = None
    duration_min: Optional[int] = None
    metrics: Dict[str, Optional[str]] = Field(default_factory=dict)
    platform: str

class AnalyzeResponse(BaseModel):
    request_id: str
    platform: str
    video: VideoMeta
    summary: SummaryResult
    sentiment: SentimentResult
    flags: FlagsResult
    tokens_used: int
    runtime_ms: int
    cache: str = Field(default="MISS")
    cost_usd_estimated: float

class AnalyzeMultiItem(BaseModel):
    url: str
    ok: bool
    data: Optional[AnalyzeResponse] = None
    error: Optional[str] = None

class AnalyzeMultiResponse(BaseModel):
    request_id: str
    results: List[AnalyzeMultiItem]
    aggregate: Optional[Dict[str, Any]] = None
    runtime_ms: int

class BadgeProductResponse(BaseModel):
    request_id: str
    product_name: str
    analyses: List[AnalyzeMultiItem]
    badge: Dict[str, Any]
    runtime_ms: int

# =========================
# 🔐 Auth & Rate limiting
# =========================

def require_api_key(x_api_key: Optional[str] = Header(default=None)):
    # Si aucune clé configurée côté serveur, on passe (dev only)
    if not API_KEYS_ALLOWED:
        if ENVIRONMENT == "production":
            raise HTTPException(status_code=500, detail="API keys non configurées")
        return "dev-open"
    if not x_api_key or x_api_key not in API_KEYS_ALLOWED:
        raise HTTPException(status_code=401, detail="API key invalide ou manquante (header: x-api-key)")
    return x_api_key

# =========================
# 🎬 Utils plateformes
# =========================

def extract_youtube_id(url: str) -> str:
    patterns = [
        r"(?:v=)([A-Za-z0-9_-]{11})",
        r"youtu\.be/([A-Za-z0-9_-]{11})"
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    raise ValueError("Impossible d'extraire l'ID YouTube")

def parse_iso_duration_minutes(iso_duration: str) -> int:
    try:
        td = isodate.parse_duration(iso_duration)
        return int(td.total_seconds() // 60)
    except Exception:
        return 0  # Durée inconnue si parsing échoue

def get_youtube_metadata(video_id: str) -> VideoMeta:
    if not YOUTUBE_API_KEY:
        raise HTTPException(status_code=500, detail="YOUTUBE_API_KEY manquante")
    
    try:
        youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
        resp = youtube.videos().list(part="snippet,statistics,contentDetails", id=video_id).execute()
        items = resp.get("items", [])
        if not items:
            raise HTTPException(status_code=404, detail="Vidéo YouTube introuvable")

        it = items[0]
        duration = parse_iso_duration_minutes(it["contentDetails"]["duration"])

        return VideoMeta(
            id=video_id,
            title=it["snippet"]["title"],
            creator=it["snippet"]["channelTitle"],
            publishedAt=it["snippet"]["publishedAt"],
            duration_min=duration,
            metrics={
                "views": it["statistics"].get("viewCount"),
                "likes": it["statistics"].get("likeCount")
            },
            platform="youtube"
        )
    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=f"Erreur YouTube API: {str(e)}")

def get_youtube_comments(video_id: str, max_comments: int = 30) -> List[str]:
    if not YOUTUBE_API_KEY:
        return []
    
    try:
        youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
        req = youtube.commentThreads().list(
            part="snippet", videoId=video_id,
            maxResults=min(max_comments, 100), textFormat="plainText"
        )
        resp = req.execute()
        comments = []
        for item in resp.get("items", []):
            text = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(text)
        return comments
    except Exception:
        return []  # Si les commentaires sont désactivés ou erreur

def get_tiktok_oembed(url: str) -> Dict[str, Any]:
    try:
        r = requests.get("https://www.tiktok.com/oembed", params={"url": url}, timeout=15)
        if r.status_code != 200:
            raise HTTPException(status_code=400, detail="Impossible de récupérer les métadonnées TikTok")
        return r.json()
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Erreur réseau TikTok: {str(e)}")

def get_tiktok_metadata(url: str) -> VideoMeta:
    data = get_tiktok_oembed(url)
    return VideoMeta(
        id=data.get("author_unique_id", "") or url,
        title=data.get("title", ""),
        creator=data.get("author_name", ""),
        publishedAt=None,
        duration_min=None,
        metrics={},
        platform="tiktok"
    )

# =========================
# 🧪 OpenAI helpers
# =========================

def oa_summarize(text: str, lang: str) -> (SummaryResult, FlagsResult, int):
    """
    Retourne SummaryResult + FlagsResult + tokens_used
    """
    if not openai_client:
        logger.error("OpenAI client non configuré")
        raise HTTPException(status_code=500, detail="OpenAI client non configuré")
        
    prompt = f"""
Tu es un analyste média spécialisé dans l'analyse de commentaires vidéo.
Objectif : extraire 5 réactions dominantes ET détecter des signaux d'alerte ("red flags").

Réponds UNIQUEMENT en JSON strict :
{{
  "bullets": ["réaction 1","réaction 2","réaction 3","réaction 4","réaction 5"],
  "flags": ["flag_1","flag_2"]
}}

CONTENU A ANALYSER (langue cible: {lang}) :
{text}
"""
    
    try:
        res = retry_with_backoff(
            openai_client.chat.completions.create,
            max_retries=3,
            base_delay=2,
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            timeout=30
        )
        
        tokens_used = getattr(res, "usage", None).total_tokens if getattr(res, "usage", None) else 0
        content = res.choices[0].message.content.strip()

        # Validation et extraction JSON plus robuste
        if "{" not in content or "}" not in content:
            raise ValueError("Réponse OpenAI ne contient pas de JSON valide")
            
        start_idx = content.index("{")
        end_idx = content.rindex("}") + 1
        json_content = content[start_idx:end_idx]
        
        data = json.loads(json_content)
        bullets = data.get("bullets", [])[:5]
        flags = data.get("flags", [])
        
        # S'assurer qu'on a bien 5 bullets
        while len(bullets) < 5:
            bullets.append("")
            
        return SummaryResult(bullets=bullets), FlagsResult(items=flags), tokens_used
        
    except json.JSONDecodeError as e:
        logger.error(f"OpenAI JSON decode error: {e}")
        openai_errors_total.labels(error_type="json_decode").inc()
        return fallback_summary(), 0
    except Exception as e:
        logger.error(f"OpenAI summarize error: {e}")
        openai_errors_total.labels(error_type="api_error").inc()
        return fallback_summary(), 0

def oa_sentiment(comments: List[str]) -> (SentimentResult, int):
    if not comments:
        return SentimentResult(label="neutral", score=0.5, counts={"pos": 0, "neu": 1, "neg": 0}), 0

    if not openai_client:
        logger.warning("OpenAI indisponible, utilisation du fallback heuristique")
        return fallback_sentiment_analysis(comments), 0

    sample = comments[:10]
    comments_text = "\n".join(sample)
    prompt = f"""
Analyse ces commentaires et réponds UNIQUEMENT avec un JSON valide :
{{
  "counts": {{"pos": 2, "neu": 1, "neg": 3}},
  "label": "negative",
  "score": 0.67
}}

Commentaires :
{comments_text}
"""
    
    try:
        res = retry_with_backoff(
            openai_client.chat.completions.create,
            max_retries=3,
            base_delay=2,
            model="gpt-4o-mini",
            messages=[{"role":"user","content": prompt}],
            temperature=0.1,
            timeout=30
        )
        
        tokens_used = getattr(res, "usage", None).total_tokens if getattr(res, "usage", None) else 0
        content = res.choices[0].message.content.strip()

        if "{" not in content or "}" not in content:
            raise ValueError("Réponse OpenAI ne contient pas de JSON valide")
            
        start_idx = content.index("{")
        end_idx = content.rindex("}") + 1
        json_content = content[start_idx:end_idx]
        
        data = json.loads(json_content)
        counts = data.get("counts", {"pos": 0, "neu": 1, "neg": 0})
        label = data.get("label", "neutral")
        score = float(data.get("score", 0.5))
        
        return SentimentResult(label=label, score=score, counts=counts), tokens_used
        
    except json.JSONDecodeError as e:
        logger.error(f"OpenAI sentiment JSON decode error: {e}")
        openai_errors_total.labels(error_type="json_decode").inc()
        return fallback_sentiment_analysis(sample), 0
    except Exception as e:
        logger.error(f"OpenAI sentiment error: {e}")
        openai_errors_total.labels(error_type="api_error").inc()
        return fallback_sentiment_analysis(sample), 0

def fallback_summary() -> (SummaryResult, FlagsResult):
    """Summary de secours quand OpenAI échoue"""
    return SummaryResult(bullets=[
        "Analyse indisponible temporairement",
        "Service OpenAI inaccessible",
        "Réessayez dans quelques minutes",
        "",
        ""
    ]), FlagsResult(items=["service_unavailable"])

def fallback_sentiment_analysis(comments: List[str]) -> SentimentResult:
    """Analyse de sentiment heuristique de secours"""
    neg_words = ["colère", "injustice", "arnaque", "plaintes", "déçu", "mauvais", "hate", "awful", "terrible"]
    pos_words = ["génial", "parfait", "excellent", "love", "amazing", "great", "parfait"]
    
    neg_count = sum(1 for c in comments for w in neg_words if w.lower() in c.lower())
    pos_count = sum(1 for c in comments for w in pos_words if w.lower() in c.lower())
    neu_count = max(1, len(comments) - neg_count - pos_count)
    
    total = neg_count + pos_count + neu_count
    if neg_count > pos_count and neg_count >= len(comments) // 3:
        return SentimentResult(label="negative", score=0.3,
                             counts={"pos": pos_count, "neu": neu_count, "neg": neg_count})
    elif pos_count > neg_count and pos_count >= len(comments) // 3:
        return SentimentResult(label="positive", score=0.7,
                             counts={"pos": pos_count, "neu": neu_count, "neg": neg_count})
    else:
        return SentimentResult(label="neutral", score=0.5,
                             counts={"pos": pos_count, "neu": neu_count, "neg": neg_count})

# =========================
# 💵 Billing système complet
# =========================

def estimate_cost_usd(tokens_used: int) -> float:
    # Estimation moyenne : $0.0003 / 1K tokens
    return round((tokens_used / 1000.0) * 0.0003, 6)

def create_stripe_usage_record(api_key: str, amount_usd: float, meta: Dict[str, Any]):
    """Crée un usage record Stripe fiable"""
    if not STRIPE_API_KEY:
        logger.debug("Stripe non configuré → skip billing")
        return
    
    try:
        customer_info = get_stripe_customer_for_api_key(api_key)
        if not customer_info:
            logger.warning(f"Stripe mapping manquant pour API key {hash_api_key(api_key)}")
            return

        usage_record = stripe.UsageRecord.create(
            subscription_item=customer_info["subscription_item"],
            quantity=int(amount_usd * 1_000_000),  # micro-USD
            timestamp=int(time.time()),
            action="increment"
        )

        logger.info(f"Stripe usage record OK: {usage_record.id} pour {hash_api_key(api_key)}")

    except stripe.error.StripeError as e:
        logger.error(f"Stripe billing error: {str(e)}")
    except Exception as e:
        logger.error(f"Erreur inconnue billing: {str(e)}")

def record_billing_event(api_key: str, amount_usd: float, meta: Dict[str, Any]):
    """
    Enregistre un événement de facturation avec les providers configurés
    """
    # Log structuré pour audit
    logger.info(f"Billing event: api_key={hash_api_key(api_key)} amount={amount_usd} meta={meta}")
    
    # Stripe en priorité
    if ENVIRONMENT == "production" and STRIPE_API_KEY:
        create_stripe_usage_record(api_key, amount_usd, meta)
    elif PAYPAL_CLIENT_ID:
        create_paypal_invoice(api_key, amount_usd, meta)
    
    # Stockage Redis pour historique
    if redis_client:
        try:
            billing_record = {
                "timestamp": datetime.utcnow().isoformat(),
                "amount_usd": amount_usd,
                "meta": meta
            }
            redis_client.lpush(f"billing:{api_key}", json.dumps(billing_record))
            redis_client.ltrim(f"billing:{api_key}", 0, 999)  # Garde les 1000 derniers
            redis_client.expire(f"billing:{api_key}", 86400 * 90)  # 90 jours
        except Exception as e:
            logger.error(f"Erreur stockage billing Redis: {e}")
    
    # Audit persistant Postgres
    if pg_pool:
        asyncio.create_task(persist_billing_event(api_key, amount_usd, meta))

# =========================
# 🧰 Helpers métier
# =========================

def analyze_single_url(url: str, lang: str, max_comments: int) -> (AnalyzeResponse, int):
    t0 = time.time()
    tokens_total = 0

    # Cache Redis
    cache_key = f"analyze:{url}:{lang}:{max_comments}"
    cached = get_from_cache(cache_key)
    if cached:
        elapsed = int((time.time() - t0) * 1000)
        cached["cache"] = "HIT"
        cached["runtime_ms"] = elapsed
        return AnalyzeResponse(**cached), 0

    # Router
    if "youtube.com" in url or "youtu.be" in url:
        vid = extract_youtube_id(url)
        meta = get_youtube_metadata(vid)
        if meta.duration_min and meta.duration_min > 35:
            raise HTTPException(status_code=400, detail=f"Vidéo trop longue ({meta.duration_min} min). Limite actuelle : 35 minutes.")

        comments = get_youtube_comments(vid, max_comments=max_comments)
        # Construire le texte d'analyse
        if comments and len(comments) >= 5:
            selected = comments[: min(15, len(comments))]
            text_to_analyze = f"VIDÉO: {meta.title}\nCHAÎNE: {meta.creator}\n\nCOMMENTAIRES:\n- " + "\n- ".join(selected)
        else:
            text_to_analyze = f"VIDÉO: {meta.title}\nCHAÎNE: {meta.creator}\nNOTE: Peu de commentaires disponibles."

        summary, flags, tk1 = oa_summarize(text_to_analyze, lang)
        senti, tk2 = oa_sentiment(comments)
        tokens_total = tk1 + tk2

        result = AnalyzeResponse(
            request_id=str(uuid.uuid4()),
            platform=meta.platform,
            video=meta,
            summary=summary,
            sentiment=senti,
            flags=flags,
            tokens_used=tokens_total,
            runtime_ms=int((time.time() - t0) * 1000),
            cache="MISS",
            cost_usd_estimated=estimate_cost_usd(tokens_total)
        )
        
        # Cache pendant 1 heure
        set_cache(cache_key, result.dict(), ttl=3600)
        return result, tokens_total

    elif "tiktok.com" in url:
        meta = get_tiktok_metadata(url)
        text_to_analyze = f"VIDÉO TIKTOK: {meta.title} — {meta.creator}"
        summary, flags, tk1 = oa_summarize(text_to_analyze, lang)
        # TikTok comments: non standard → neutre par défaut
        senti = SentimentResult(label="neutral", score=0.5, counts={"pos": 0, "neu": 1, "neg": 0})
        tokens_total = tk1

        result = AnalyzeResponse(
            request_id=str(uuid.uuid4()),
            platform=meta.platform,
            video=meta,
            summary=summary,
            sentiment=senti,
            flags=flags,
            tokens_used=tokens_total,
            runtime_ms=int((time.time() - t0) * 1000),
            cache="MISS",
            cost_usd_estimated=estimate_cost_usd(tokens_total)
        )
        
        set_cache(cache_key, result.dict(), ttl=3600)
        return result, tokens_total

    else:
        raise HTTPException(status_code=400, detail="Plateforme non supportée (YouTube ou TikTok uniquement).")

def aggregate_multi(items: List[AnalyzeResponse]) -> Dict[str, Any]:
    if not items:
        return {}
    # Agrégation simple : moyenne des sentiments, concat flags uniques, top mots-clés (sur bullets)
    avg_score = sum(i.sentiment.score for i in items) / len(items)
    labels = {"positive":0,"neutral":0,"negative":0}
    for i in items:
        if i.sentiment.label in labels:
            labels[i.sentiment.label] += 1
    flags = []
    for i in items:
        flags += i.flags.items
    unique_flags = sorted(set(flags))
    bullets = []
    for i in items:
        bullets += i.summary.bullets
    # On garde les 10 premières bullets pour rester concis
    bullets = [b for b in bullets if b][:10]
    return {
        "avg_sentiment_score": round(avg_score, 3),
        "labels_count": labels,
        "unique_flags": unique_flags,
        "bullets_sample": bullets
    }

def compute_product_badge(analyses: List[AnalyzeResponse]) -> Dict[str, Any]:
    """
    Heuristique simple :
      - Base score = moyenne des (1 - |0.5 - score|*2) → valorise la neutralité/positif
      - Malus si beaucoup de flags
      - Label qualitatif
    """
    if not analyses:
        return {"score": 0.0, "label": "insufficient_data", "reasons": ["Aucune vidéo analysée"]}

    # score proximité du neutre/positif
    base_scores = []
    flags_count = 0
    for a in analyses:
        base = max(0.0, 1.0 - abs(0.5 - a.sentiment.score) * 2.0)
        # favorise légèrement les positifs
        if a.sentiment.label == "positive":
            base = min(1.0, base + 0.05)
        base_scores.append(base)
        flags_count += len(a.flags.items)

    score = sum(base_scores) / len(base_scores)
    # malus flags
    score = max(0.0, score - min(0.2, flags_count * 0.02))
    label = "low"
    if score >= 0.75: label = "high"
    elif score >= 0.5: label = "medium"

    return {
        "score": round(score, 3),
        "label": label,
        "reasons": [
            f"{len(analyses)} vidéos analysées",
            f"{flags_count} signaux d'alerte détectés"
        ]
    }

# =========================
# 🌐 Endpoints
# =========================

@app.get("/health")
def health():
    redis_status = "connected" if redis_client else "unavailable"
    redis_connected.set(1 if redis_client else 0)
    active_api_keys.set(len(API_KEYS_ALLOWED))
    
    return {
        "ok": True, 
        "app": APP_NAME, 
        "version": APP_VERSION, 
        "time": datetime.utcnow().isoformat(),
        "environment": ENVIRONMENT,
        "redis": redis_status,
        "openai": "configured" if openai_client else "missing",
        "billing": "stripe" if STRIPE_API_KEY else ("paypal" if PAYPAL_CLIENT_ID else "stub")
    }

@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    """Endpoint Prometheus metrics"""
    return generate_latest()

# ---- UI minimale pour test (development/staging uniquement)
@app.get("/", response_class=HTMLResponse)
def home():
    if ENVIRONMENT == "production":
        raise HTTPException(status_code=404, detail="Not Found")
    
    return f"""
    <html>
      <head><title>{APP_NAME} {APP_VERSION}</title></head>
      <body style="font-family: system-ui, sans-serif; max-width: 720px; margin: 2rem auto;">
        <h2>🎥 {APP_NAME} — Playground ({ENVIRONMENT})</h2>
        <p>Testez rapidement l'API (YouTube/TikTok) :</p>
        <input id="url" type="text" style="width:100%;padding:8px" placeholder="https://..."><br><br>
        <button onclick="analyze()">/analyze/video</button>
        <pre id="result" style="background:#f5f5f5;padding:1rem;margin-top:1rem;white-space:pre-wrap;"></pre>
        <script>
        async function analyze() {{
          const url = document.getElementById("url").value;
          const res = await fetch('/analyze/video', {{
            method: 'POST',
            headers: {{'Content-Type':'application/json','x-api-key':'your-api-key-here'}},
            body: JSON.stringify({{url: url, lang:"fr", max_comments:10}})
          }});
          const data = await res.json();
          document.getElementById("result").textContent = JSON.stringify(data,null,2);
        }}
        </script>
        <hr/>
        <p style="color:#666">Remplacez 'your-api-key-here' par votre vraie clé API.</p>
        <p style="color:#999;font-size:0.9em">Environment: {ENVIRONMENT}</p>
        <p><a href="/metrics" target="_blank">📊 Métriques Prometheus</a> | <a href="/health">🔍 Health Check</a></p>
      </body>
    </html>
    """

# ---- Analyze 1 vidéo
@app.post("/analyze/video", response_model=AnalyzeResponse)
def analyze_video(body: AnalyzeVideoBody, api_key: str = Depends(require_api_key)):
    with request_duration.labels(endpoint="analyze_video").time():
        t0 = time.time()
        api_key_hash = hash_api_key(api_key)
        
        try:
            check_rate_limit_redis(api_key)
            result, tokens = analyze_single_url(str(body.url), body.lang, body.max_comments)
            
            incr_usage_redis(api_key, tokens)
            record_billing_event(api_key, result.cost_usd_estimated, {"endpoint": "analyze/video"})
            
            # runtime recalcul
            res = result.dict()
            res["runtime_ms"] = int((time.time() - t0) * 1000)
            
            api_calls_total.labels(endpoint="analyze_video", api_key_hash=api_key_hash, status="success").inc()
            return AnalyzeResponse(**res)
            
        except HTTPException as e:
            api_calls_total.labels(endpoint="analyze_video", api_key_hash=api_key_hash, status="error").inc()
            raise
        except Exception as e:
            api_calls_total.labels(endpoint="analyze_video", api_key_hash=api_key_hash, status="error").inc()
            logger.error(f"Erreur inattendue analyze_video: {e}")
            raise HTTPException(status_code=500, detail="Erreur interne du serveur")

# ---- Analyze multi vidéos
@app.post("/analyze/multi", response_model=AnalyzeMultiResponse)
def analyze_multi(body: AnalyzeMultiBody, api_key: str = Depends(require_api_key)):
    with request_duration.labels(endpoint="analyze_multi").time():
        t0 = time.time()
        api_key_hash = hash_api_key(api_key)
        
        try:
            check_rate_limit_redis(api_key)

            items: List[AnalyzeMultiItem] = []
            collected_ok: List[AnalyzeResponse] = []
            tokens_total = 0

            for u in body.urls:
                try:
                    data, tk = analyze_single_url(str(u), body.lang, body.max_comments)
                    items.append(AnalyzeMultiItem(url=str(u), ok=True, data=data))
                    collected_ok.append(data)
                    tokens_total += tk
                except HTTPException as he:
                    items.append(AnalyzeMultiItem(url=str(u), ok=False, error=he.detail))
                except Exception as e:
                    items.append(AnalyzeMultiItem(url=str(u), ok=False, error=str(e)))

            aggregate = aggregate_multi(collected_ok) if collected_ok else None
            incr_usage_redis(api_key, tokens_total)

            elapsed = int((time.time() - t0) * 1000)
            api_calls_total.labels(endpoint="analyze_multi", api_key_hash=api_key_hash, status="success").inc()
            
            return AnalyzeMultiResponse(
                request_id=str(uuid.uuid4()),
                results=items,
                aggregate=aggregate,
                runtime_ms=elapsed
            )
            
        except HTTPException as e:
            api_calls_total.labels(endpoint="analyze_multi", api_key_hash=api_key_hash, status="error").inc()
            raise
        except Exception as e:
            api_calls_total.labels(endpoint="analyze_multi", api_key_hash=api_key_hash, status="error").inc()
            logger.error(f"Erreur inattendue analyze_multi: {e}")
            raise HTTPException(status_code=500, detail="Erreur interne du serveur")

# ---- Badge produit (pour e-commerce)
@app.post("/badge/product", response_model=BadgeProductResponse)
def badge_product(body: BadgeProductBody, api_key: str = Depends(require_api_key)):
    with request_duration.labels(endpoint="badge_product").time():
        t0 = time.time()
        api_key_hash = hash_api_key(api_key)
        
        try:
            check_rate_limit_redis(api_key)

            items: List[AnalyzeMultiItem] = []
            analyses: List[AnalyzeResponse] = []
            tokens_total = 0

            for u in body.video_urls:
                try:
                    data, tk = analyze_single_url(str(u), body.lang, body.max_comments)
                    items.append(AnalyzeMultiItem(url=str(u), ok=True, data=data))
                    analyses.append(data)
                    tokens_total += tk
                except HTTPException as he:
                    items.append(AnalyzeMultiItem(url=str(u), ok=False, error=he.detail))
                except Exception as e:
                    items.append(AnalyzeMultiItem(url=str(u), ok=False, error=str(e)))

            badge = compute_product_badge(analyses)
            incr_usage_redis(api_key, tokens_total)
            
            # Facturation agrégée pour badge produit
            total_cost_est = round(sum(a.cost_usd_estimated for a in analyses), 6)
            record_billing_event(api_key, total_cost_est, {"endpoint": "badge/product", "videos_analyzed": len(analyses)})

            elapsed = int((time.time() - t0) * 1000)
            api_calls_total.labels(endpoint="badge_product", api_key_hash=api_key_hash, status="success").inc()
            
            return BadgeProductResponse(
                request_id=str(uuid.uuid4()),
                product_name=body.product_name,
                analyses=items,
                badge={
                    "label": badge["label"],
                    "score": badge["score"],
                    "explanation": badge["reasons"],
                    "display": {
                        "title": f"Videolyze Trust — {badge['label'].upper()}",
                        "tooltip": f"{badge['score']*100:.0f}% de confiance basé sur l'analyse de vidéos d'avis."
                    }
                },
                runtime_ms=elapsed
            )
            
        except HTTPException as e:
            api_calls_total.labels(endpoint="badge_product", api_key_hash=api_key_hash, status="error").inc()
            raise
        except Exception as e:
            api_calls_total.labels(endpoint="badge_product", api_key_hash=api_key_hash, status="error").inc()
            logger.error(f"Erreur inattendue badge_product: {e}")
            raise HTTPException(status_code=500, detail="Erreur interne du serveur")

# =========================
# 🤝 Agents manifest (développement/staging uniquement)
# =========================

@app.get("/agents/manifest")
def agents_manifest():
    """
    Manifest pour agents (MCP/A2A-like) : expose les capacités de l'API.
    Désactivé en production pour éviter l'exposition de l'architecture.
    """
    if ENVIRONMENT == "production":
        raise HTTPException(status_code=404, detail="Not Found")
    
    return {
        "name": "videolyze",
        "version": APP_VERSION,
        "environment": ENVIRONMENT,
        "capabilities": [
            {
                "name": "analyze_video",
                "endpoint": "/analyze/video",
                "input": {"url": "HttpUrl", "lang": "string", "max_comments": "int<=100"},
                "output": {"summary": "bullets[5]", "sentiment": "label/score/counts", "flags": "list"}
            },
            {
                "name": "analyze_multi",
                "endpoint": "/analyze/multi",
                "input": {"urls": "HttpUrl[]<=10", "lang": "string"},
                "output": {"results": "AnalyzeResponse[]", "aggregate": "object"}
            },
            {
                "name": "badge_product",
                "endpoint": "/badge/product",
                "input": {"product_name": "string", "video_urls": "HttpUrl[]", "lang": "string"},
                "output": {"badge": "label/score/explanation", "analyses": "AnalyzeResponse[]"}
            }
        ],
        "auth": {"type": "api_key", "header": "x-api-key"},
        "rate_limits": {"requests_per_hour": MAX_REQUESTS_PER_HOUR},
        "contact": {"email": "support@videolyze.tech"},
        "monitoring": {
            "metrics": "/metrics",
            "health": "/health"
        }
    }

# =========================
# Production security
# =========================

if ENVIRONMENT == "production":
    @app.get("/robots.txt", response_class=PlainTextResponse)
    def robots():
        return "User-agent: *\nDisallow: /"
