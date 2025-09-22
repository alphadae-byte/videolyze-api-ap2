# 🎥 Videolyze API

**Service d'intelligence vidéo compatible Agent Payments Protocol (AP2)**

Videolyze analyse automatiquement les vidéos **YouTube** et **TikTok** pour fournir aux agents IA des insights structurés sur les produits, les tendances et les opinions. Parfaitement intégré dans l'écosystème AP2 comme service de données pré-achat.

## 🤖 Positionnement dans l'écosystème AP2

Videolyze agit comme **service d'intelligence** entre le Shopping Agent et le Merchant Endpoint :

```
User Request → Shopping Agent → Videolyze → Product Analysis → Merchant → Purchase Decision
```

**Flux typique :**
1. Agent reçoit : "Find me a vacuum under $300 with good reviews"
2. Agent appelle Videolyze : `/analyze/multi` avec URLs de reviews vidéo
3. Videolyze retourne : sentiment, flags, résumé des réactions
4. Agent utilise cette intel pour sélectionner le meilleur produit
5. Cart Mandate généré avec le merchant final

Compatible avec les protocoles **MCP** et **A2A** pour une intégration transparente.

---

## 🚀 Cas d'usage

### Shopping Intelligent
```python
# Agent analysant des reviews vidéo avant achat
response = await videolyze.analyze_multi([
    "https://youtube.com/watch?v=dyson-v8-review",
    "https://youtube.com/watch?v=shark-vs-dyson",
    "https://tiktok.com/@reviewer/dyson-test"
])

# Résultat : sentiment global, red flags, points clés
if response.aggregate.avg_sentiment_score > 0.6:
    agent.recommend_purchase()
```

### Badge de Confiance E-commerce
```python
# Pour marchands Shopify/Etsy
badge = await videolyze.badge_product(
    product_name="Dyson V8 Absolute",
    video_urls=review_urls
)

# Retourne un score de confiance + tooltip
# "85% de confiance basé sur l'analyse de 12 vidéos d'avis"
```

### Veille Concurrentielle (Agences)
```python
# Monitoring automatique de campagnes
analysis = await videolyze.analyze_video(
    "https://youtube.com/watch?v=competitor-campaign",
    lang="fr"
)

# Détecte : sentiment négatif, controverse, tendances
```

---

## 🛠️ API Endpoints

### `/analyze/video` - Analyse individuelle
```json
POST /analyze/video
{
  "url": "https://youtube.com/watch?v=abc123",
  "lang": "fr",
  "max_comments": 30
}
```

**Réponse :**
```json
{
  "summary": {
    "bullets": ["Produit très apprécié", "Bonne durée de vie", "Prix élevé mais justifié", "Design moderne", "Service client réactif"]
  },
  "sentiment": {
    "label": "positive",
    "score": 0.73,
    "counts": {"pos": 18, "neu": 8, "neg": 4}
  },
  "flags": {
    "items": ["price_complaints"]
  },
  "tokens_used": 1247,
  "cost_usd_estimated": 0.000374
}
```

### `/analyze/multi` - Analyse comparative
Analyse jusqu'à 10 vidéos simultanément avec agrégation des résultats.

### `/badge/product` - Score de confiance
Génère un badge de confiance basé sur l'analyse de vidéos d'avis produit.

---

## 📦 Installation

### 1. Cloner et installer
```bash
git clone https://github.com/your-org/videolyze-api.git
cd videolyze-api
pip install -r requirements.txt
```

### 2. Configuration des variables d'environnement

**Minimale (développement) :**
```bash
export ENVIRONMENT=development
export YOUTUBE_API_KEY=AIza...
export OPENAI_API_KEY=sk-...
export VIDEOLYZE_API_KEYS=dev_key_1,dev_key_2
```

**Production complète :**
```bash
# Core
export ENVIRONMENT=production
export YOUTUBE_API_KEY=AIza...
export OPENAI_API_KEY=sk-...
export VIDEOLYZE_API_KEYS=prod_key_1,prod_key_2

# Redis (obligatoire)
export REDIS_URL=redis://redis:6379/0

# Billing Stripe
export STRIPE_API_KEY=sk_live_...
export STRIPE_API_MAPPING="prod_key_1:cus_ABC123:si_XYZ789,prod_key_2:cus_DEF456:si_UVW012"

# Audit Postgres (optionnel)
export POSTGRES_DSN="postgresql://user:pass@postgres:5432/videolyze_audit"

# CORS
export CORS_ALLOWED_ORIGINS="https://app.yourdomain.com,https://dashboard.yourdomain.com"
```

### 3. Services requis

**Redis (obligatoire) :**
```bash
docker run -d --name redis -p 6379:6379 redis:7-alpine
```

**Postgres (optionnel, pour audit) :**
```bash
docker run -d --name postgres \
  -e POSTGRES_DB=videolyze_audit \
  -e POSTGRES_USER=videolyze \
  -e POSTGRES_PASSWORD=your_password \
  -p 5432:5432 postgres:15
```

### 4. Démarrage
```bash
# Développement
uvicorn main:app --reload --port 8000

# Production
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 5. Vérification
```bash
# Health check
curl http://localhost:8000/health

# Test d'analyse
curl -X POST http://localhost:8000/analyze/video \
  -H "Content-Type: application/json" \
  -H "x-api-key: your_api_key" \
  -d '{"url": "https://youtube.com/watch?v=example"}'
```

---

## 🔧 Architecture technique

### Stack
- **FastAPI** + Python 3.9+
- **Redis** (cache + rate limiting)
- **OpenAI GPT-4o-mini** (analyse de contenu)
- **YouTube Data API v3**
- **Postgres** (audit, optionnel)

### Robustesse
- **Circuit breaker** OpenAI (protection surcharge)
- **Retry avec backoff exponentiel** (résilience)
- **Fallback mémoire** si Redis indisponible
- **Cache distribué** avec TTL intelligent (1h)
- **Rate limiting** par API key (200 req/h par défaut)

### Monitoring
- **Prometheus metrics** (`/metrics`)
- **Health checks** (`/health`)
- **Logging structuré** par environnement
- **Audit billing** persistant

---

## 💳 Monétisation

### Modèle de pricing
- **Micropaiements** : 1-10€ par appel selon complexité
- **Facturation à l'usage** via Stripe usage records
- **Coûts transparents** : estimation OpenAI tokens incluse

### Stripe Setup
1. Créer des customers et subscriptions Stripe
2. Récupérer les `subscription_item_id`
3. Mapper dans `STRIPE_API_MAPPING`

### Exemple configuration client
```bash
export STRIPE_API_MAPPING="client_prod_key:cus_NvZZ8i5i3YqI8K:si_NvZZ8i5i3YqI8L"
```

---

## 🤝 Intégration agents

### MCP/A2A Compatible
Le manifest est exposé sur `/agents/manifest` (dev/staging uniquement) :

```json
{
  "name": "videolyze",
  "capabilities": [
    {
      "name": "analyze_video",
      "endpoint": "/analyze/video",
      "input": {"url": "HttpUrl", "lang": "string"},
      "output": {"summary": "bullets[5]", "sentiment": "label/score", "flags": "list"}
    }
  ],
  "auth": {"type": "api_key", "header": "x-api-key"}
}
```

### Agent Shopping Example
```javascript
// Agent utilisant Videolyze pour analyser avant achat
const analysis = await videolyze.analyzeVideo({
  url: productReviewUrl,
  lang: "fr"
});

if (analysis.sentiment.score > 0.6 && analysis.flags.items.length === 0) {
  return { recommendation: "BUY", confidence: analysis.sentiment.score };
} else {
  return { recommendation: "SKIP", reasons: analysis.flags.items };
}
```

---

## 📊 Exemple de monitoring

### Grafana Dashboard
Métriques disponibles via `/metrics` :
- `videolyze_api_calls_total` (par endpoint, statut)
- `videolyze_tokens_used_total` (coûts OpenAI)
- `videolyze_request_duration_seconds` (performance)
- `videolyze_cache_hits_total` (efficacité cache)
- `videolyze_redis_connected` (état Redis)

### Alertes recommandées
```yaml
# Prometheus alerts
- alert: VideolyzeHighErrorRate
  expr: rate(videolyze_api_calls_total{status="error"}[5m]) > 0.1
  
- alert: VideolyzeRedisDown
  expr: videolyze_redis_connected == 0
  
- alert: VideolyzeCircuitBreakerOpen
  expr: videolyze_openai_errors_total > 5
```

---

## 🔒 Sécurité

### Production
- **CORS strict** avec whitelist de domaines
- **Rate limiting** distribué (Redis)
- **API keys** hashées dans les logs
- **Endpoints sensibles** masqués (`/`, `/docs`)
- **Secrets** via variables d'environnement uniquement

### Compliance
- **RGPD** : pas de stockage données personnelles
- **Logs** : API keys anonymisées
- **Audit trail** : toutes transactions tracées

---

## 📈 Roadmap

### Phase actuelle : MVP production-ready
- ✅ Analyse YouTube/TikTok
- ✅ Cache Redis + fallback
- ✅ Billing Stripe
- ✅ Monitoring Prometheus

### Phase suivante : Extension AP2
- 🔄 Support Intent Mandates (human not present)
- 🔄 Dashboard client self-service
- 🔄 Widget Shopify/Etsy
- 🔄 Support vidéos Instagram Reels

### Vision long terme
- 🎯 Marketplace officiel AP2
- 🎯 Intégration directe agents Google/OpenAI
- 🎯 Analytics prédictives (tendances)

---

## 📞 Support

- **Email** : videolyzetech@proton.me
- **Documentation** : `/docs` (dev/staging)
- **Status** : `/health`
- **Issues** : GitHub Issues

---

## 📝 License

MIT License - Voir [LICENSE](./LICENSE) pour les détails.

---

**Construit pour l'écosystème Agent Payments Protocol (AP2)**  
Compatible MCP • A2A • Stripe • Production-ready