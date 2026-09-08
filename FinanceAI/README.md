# 🤖 FinanceAI — AI-Powered Stock Market Intelligence System

> Production-grade Machine Learning platform for Indian equity markets.
> Predicts high-return stocks, recommends F&O strategies, and ranks Mutual Funds
> using an ensemble of LSTM + XGBoost + LightGBM with FinBERT sentiment analysis.

---

## 📋 Table of Contents

- [Architecture Overview](#-architecture-overview)
- [Tech Stack](#-tech-stack)
- [Prerequisites](#-prerequisites)
- [Run Locally (Without Docker)](#-run-locally-without-docker)
  - [1. Clone & Setup Environment](#1-clone--setup-environment)
  - [2. Configure Environment Variables](#2-configure-environment-variables)
  - [3. Start PostgreSQL Locally](#3-start-postgresql-locally)
  - [4. Start Redis Locally](#4-start-redis-locally)
  - [5. Initialize the Database](#5-initialize-the-database)
  - [6. Start the FastAPI Server](#6-start-the-fastapi-server)
  - [7. Run the Full Pipeline (No API Server)](#7-run-the-full-pipeline-no-api-server)
  - [8. Run Unit Tests](#8-run-unit-tests)
- [Run on a Production Server](#-run-on-a-production-server)
  - [1. Server Preparation](#1-server-preparation)
  - [2. Clone & Install](#2-clone--install)
  - [3. Configure Environment](#3-configure-environment)
  - [4. Start with Docker Compose](#4-start-with-docker-compose)
  - [5. Verify All Services](#5-verify-all-services)
  - [6. Configure Nginx Reverse Proxy](#6-configure-nginx-reverse-proxy)
  - [7. SSL with Let's Encrypt](#7-ssl-with-lets-encrypt)
  - [8. Run as Systemd Service (Non-Docker)](#8-run-as-systemd-service-non-docker)
- [Pipeline Guide](#-pipeline-guide)
  - [Pipeline Architecture](#pipeline-architecture)
  - [Pipeline 1 — Daily Data Ingestion](#pipeline-1--daily-data-ingestion)
  - [Pipeline 2 — Daily Prediction Pipeline](#pipeline-2--daily-prediction-pipeline)
  - [Pipeline 3 — Weekly Model Retraining](#pipeline-3--weekly-model-retraining)
  - [Setup Airflow Locally](#setup-airflow-locally)
  - [Setup Airflow on Server](#setup-airflow-on-server)
  - [Manual Pipeline Trigger](#manual-pipeline-trigger)
  - [Schedule Reference](#schedule-reference)
- [API Reference](#-api-reference)
- [Monitoring & Alerting](#-monitoring--alerting)
- [Model Training Guide](#-model-training-guide)
- [Project Structure](#-project-structure)
- [Troubleshooting](#-troubleshooting)

---

## 🏗 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                                 │
│  Yahoo Finance │ NSE India API │ NewsAPI │ AMFI India │ Macro Data  │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    FEATURE ENGINEERING                              │
│  40+ Technical Indicators │ FinBERT Sentiment │ Fundamental Ratios  │
│  Macro Features (USD/INR, Crude, Gold, VIX, US10Y)                  │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌────────────────┬───────────────────┬────────────────────────────────┐
│  LSTM Model    │  XGBoost + Optuna │  LightGBM                      │
│  (Bi-LSTM +    │  (Tabular, HPO,   │  (Tabular, fast                │
│   Attention)   │   SHAP explain)   │   gradient boost)              │
│    35% weight  │     40% weight    │     25% weight                 │
└───────┬────────┴─────────┬─────────┴───────────┬────────────────────┘
        └──────────────────┼─────────────────────┘
                           ▼
              ┌─────────────────────────┐
              │  STACKING ENSEMBLE      │
              │  Meta-Learner (LR)      │
              │  + Composite Scoring    │
              │  (Technical 35%,        │
              │   Fundamental 20%,      │
              │   Model Prob 45%)       │
              └────────────┬────────────┘
                           │
          ┌────────────────┼─────────────────┐
          ▼                ▼                 ▼
   ┌─────────────┐  ┌───────────┐   ┌──────────────┐
   │   STOCK     │  │   FNO     │   │  MUTUAL FUND │
   │ SCREENER    │  │ ANALYZER  │   │ RECOMMENDER  │
   │ BUY/SELL    │  │ Options   │   │ Risk-profile │
   │ Top Picks   │  │ Strategy  │   │ Ranking      │
   └─────────────┘  └───────────┘   └──────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │   FastAPI REST API      │
              │   JWT Auth │ Rate Limit │
              │   Redis Cache │ Swagger  │
              └─────────────────────────┘
```

---

## 🛠 Tech Stack

| Layer              | Technology                                       |
|--------------------|--------------------------------------------------|
| **API Framework**  | FastAPI 0.111 + Uvicorn                          |
| **Deep Learning**  | TensorFlow 2.16 (Bi-LSTM + Multi-Head Attention) |
| **ML / Boosting**  | XGBoost 2.0, LightGBM 4.3, Scikit-learn 1.5     |
| **HPO**            | Optuna 3.6                                       |
| **Explainability** | SHAP 0.45                                        |
| **NLP Sentiment**  | FinBERT (HuggingFace), VADER fallback            |
| **Market Data**    | yfinance, NSE India API, AMFI API, NewsAPI       |
| **Database**       | PostgreSQL 16 (async via SQLAlchemy + asyncpg)   |
| **Cache**          | Redis 7                                          |
| **Time-Series DB** | InfluxDB 2.7                                     |
| **Orchestration**  | Apache Airflow 2.9                               |
| **Auth**           | JWT (python-jose + bcrypt)                       |
| **Monitoring**     | Prometheus + Grafana                             |
| **Containerization**| Docker + Docker Compose                         |

---

## ✅ Prerequisites

### Local Development
- Python **3.11+**
- PostgreSQL **14+**
- Redis **6+**
- Git

### Production Server
- Ubuntu **22.04 LTS** (recommended) / any Linux
- Docker **24+** and Docker Compose **2.24+**
- Minimum **8 GB RAM**, **4 vCPU**, **50 GB SSD**
- Ports open: `80`, `443`, `8000`, `8080`, `9090`, `3001`

---

## 💻 Run Locally (Without Docker)

### 1. Clone & Setup Environment

```bash
# Clone the repository
git clone https://github.com/your-org/financeai.git
cd financeai

# Create Python virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt
```

> ⚠️ **Note on TensorFlow**: If you're on Apple Silicon (M1/M2), replace `tensorflow` with `tensorflow-macos` in `requirements.txt` before installing.
>
> ⚠️ **Note on PyTorch**: FinBERT requires `torch`. Install the CPU version if you don't have a GPU:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> ```

---

### 2. Configure Environment Variables

```bash
# Copy the example env file
cp .env.example .env
```

Open `.env` and fill in your values:

```env
# ── App Settings ─────────────────────────────────────────────────────────────
ENVIRONMENT=development
DEBUG=true
SECRET_KEY=your-super-secret-key-minimum-32-characters

# ── Database ──────────────────────────────────────────────────────────────────
POSTGRES_URL=postgresql+asyncpg://finai:finai@localhost:5432/financeai
REDIS_URL=redis://localhost:6379/0
INFLUX_URL=http://localhost:8086
INFLUX_TOKEN=my-super-secret-token
INFLUX_ORG=financeai
INFLUX_BUCKET=market_data

# ── API Keys ──────────────────────────────────────────────────────────────────
# Get free key at: https://newsapi.org/register
NEWSAPI_KEY=your_newsapi_key_here

# Get free key at: https://www.alphavantage.co/support/#api-key
ALPHAVANTAGE_KEY=your_alphavantage_key_here

# Razorpay keys for UPI and card payments
RAZORPAY_KEY_ID=rzp_test_xxxxxxxx
RAZORPAY_KEY_SECRET=your_razorpay_secret
RAZORPAY_WEBHOOK_SECRET=your_razorpay_webhook_secret

# ── Model Config ──────────────────────────────────────────────────────────────
MODEL_STORE_PATH=models_store/
LSTM_LOOKBACK=60
PREDICTION_CONFIDENCE_THRESHOLD=0.65
MIN_TRAINING_SAMPLES=500
```

> 💡 **Free API Keys needed:**
> - **NewsAPI**: [newsapi.org/register](https://newsapi.org/register) — 100 requests/day free
> - **AlphaVantage**: [alphavantage.co](https://www.alphavantage.co/support/#api-key) — 25 requests/day free
> - **Yahoo Finance (yfinance)**: No key needed — works out of the box

---

### 3. Start PostgreSQL Locally

#### Option A — If PostgreSQL is already installed:

```bash
# On Windows (PowerShell as Admin):
net start postgresql-x64-16

# On macOS (Homebrew):
brew services start postgresql@16

# On Linux:
sudo systemctl start postgresql
```

Create the database and user:

```sql
-- Connect as superuser (psql -U postgres)
CREATE USER finai WITH PASSWORD 'finai';
CREATE DATABASE financeai OWNER finai;
GRANT ALL PRIVILEGES ON DATABASE financeai TO finai;
\q
```

#### Option B — Quick start with Docker (only PostgreSQL):

```bash
docker run -d \
  --name finai-postgres \
  -e POSTGRES_USER=finai \
  -e POSTGRES_PASSWORD=finai \
  -e POSTGRES_DB=financeai \
  -p 5432:5432 \
  postgres:16-alpine
```

---

### 4. Start Redis Locally

#### Option A — If Redis is installed:

```bash
# macOS/Linux:
redis-server

# Windows (via WSL or Redis for Windows):
redis-server --daemonize yes
```

#### Option B — Quick start with Docker:

```bash
docker run -d \
  --name finai-redis \
  -p 6379:6379 \
  redis:7-alpine
```

Verify Redis is running:

```bash
redis-cli ping
# Expected output: PONG
```

---

### 5. Initialize the Database

```bash
# Run from the project root with venv activated
python -c "
import asyncio
from src.database.models import init_db
asyncio.run(init_db())
print('✅ Database tables created successfully')
"
```

Expected output:
```
✅ Database tables created successfully
```

---

### 6. Start the FastAPI Server

```bash
# Development mode (auto-reload on code changes)
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# OR run the main module directly
python -m src.api.main
```

Verify the API is running:

```bash
# Health check
curl http://localhost:8000/api/v1/health

# Expected response:
# {"status":"ok","timestamp":"2026-08-22T06:04:09.386Z","service":"FinanceAI"}
```

**API Documentation** available at:
- Swagger UI: [http://localhost:8000/api/docs](http://localhost:8000/api/docs)
- ReDoc: [http://localhost:8000/api/redoc](http://localhost:8000/api/redoc)

---

### 7. Run the Full Pipeline (No API Server)

This runs the complete data → features → prediction → recommendation pipeline standalone:

```bash
# From project root with venv activated
python scripts/run_pipeline.py
```

Expected output:
```
============================================================
  FinanceAI — Full Pipeline Run
============================================================

📥 Step 1: Fetching market data…
  Fetching RELIANCE.NS… ✓ 500 rows
  Fetching TCS.NS… ✓ 500 rows
  Fetching INFY.NS… ✓ 500 rows
  ...

📰 Step 2: Fetching & analyzing news sentiment…
  Processed 87 articles

⚙️  Step 3: Computing technical indicators…
  RELIANCE.NS: 62 features computed
  TCS.NS: 62 features computed
  ...

🤖 Step 4: Running ensemble predictions…
  RELIANCE.NS           | Signal: BUY  | Confidence: 72.3% | Target: ₹3105 | Stop: ₹2821
  TCS.NS                | Signal: BUY  | Confidence: 68.1% | Target: ₹4215 | Stop: ₹3765
  ...

🎯 Step 5: Top Stock Recommendations
  Rank  Symbol               Score    R:R    Rationale
  ──────────────────────────────────────────────────────────────────────────────────
  #1    TCS.NS               78.2%    2.0    Strong technical setup. Sound fundamentals
  #2    RELIANCE.NS          72.1%    2.0    High model confidence (74%). Favorable R:R

📊 Step 6: FNO Strategy Analysis
  RELIANCE.NS          | Strategy: BUY FUTURE                  | Risk: HIGH
  TCS.NS               | Strategy: BULL CALL SPREAD            | Risk: LOW
  ...

💰 Step 7: Mutual Fund Recommendations (MODERATE profile)
  #1 Parag Parikh Flexi Cap Fund                      | Score: 67.50%
  #2 Mirae Asset Large Cap Fund                       | Score: 65.20%
  ...

✅ Pipeline completed successfully!
============================================================
```

---

### 8. Run Unit Tests

```bash
# Run all tests
pytest tests/ -v

# Run only unit tests
pytest tests/unit/ -v

# Run with coverage report
pytest tests/ -v --cov=src --cov-report=term-missing

# Run a specific test class
pytest tests/unit/test_features_and_recommendation.py::TestTechnicalFeatures -v
```

Expected output:
```
tests/unit/test_features_and_recommendation.py::TestTechnicalFeatures::test_compute_returns_columns PASSED
tests/unit/test_features_and_recommendation.py::TestTechnicalFeatures::test_rsi_bounds PASSED
tests/unit/test_features_and_recommendation.py::TestTechnicalFeatures::test_no_inf_values PASSED
...
15 passed in 8.42s
```

---

## 🖥 Run on a Production Server

### 1. Server Preparation

```bash
# Update system packages
sudo apt-get update && sudo apt-get upgrade -y

# Install required system tools
sudo apt-get install -y \
    git curl wget unzip \
    build-essential libpq-dev \
    python3.11 python3.11-venv python3-pip \
    nginx certbot python3-certbot-nginx \
    ufw

# Install Docker
curl -fsSL https://get.docker.com | bash
sudo usermod -aG docker $USER
newgrp docker

# Install Docker Compose
sudo apt-get install -y docker-compose-plugin
docker compose version   # Verify: Docker Compose version v2.x.x

# Configure firewall
sudo ufw allow 22      # SSH
sudo ufw allow 80      # HTTP
sudo ufw allow 443     # HTTPS
sudo ufw allow 8000    # API (close after Nginx setup)
sudo ufw enable
```

---

### 2. Clone & Install

```bash
# Create app directory
sudo mkdir -p /opt/financeai
sudo chown $USER:$USER /opt/financeai

# Clone project
git clone https://github.com/your-org/financeai.git /opt/financeai
cd /opt/financeai
```

---

### 3. Configure Environment

```bash
# Copy and edit environment file
cp .env.example .env
nano .env
```

Update these values for production:

```env
ENVIRONMENT=production
DEBUG=false

# Use a strong randomly generated secret key:
# python -c "import secrets; print(secrets.token_hex(32))"
SECRET_KEY=<your-64-char-hex-secret>

# Point to Docker service names (not localhost)
POSTGRES_URL=postgresql+asyncpg://finai:StrongPassw0rd@postgres:5432/financeai
REDIS_URL=redis://redis:6379/0
INFLUX_URL=http://influxdb:8086
INFLUX_TOKEN=<strong-influx-token>

# Real API keys
NEWSAPI_KEY=<your-production-newsapi-key>
ALPHAVANTAGE_KEY=<your-alphavantage-key>
```

Also update database passwords in `docker/docker-compose.yml`:

```yaml
postgres:
  environment:
    POSTGRES_PASSWORD: StrongPassw0rd   # ← Change this
```

---

### 4. Start with Docker Compose

```bash
cd /opt/financeai

# Build and start all services in background
docker compose -f docker/docker-compose.yml up -d --build

# View startup logs
docker compose -f docker/docker-compose.yml logs -f

# Check service status
docker compose -f docker/docker-compose.yml ps
```

Expected services running:
```
NAME                    STATUS          PORTS
financeai-api           Up (healthy)    0.0.0.0:8000->8000/tcp
financeai-postgres      Up (healthy)    0.0.0.0:5432->5432/tcp
financeai-redis         Up (healthy)    0.0.0.0:6379->6379/tcp
financeai-influx        Up              0.0.0.0:8086->8086/tcp
financeai-airflow       Up              0.0.0.0:8080->8080/tcp
financeai-scheduler     Up
financeai-prometheus    Up              0.0.0.0:9090->9090/tcp
financeai-grafana       Up              0.0.0.0:3001->3000/tcp
```

Initialize the database (first time only):

```bash
docker compose -f docker/docker-compose.yml exec api \
  python -c "import asyncio; from src.database.models import init_db; asyncio.run(init_db())"
```

---

### 5. Verify All Services

```bash
# API health check
curl http://your-server-ip:8000/api/v1/health

# API readiness (checks DB + Redis)
curl http://your-server-ip:8000/api/v1/ready

# Prometheus metrics
curl http://your-server-ip:9090/-/healthy

# Grafana
curl http://your-server-ip:3001/api/health
```

---

### 6. Configure Nginx Reverse Proxy

```bash
# Create Nginx config
sudo nano /etc/nginx/sites-available/financeai
```

Paste the following:

```nginx
upstream financeai_api {
    server 127.0.0.1:8000;
    keepalive 32;
}

server {
    listen 80;
    server_name api.yourdomain.com;

    # Security headers
    add_header X-Frame-Options          DENY;
    add_header X-Content-Type-Options   nosniff;
    add_header X-XSS-Protection         "1; mode=block";
    add_header Referrer-Policy          strict-origin-when-cross-origin;

    # Gzip compression
    gzip on;
    gzip_types application/json text/plain;
    gzip_min_length 1000;

    # API proxy
    location / {
        proxy_pass          http://financeai_api;
        proxy_http_version  1.1;
        proxy_set_header    Upgrade            $http_upgrade;
        proxy_set_header    Connection         "upgrade";
        proxy_set_header    Host               $host;
        proxy_set_header    X-Real-IP          $remote_addr;
        proxy_set_header    X-Forwarded-For    $proxy_add_x_forwarded_for;
        proxy_set_header    X-Forwarded-Proto  $scheme;
        proxy_read_timeout  300s;
        proxy_connect_timeout 75s;
        client_max_body_size 10M;
    }

    # WebSocket support
    location /ws/ {
        proxy_pass          http://financeai_api;
        proxy_http_version  1.1;
        proxy_set_header    Upgrade    $http_upgrade;
        proxy_set_header    Connection "Upgrade";
    }
}
```

Enable the site:

```bash
sudo ln -s /etc/nginx/sites-available/financeai /etc/nginx/sites-enabled/
sudo nginx -t        # Test config
sudo systemctl restart nginx
```

---

### 7. SSL with Let's Encrypt

```bash
# Obtain SSL certificate (replace with your domain)
sudo certbot --nginx -d api.yourdomain.com

# Test automatic renewal
sudo certbot renew --dry-run

# Certbot auto-renews via cron — verify:
sudo systemctl status certbot.timer
```

---

### 8. Run as Systemd Service (Non-Docker)

If you prefer running without Docker, create a systemd service:

```bash
# Create service file
sudo nano /etc/systemd/system/financeai.service
```

```ini
[Unit]
Description=FinanceAI FastAPI Application
After=network.target postgresql.service redis.service

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/financeai
Environment=PYTHONPATH=/opt/financeai
EnvironmentFile=/opt/financeai/.env
ExecStart=/opt/financeai/venv/bin/uvicorn src.api.main:app \
          --host 0.0.0.0 \
          --port 8000 \
          --workers 4 \
          --log-level info \
          --access-log
Restart=on-failure
RestartSec=5s
StandardOutput=append:/var/log/financeai/api.log
StandardError=append:/var/log/financeai/api_error.log

[Install]
WantedBy=multi-user.target
```

```bash
# Create log directory
sudo mkdir -p /var/log/financeai
sudo chown www-data:www-data /var/log/financeai

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable financeai
sudo systemctl start financeai

# Check status
sudo systemctl status financeai

# View logs
sudo journalctl -u financeai -f
```

---

## 🔄 Pipeline Guide

### Pipeline Architecture

```
Market Opens (9:15 AM IST)
        │
        ▼  9:30 AM IST (Mon–Fri)
┌─────────────────────────┐
│  DAG 1: Data Ingestion  │  ← Parallel tasks
│  ├── ingest_ohlcv        │    - Yahoo Finance OHLCV
│  ├── ingest_news         │    - NewsAPI + RSS feeds
│  └── ingest_macro        │    - Macro indicators
└────────────┬────────────┘
             │ triggers on success
             ▼  10:00 AM IST (Mon–Fri)
┌─────────────────────────┐
│  DAG 2: Predictions     │
│  ├── feature_engineering │
│  ├── run_ensemble        │
│  ├── screen_stocks       │
│  └── persist_to_db       │
└────────────┬────────────┘
             │
             ▼  Every Sunday 2 AM UTC
┌─────────────────────────┐
│  DAG 3: Model Retrain   │
│  ├── train_lstm          │
│  ├── train_xgboost_hpo   │
│  └── train_lightgbm      │
└─────────────────────────┘
```

---

### Pipeline 1 — Daily Data Ingestion

**Schedule**: `30 9 * * 1-5` (9:30 AM IST, Monday–Friday)

**What it does:**
- Fetches 5 days of OHLCV data for all 20 watchlist stocks from Yahoo Finance
- Downloads financial news from NewsAPI + Moneycontrol RSS + Economic Times RSS
- Pulls macro indicators: USD/INR, Gold, Crude Oil, VIX India, US 10Y Yield
- Saves raw data as Parquet files in `data/raw/`

**Tasks (run in parallel):**

| Task | Duration | Output |
|------|----------|--------|
| `ingest_ohlcv` | ~30 sec | `data/raw/{SYMBOL}_ohlcv.parquet` |
| `ingest_news` | ~45 sec | `data/raw/news_{timestamp}.json` |
| `ingest_macro` | ~20 sec | In-memory (passed to next DAG) |

---

### Pipeline 2 — Daily Prediction Pipeline

**Schedule**: `0 10 * * 1-5` (10:00 AM IST, Monday–Friday)

**What it does:**
1. Loads OHLCV data fetched in Pipeline 1
2. Computes 40+ technical indicators per stock
3. Runs FinBERT sentiment analysis on news articles
4. Loads pre-trained LSTM, XGBoost, and LightGBM models
5. Generates ensemble predictions (BUY / SELL / HOLD)
6. Applies StockScreener filters (confidence ≥ 0.65, R:R ≥ 1.5)
7. Persists predictions to PostgreSQL

**Output stored in DB:**
- Signal (BUY/SELL/HOLD) + confidence score
- Target price + stop-loss + risk-reward ratio
- Technical, fundamental, and sentiment sub-scores
- SHAP feature importance (top 20 features)
- Cached in Redis for 15 minutes

---

### Pipeline 3 — Weekly Model Retraining

**Schedule**: `0 2 * * 0` (Sunday 2:00 AM UTC = 7:30 AM IST)

**What it does:**
- Fetches 5 years of OHLCV history per symbol
- Re-engineers all features with latest data
- Retrains LSTM (50 epochs with early stopping)
- Runs Optuna HPO for XGBoost (30 trials, ~10 min per stock)
- Retrains LightGBM with early stopping
- Saves new model artifacts to `models_store/`
- Registers new version in `model_registry` DB table

**Expected duration**: ~2–4 hours for 20 stocks

---

### Setup Airflow Locally

```bash
# Set Airflow home directory
export AIRFLOW_HOME=/opt/financeai/airflow

# Initialize Airflow database
airflow db init

# Create admin user
airflow users create \
    --username admin \
    --firstname FinanceAI \
    --lastname Admin \
    --role Admin \
    --email admin@financeai.io \
    --password admin123

# Copy DAGs to Airflow dag folder
mkdir -p $AIRFLOW_HOME/dags
cp dags/pipeline_dags.py $AIRFLOW_HOME/dags/

# Set PYTHONPATH so DAGs can import src/
export PYTHONPATH=/opt/financeai

# Start Airflow webserver (Terminal 1)
airflow webserver --port 8080

# Start Airflow scheduler (Terminal 2)
export PYTHONPATH=/opt/financeai
airflow scheduler
```

**Airflow UI**: [http://localhost:8080](http://localhost:8080)
- Username: `admin`
- Password: `admin123`

Enable the DAGs from the UI:

```
daily_market_data_ingestion   → Toggle ON
daily_prediction_pipeline     → Toggle ON
weekly_model_retraining       → Toggle ON
```

---

### Setup Airflow on Server

The Airflow webserver and scheduler are included in `docker-compose.yml`.

After `docker compose up`:

```bash
# Create Airflow admin user (first time only)
docker compose -f docker/docker-compose.yml exec airflow-webserver \
  airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@financeai.io \
    --password your_strong_password

# Initialize Airflow database
docker compose -f docker/docker-compose.yml exec airflow-webserver \
  airflow db init
```

**Airflow UI**: `http://your-server-ip:8080`

---

### Manual Pipeline Trigger

#### Via Airflow UI:
1. Open Airflow UI → DAGs tab
2. Find the DAG you want to trigger
3. Click the ▶ (Trigger DAG) button

#### Via Airflow CLI:

```bash
# Trigger data ingestion DAG immediately
airflow dags trigger daily_market_data_ingestion

# Trigger with a specific execution date
airflow dags trigger daily_prediction_pipeline \
    --exec-date 2026-08-22T09:30:00

# Check DAG run status
airflow dags list-runs -d daily_market_data_ingestion

# View task logs
airflow tasks logs daily_prediction_pipeline run_predictions 2026-08-22
```

#### Via Docker (on server):

```bash
# Trigger ingestion pipeline
docker compose -f docker/docker-compose.yml exec airflow-webserver \
  airflow dags trigger daily_market_data_ingestion

# Trigger prediction pipeline
docker compose -f docker/docker-compose.yml exec airflow-webserver \
  airflow dags trigger daily_prediction_pipeline
```

#### Via Python script (no Airflow needed):

```bash
# Run just the ingestion step
python -c "
import sys; sys.path.insert(0, '.')
from dags.pipeline_dags import ingest_ohlcv, ingest_news, ingest_macro
ingest_ohlcv()
ingest_news()
ingest_macro()
"

# Run just the predictions step
python -c "
import sys; sys.path.insert(0, '.')
from dags.pipeline_dags import run_predictions
run_predictions()
"
```

#### Via API endpoint:

```bash
# First login to get JWT token
TOKEN=$(curl -s -X POST http://localhost:8000/api/v1/auth/login \
  -d "username=admin@financeai.io&password=admin123" \
  | python -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

# Trigger on-demand prediction for a stock
curl -X POST http://localhost:8000/api/v1/predictions/run/RELIANCE.NS \
  -H "Authorization: Bearer $TOKEN"
```

---

### Schedule Reference

| DAG | Schedule | IST Time | Description |
|-----|----------|----------|-------------|
| `daily_market_data_ingestion` | `30 9 * * 1-5` | 3:00 PM UTC / 9:30 AM IST | OHLCV + News + Macro |
| `daily_prediction_pipeline` | `0 10 * * 1-5` | 4:30 PM UTC / 10:00 AM IST | Ensemble ML Predictions |
| `weekly_model_retraining` | `0 2 * * 0` | Sun 2:00 AM UTC / 7:30 AM IST | Full model retrain |

> **Note**: NSE market opens at 9:15 AM IST. Pipelines start at 9:30 AM to ensure opening data is available.

---

## 📡 API Reference

### Authentication

```bash
# Register a new user
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"user@example.com","password":"SecurePass123","risk_profile":"MODERATE"}'

# Login and get JWT token
curl -X POST http://localhost:8000/api/v1/auth/login \
  -d "username=user@example.com&password=SecurePass123"
# Response: {"access_token": "eyJ...", "token_type": "bearer"}

# Get current user info
curl http://localhost:8000/api/v1/auth/me \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### Stocks

```bash
# Get OHLCV data
curl "http://localhost:8000/api/v1/stocks/RELIANCE.NS/ohlcv?period=1y&interval=1d" \
  -H "Authorization: Bearer YOUR_TOKEN"

# Get fundamentals (PE, ROE, D/E, etc.)
curl "http://localhost:8000/api/v1/stocks/TCS.NS/fundamentals" \
  -H "Authorization: Bearer YOUR_TOKEN"

# Get technical indicators (last 30 rows)
curl "http://localhost:8000/api/v1/stocks/INFY.NS/technicals" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### Predictions

```bash
# Get top BUY/SELL picks (from cached predictions)
curl "http://localhost:8000/api/v1/predictions/top-picks?top_n=10" \
  -H "Authorization: Bearer YOUR_TOKEN"

# Trigger fresh on-demand prediction for a symbol
curl -X POST "http://localhost:8000/api/v1/predictions/run/HDFCBANK.NS" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

```bash
# Generate trading actions with sell and buy agents
curl -X POST "http://localhost:8000/api/v1/predictions/trade-agents/plan" \
  -H "Content-Type: application/json" \
  -H "Authorization: ******" \
  -d '{"portfolio_symbols":["INFY.NS","RELIANCE.NS"],"top_n_buys":5}'
```

#### Trading Agents endpoint details

- `SellOnDropAgent` checks your current holdings (`portfolio_symbols`) and creates `SELL` actions for stocks with strong downside signals.
- `GrowthStockBuyerAgent` scans the remaining universe and returns top `BUY` actions with stronger expected growth.

Sample response:

```json
{
  "sell_actions": [
    {
      "symbol": "INFY.NS",
      "action": "SELL",
      "confidence": 0.71,
      "expected_return_5d": -4.2,
      "composite_score": 0.33,
      "rationale": "Downside risk detected: signal=SELL, 5D return=-4.20%",
      "rank": 1
    }
  ],
  "buy_actions": [
    {
      "symbol": "TCS.NS",
      "action": "BUY",
      "confidence": 0.8,
      "expected_return_5d": 7.7,
      "composite_score": 0.78,
      "rationale": "Growth setup confirmed: confidence=0.80, 5D return=7.70%, R:R=2.00",
      "rank": 1
    }
  ],
  "total_analyzed": 100,
  "generated_at": "2026-08-31T12:00:00.000000"
}
```

### Payments (UPI + Cards via Razorpay)

```bash
# List available payment methods
curl "http://localhost:8000/api/v1/payments/methods" \
  -H "Authorization: ******"

# Create a payment order
curl -X POST "http://localhost:8000/api/v1/payments/orders" \
  -H "Content-Type: application/json" \
  -H "Authorization: ******" \
  -d '{"amount":499.0,"method":"UPI","description":"FinanceAI Pro plan","notes":{"plan":"pro_monthly"}}'

# Verify payment after Razorpay checkout success
curl -X POST "http://localhost:8000/api/v1/payments/verify" \
  -H "Content-Type: application/json" \
  -H "Authorization: ******" \
  -d '{"order_id":"order_xxx","payment_id":"pay_xxx","signature":"signature_xxx","status":"CAPTURED"}'

# Fetch user transaction history
curl "http://localhost:8000/api/v1/payments/transactions?limit=20" \
  -H "Authorization: ******"
```

### F&O Strategy

```bash
# Get FNO strategy recommendation for a stock
curl "http://localhost:8000/api/v1/fno/RELIANCE.NS/strategy" \
  -H "Authorization: Bearer YOUR_TOKEN"

# Get index summary (Nifty50, Sensex, Bank Nifty)
curl "http://localhost:8000/api/v1/fno/indices/summary" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### Mutual Funds

```bash
# Get MF recommendations (MODERATE risk profile)
curl "http://localhost:8000/api/v1/mutual-funds/recommendations?risk_profile=MODERATE&top_n=5" \
  -H "Authorization: Bearer YOUR_TOKEN"

# Filter by category
curl "http://localhost:8000/api/v1/mutual-funds/recommendations?risk_profile=AGGRESSIVE&category=Mid+Cap&top_n=5" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### Health Checks

```bash
# Basic liveness probe
curl http://localhost:8000/api/v1/health

# Readiness probe (checks PostgreSQL + Redis)
curl http://localhost:8000/api/v1/ready
```

---

## 📊 Monitoring & Alerting

### Grafana Dashboards

Access: `http://your-server-ip:3001`
- Default login: `admin` / `admin123`

**Pre-configured dashboards:**
1. **API Performance** — request rate, latency, error rate
2. **Prediction Quality** — daily signal counts, confidence distribution
3. **System Resources** — CPU, memory, disk usage
4. **Database Health** — PostgreSQL connections, Redis memory

### Prometheus Metrics

Access: `http://your-server-ip:9090`

Key metrics exposed at `/metrics`:
```
# API request count
http_requests_total{method="GET", endpoint="/api/v1/predictions/top-picks", status="200"}

# Prediction confidence histogram
financeai_prediction_confidence_bucket{symbol="RELIANCE.NS", le="0.7"}

# Model inference latency
financeai_model_inference_seconds{model="ensemble"}
```

### Log Locations

```bash
# Application logs (local)
tail -f logs/financeai.log

# Application logs (Docker)
docker compose logs -f api

# Airflow task logs
docker compose logs -f airflow-scheduler

# Nginx access logs
sudo tail -f /var/log/nginx/access.log
```

---

## 🧠 Model Training Guide

### First-Time Training (No Pre-trained Models)

When the system starts without pre-trained model files, it falls back to a default probability of `0.5`. To train models from scratch:

```bash
# Train all models for all watchlist stocks
python -c "
import sys; sys.path.insert(0, '.')
from dags.pipeline_dags import retrain_models
retrain_models()
"
```

This will:
1. Fetch 5 years of data per stock
2. Train LSTM (Bi-LSTM + Attention) — ~10 min per stock
3. Run Optuna HPO for XGBoost (30 trials) — ~5 min per stock
4. Train LightGBM — ~2 min per stock
5. Save artifacts to `models_store/`

**Expected model files after training:**
```
models_store/
├── lstm/
│   ├── RELIANCE_lstm.keras
│   ├── TCS_lstm.keras
│   └── ...
├── xgboost/
│   ├── RELIANCE_xgb.pkl
│   ├── TCS_xgb.pkl
│   └── ...
├── lgbm/
│   ├── RELIANCE_lgbm.pkl
│   └── ...
└── ensemble/
    ├── RELIANCE_ensemble.pkl
    └── ...
```

### Train a Single Stock

```python
# train_single.py
from src.ingestion.data_fetcher import YFinanceFetcher
from src.features.technical import TechnicalFeatures, FeaturePipeline
from src.models.lstm.model import LSTMTrainer
from src.models.xgboost_model.model import XGBoostModel

symbol = "RELIANCE.NS"
sym_clean = "RELIANCE"

# Fetch data
yf = YFinanceFetcher()
ohlcv = yf.fetch_ohlcv(symbol, period="5y")
fund  = yf.fetch_fundamentals(symbol)

# Feature engineering
tf = TechnicalFeatures()
fp = FeaturePipeline()
enriched = tf.compute(ohlcv)

# Train LSTM
X_seq, y_seq, scaler, feats = fp.prepare_lstm_input(enriched, lookback=60)
split = int(0.8 * len(X_seq))
trainer = LSTMTrainer()
result = trainer.train(X_seq[:split], y_seq[:split], X_seq[split:], y_seq[split:], sym_clean)
print("LSTM:", result["metrics"])

# Train XGBoost with HPO
X_tab, y_tab, _, _ = fp.prepare_xgboost_input(enriched, fund)
xgb = XGBoostModel()
result = xgb.train_with_hpo(X_tab, y_tab, sym_clean, n_trials=50)
print("XGB:", result["metrics"])
```

---

## 📁 Project Structure

```
FinanceAI/
│
├── 📄 .env.example               ← Environment template
├── 📄 requirements.txt           ← Python dependencies
├── 📄 README.md                  ← This file
│
├── 📁 config/
│   └── settings.py               ← Pydantic-based configuration
│
├── 📁 src/
│   ├── __init__.py
│   ├── 📁 ingestion/
│   │   └── data_fetcher.py       ← YFinance, NSE, NewsAPI, AMFI, Macro fetchers
│   │
│   ├── 📁 features/
│   │   └── technical.py          ← 40+ technical indicators + feature pipeline
│   │
│   ├── 📁 sentiment/
│   │   └── analyzer.py           ← FinBERT/VADER + symbol extraction + aggregation
│   │
│   ├── 📁 models/
│   │   ├── lstm/
│   │   │   └── model.py          ← Bi-LSTM + Multi-Head Attention (TensorFlow)
│   │   ├── xgboost_model/
│   │   │   └── model.py          ← XGBoost + LightGBM + Optuna HPO + SHAP
│   │   └── ensemble/
│   │       └── predictor.py      ← Stacking ensemble + ModelOrchestrator
│   │
│   ├── 📁 recommendation/
│   │   └── engine.py             ← StockScreener, FNOAnalyzer, MFRecommender
│   │
│   ├── 📁 database/
│   │   └── models.py             ← SQLAlchemy ORM, Redis, InfluxDB clients
│   │
│   └── 📁 api/
│       ├── main.py               ← FastAPI app + middleware + routers
│       ├── 📁 routers/
│       │   ├── auth.py           ← JWT login/register/me
│       │   ├── stocks.py         ← OHLCV, fundamentals, technicals
│       │   ├── predictions.py    ← Top picks, on-demand prediction
│       │   ├── fno.py            ← FNO strategy, index summary
│       │   ├── mutual_funds.py   ← MF recommendations
│       │   └── health.py         ← /health, /ready probes
│       └── 📁 middleware/
│           ├── rate_limit.py     ← Token-bucket rate limiting
│           └── logging_mw.py     ← Request/response logging
│
├── 📁 dags/
│   └── pipeline_dags.py          ← 3 Airflow DAGs (ingestion, prediction, retrain)
│
├── 📁 docker/
│   ├── Dockerfile                ← Multi-stage production image
│   └── docker-compose.yml        ← 9 services (API, DB, Redis, Airflow, Monitoring)
│
├── 📁 monitoring/
│   └── prometheus/
│       └── prometheus.yml        ← Prometheus scrape config
│
├── 📁 scripts/
│   └── run_pipeline.py           ← End-to-end pipeline demo (no API needed)
│
├── 📁 tests/
│   └── unit/
│       └── test_features_and_recommendation.py  ← 15+ unit tests
│
├── 📁 models_store/              ← Trained model artifacts (.keras, .pkl)
├── 📁 data/
│   ├── raw/                      ← Raw OHLCV Parquet + News JSON files
│   └── processed/                ← Feature-engineered datasets
├── 📁 notebooks/                 ← Jupyter notebooks for exploration
└── 📁 logs/                      ← Application logs
```

---

## 🔧 Troubleshooting

### ❌ `asyncpg.exceptions.InvalidCatalogNameError: database "financeai" does not exist`

```bash
# Create the database manually
psql -U postgres -c "CREATE DATABASE financeai OWNER finai;"
```

### ❌ `redis.exceptions.ConnectionError: Error connecting to localhost:6379`

```bash
# Check if Redis is running
redis-cli ping

# Start Redis
sudo systemctl start redis   # Linux
brew services start redis    # macOS
```

### ❌ `OSError: [Errno 98] Address already in use` (port 8000)

```bash
# Find and kill process using port 8000
# Windows:
netstat -ano | findstr :8000
taskkill /PID <pid> /F

# Linux/macOS:
lsof -ti:8000 | xargs kill -9
```

### ❌ FinBERT model download fails

```bash
# Manually download FinBERT
python -c "
from transformers import AutoTokenizer, AutoModelForSequenceClassification
AutoTokenizer.from_pretrained('ProsusAI/finbert')
AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert')
print('FinBERT downloaded successfully')
"
```

If download fails due to network restrictions, the system **automatically falls back to VADER** sentiment analysis.

### ❌ `ModuleNotFoundError: No module named 'src'`

```bash
# Always run from the project root with PYTHONPATH set
export PYTHONPATH=/opt/financeai   # Linux/macOS
$env:PYTHONPATH = "C:\path\to\financeai"  # Windows PowerShell
```

### ❌ Docker services not starting

```bash
# Check logs for specific service
docker compose -f docker/docker-compose.yml logs postgres
docker compose -f docker/docker-compose.yml logs api

# Reset everything (WARNING: deletes all data)
docker compose -f docker/docker-compose.yml down -v
docker compose -f docker/docker-compose.yml up -d --build
```

### ❌ Airflow DAG not showing up

```bash
# Check for syntax errors in the DAG file
python dags/pipeline_dags.py

# Reload DAGs in Airflow
airflow dags reserialize

# Check Airflow scheduler logs
docker compose logs airflow-scheduler | grep ERROR
```

### ❌ `yfinance` returns empty DataFrame

This happens when NSE symbols are queried outside market hours or with wrong suffix.

```python
# Correct symbol format for NSE stocks
"RELIANCE.NS"    # NSE
"RELIANCE.BO"    # BSE
"^NSEI"          # Nifty 50 index
"^BSESN"         # Sensex index
```

---

## 📌 Quick Reference Commands

```bash
# ── Local Dev ────────────────────────────────────────────────────────────────
source venv/bin/activate                           # Activate venv
uvicorn src.api.main:app --reload --port 8000      # Start API
python scripts/run_pipeline.py                     # Run full pipeline
pytest tests/ -v                                   # Run tests

# ── Docker ───────────────────────────────────────────────────────────────────
docker compose -f docker/docker-compose.yml up -d  # Start all services
docker compose -f docker/docker-compose.yml down   # Stop all services
docker compose -f docker/docker-compose.yml logs -f api  # Stream API logs
docker compose -f docker/docker-compose.yml ps     # Service status

# ── Airflow ──────────────────────────────────────────────────────────────────
airflow dags trigger daily_market_data_ingestion   # Manual ingestion run
airflow dags trigger daily_prediction_pipeline     # Manual prediction run
airflow dags list                                  # List all DAGs

# ── Database ─────────────────────────────────────────────────────────────────
psql -U finai -d financeai                         # Connect to DB
redis-cli monitor                                  # Watch Redis commands
```

---

## 📜 License

MIT License — See [LICENSE](LICENSE) for details.

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/your-feature`
3. Commit changes: `git commit -m "feat: add your feature"`
4. Push branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

> ⚠️ **Disclaimer**: This software is for educational and research purposes only.
> It does not constitute financial advice. Always do your own research before
> making investment decisions. Past performance does not guarantee future results.
