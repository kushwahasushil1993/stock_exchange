"""
Central configuration management using pydantic-settings.
All secrets loaded from environment variables / .env file.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import List, Optional
from functools import lru_cache


class DatabaseSettings(BaseSettings):
    postgres_url: str = Field(default="postgresql+asyncpg://finai:finai@localhost:5432/financeai")
    redis_url: str = Field(default="redis://localhost:6379/0")
    influx_url: str = Field(default="http://localhost:8086")
    influx_token: str = Field(default="")
    influx_org: str = Field(default="financeai")
    influx_bucket: str = Field(default="market_data")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


class APIKeys(BaseSettings):
    newsapi_key: str = Field(default="")
    alphavantage_key: str = Field(default="")
    fmp_api_key: str = Field(default="")       # Financial Modeling Prep
    polygon_api_key: str = Field(default="")
    nse_cookie: str = Field(default="")        # NSE India session cookie
    razorpay_key_id: str = Field(default="")
    razorpay_key_secret: str = Field(default="")
    razorpay_webhook_secret: str = Field(default="")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


class ModelSettings(BaseSettings):
    model_store_path: str = Field(default="models_store/")
    lstm_lookback: int = Field(default=60)
    lstm_forecast_horizon: int = Field(default=5)
    retrain_interval_hours: int = Field(default=24)
    prediction_confidence_threshold: float = Field(default=0.65)
    min_training_samples: int = Field(default=500)

    # `model_store_path` collides with pydantic's protected "model_" namespace
    # (used for methods like model_dump); silence the warning explicitly.
    model_config = SettingsConfigDict(
        protected_namespaces=("settings_",),
        env_file=".env",
        env_file_encoding="utf-8",
    )


class AppSettings(BaseSettings):
    app_name: str = "FinanceAI"
    version: str = "1.0.0"
    environment: str = Field(default="development")  # development | staging | production
    debug: bool = Field(default=False)
    secret_key: str = Field(default="change-me-in-production-32-char-secret")
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60
    allowed_origins: List[str] = Field(default=["http://localhost:3000"])

    # Market config
    default_exchange: str = "NSE"
    watchlist_nse: List[str] = Field(default=[
        "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
        "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "KOTAKBANK.NS",
        "LT.NS", "HCLTECH.NS", "WIPRO.NS", "AXISBANK.NS", "MARUTI.NS",
        "BAJFINANCE.NS", "TITAN.NS", "NESTLEIND.NS", "ULTRACEMCO.NS", "ASIANPAINT.NS"
    ])
    nifty50_index: str = "^NSEI"
    sensex_index: str = "^BSESN"
    payment_currency: str = Field(default="INR")

    # FNO eligible stocks
    fno_stocks: List[str] = Field(default=[
        "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "NIFTY",
        "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"
    ])

    # Mutual fund categories to track
    mf_categories: List[str] = Field(default=[
        "Large Cap", "Mid Cap", "Small Cap", "Flexi Cap",
        "ELSS", "Index Fund", "Sectoral/Thematic"
    ])

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


class Settings(AppSettings, DatabaseSettings, APIKeys, ModelSettings):
    model_config = SettingsConfigDict(
        protected_namespaces=("settings_",),
        env_file=".env",
        env_file_encoding="utf-8",
    )


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
