from functools import lru_cache
from urllib.parse import urlsplit

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "sqlite:///./sentinel.db"
    edge_api_key: str
    operator_api_key: str
    watchlist_hmac_key: str
    demo_mode: bool = False
    catalogue_gateway_url: str = ""
    recognition_gateway_url: str = ""
    # Off by default. When on, the dashboard may show the analysed frame and full
    # unverified readings for permitted RECORDED footage only, never a live camera.
    dashboard_recorded_evidence: bool = False
    local_dashboard_mode: bool = False
    local_dashboard_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    local_dashboard_bind_host: str = "127.0.0.1"
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def cors_origin_list(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]

    @property
    def local_origin_list(self) -> list[str]:
        return [item.strip() for item in self.local_dashboard_origins.split(",") if item.strip()]

    @model_validator(mode="after")
    def local_profile_is_loopback(self):
        if self.local_dashboard_mode:
            if self.local_dashboard_bind_host not in {"localhost", "127.0.0.1", "::1"}:
                raise ValueError("Local dashboard requires a loopback published listener")
            if not self.local_origin_list:
                raise ValueError("Local dashboard requires explicit local UI origins")
            for origin in self.local_origin_list:
                url = urlsplit(origin)
                if (url.scheme not in {"http", "https"} or url.hostname not in {"localhost", "127.0.0.1", "::1"}
                        or url.username or url.password or url.path or url.query or url.fragment):
                    raise ValueError("Local dashboard origins must be exact loopback origins")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()

