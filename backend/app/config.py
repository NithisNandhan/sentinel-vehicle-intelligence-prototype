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
    # Off by default. When on, "no recording is being analysed" after Stop is a 404
    # (which the dashboard treats as empty), not a 503 "service unavailable".
    recognition_idle_evidence_404: bool = False
    # Result states and review (docs/SECURITY_AND_PRIVACY.md). Reviewer tokens are
    # "reviewer-id:sha256-hex-of-token" pairs, comma separated; empty disables review.
    reviewer_tokens: str = ""
    # Off by default: proxy edge-held review evidence (crops, frame, full partial
    # pattern) to authorised reviewers. Never stored centrally.
    review_evidence_enabled: bool = False
    retention_confirmed_days: int = 90
    retention_partial_days: int = 30
    retention_unreadable_days: int = 30
    retention_review_audit_days: int = 365
    # Watchlist snapshot lifetime on the edge, and device revocation.
    watchlist_snapshot_ttl_seconds: int = 900
    revoked_edge_ids: str = ""
    require_edge_id: bool = False
    local_dashboard_mode: bool = False
    local_dashboard_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    local_dashboard_bind_host: str = "127.0.0.1"
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def cors_origin_list(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]

    @property
    def revoked_edge_id_set(self) -> frozenset[str]:
        return frozenset(item.strip() for item in self.revoked_edge_ids.split(",") if item.strip())

    @property
    def reviewer_token_map(self) -> dict[str, str]:
        """sha256(token) -> reviewer id. Malformed entries are ignored, never logged."""
        result = {}
        for item in self.reviewer_tokens.split(","):
            reviewer, _, digest = item.strip().partition(":")
            digest = digest.strip().lower()
            if reviewer.strip() and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest):
                result[digest] = reviewer.strip()[:64]
        return result

    @property
    def local_origin_list(self) -> list[str]:
        return [item.strip() for item in self.local_dashboard_origins.split(",") if item.strip()]

    @model_validator(mode="after")
    def retention_is_bounded(self):
        for name in ("retention_confirmed_days", "retention_partial_days", "retention_unreadable_days",
                     "retention_review_audit_days"):
            if not 1 <= getattr(self, name) <= 3650:
                raise ValueError(name + " must be between 1 and 3650 days")
        if not 30 <= self.watchlist_snapshot_ttl_seconds <= 3600:
            raise ValueError("watchlist_snapshot_ttl_seconds must be between 30 and 3600")
        return self

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

