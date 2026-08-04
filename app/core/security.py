import os
from datetime import datetime, UTC, timedelta
from typing import Optional, Dict, Any
import jwt
from jwt.exceptions import InvalidTokenError
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, Header, status
from fastapi.security import OAuth2PasswordBearer, APIKeyHeader
from app.core.config import settings

# Authentication configuration
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)
api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=False)

ALGORITHM = "HS256"

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(UTC) + expires_delta
    else:
        expire = datetime.now(UTC) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def verify_auth(
    token: Optional[str] = Depends(oauth2_scheme),
    api_key: Optional[str] = Depends(api_key_header),
    x_vit_service_token: Optional[str] = Header(None),
) -> Dict[str, Any]:
    # ── 1. HMAC inter-service token (vitnetwork → vit-ai) ────────────────────
    if x_vit_service_token:
        from app.core.service_auth import verify_service_token
        if verify_service_token(x_vit_service_token):
            return {"sub": "internal_service", "scopes": ["full"], "auth": "hmac"}

    # ── 2. Static API key (backward compat / fallback) ───────────────────────
    internal_key = settings.VIT_AI_API_KEY or os.getenv("INTERNAL_API_KEY", "")
    if api_key and internal_key and api_key == internal_key:
        return {"sub": "internal_service", "scopes": ["full"], "auth": "api_key"}

    # ── 3. JWT Bearer token ───────────────────────────────────────────────────
    if token:
        try:
            payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[ALGORITHM])
            username: str = payload.get("sub")
            if username is None:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
            return payload
        except InvalidTokenError:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate credentials")

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Not authenticated",
        headers={"WWW-Authenticate": "Bearer or X-API-KEY or X-VIT-Service-Token"},
    )
