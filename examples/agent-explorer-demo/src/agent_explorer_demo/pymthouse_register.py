from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import urlencode

import httpx

from .ed25519_keys import load_or_create_keypair, sign_nonce_hex
from .state import RegisterState, StateStore


class RegisterError(RuntimeError):
    """PymtHouse network agent registration failure."""

    def __init__(
        self,
        message: str,
        *,
        status: Optional[int] = None,
        code: Optional[str] = None,
        body: Any = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.body = body


@dataclass(frozen=True)
class ChallengeResponse:
    challenge_id: str
    nonce: str
    expires_at: Optional[str]
    alg: str


def _base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def fetch_challenge(
    client: httpx.Client,
    *,
    base_url: str,
    public_key_hex: str,
) -> ChallengeResponse:
    query = urlencode({"publicKey": public_key_hex})
    url = f"{_base_url(base_url)}/api/v1/network/register/challenge?{query}"
    response = client.get(url)
    if response.status_code >= 400:
        raise RegisterError(
            f"Challenge failed: {response.status_code} {response.text}",
            status=response.status_code,
            body=_safe_json(response),
        )
    body = response.json()
    challenge_id = body.get("challengeId")
    nonce = body.get("nonce")
    if not isinstance(challenge_id, str) or not challenge_id:
        raise RegisterError("Challenge response missing challengeId", body=body)
    if not isinstance(nonce, str) or not nonce:
        raise RegisterError("Challenge response missing nonce", body=body)
    return ChallengeResponse(
        challenge_id=challenge_id,
        nonce=nonce,
        expires_at=body.get("expiresAt") if isinstance(body.get("expiresAt"), str) else None,
        alg=str(body.get("alg") or "Ed25519"),
    )


def post_register(
    client: httpx.Client,
    *,
    base_url: str,
    public_key_hex: str,
    challenge_id: str,
    signature_hex: str,
    label: Optional[str],
) -> dict[str, Any]:
    url = f"{_base_url(base_url)}/api/v1/network/register"
    payload: dict[str, Any] = {
        "publicKey": public_key_hex,
        "challengeId": challenge_id,
        "signature": signature_hex,
    }
    if label:
        payload["label"] = label
    response = client.post(url, json=payload)
    body = _safe_json(response)
    if response.status_code == 409:
        raise RegisterError(
            "Public key already registered (409). Reuse the stored apiKey in "
            ".agent-demo/state.json, or generate a new Ed25519 keypair.",
            status=409,
            code=(body or {}).get("code") if isinstance(body, dict) else "conflict",
            body=body,
        )
    if response.status_code >= 400:
        message = "Register failed"
        if isinstance(body, dict) and isinstance(body.get("error"), str):
            message = body["error"]
        raise RegisterError(
            f"{message}: {response.status_code} {response.text}",
            status=response.status_code,
            code=(body or {}).get("code") if isinstance(body, dict) else None,
            body=body,
        )
    if not isinstance(body, dict):
        raise RegisterError("Register returned non-object JSON", body=body)
    return body


def register_agent(
    store: StateStore,
    *,
    base_url: str,
    label: Optional[str] = None,
    timeout: float = 30.0,
) -> RegisterState:
    """Ed25519 challenge → register. Persists one-time apiKey + sdkToken."""
    state = store.load()
    if state.register.api_key:
        return state.register

    keypair = load_or_create_keypair(store)
    with httpx.Client(timeout=timeout) as client:
        # Retry once: Next.js cold-compiling the POST route can reload the
        # in-memory challenge map between GET challenge and POST register.
        body: dict[str, Any] | None = None
        last_error: RegisterError | None = None
        for attempt in range(2):
            challenge = fetch_challenge(
                client,
                base_url=base_url,
                public_key_hex=keypair.public_key_hex,
            )
            signature = sign_nonce_hex(keypair.private_key_hex, challenge.nonce)
            try:
                body = post_register(
                    client,
                    base_url=base_url,
                    public_key_hex=keypair.public_key_hex,
                    challenge_id=challenge.challenge_id,
                    signature_hex=signature,
                    label=label,
                )
                break
            except RegisterError as exc:
                last_error = exc
                if (
                    attempt == 0
                    and exc.status == 400
                    and (exc.code == "invalid_challenge" or "challenge" in str(exc).lower())
                ):
                    continue
                raise
        if body is None:
            assert last_error is not None
            raise last_error

    api_key = body.get("apiKey")
    if not isinstance(api_key, str) or not api_key:
        raise RegisterError("Register response missing apiKey", body=body)

    register = RegisterState(
        public_key_hex=keypair.public_key_hex,
        client_id=body.get("clientId") if isinstance(body.get("clientId"), str) else None,
        external_user_id=(
            body.get("externalUserId")
            if isinstance(body.get("externalUserId"), str)
            else None
        ),
        api_key=api_key,
        sdk_token=body.get("sdkToken") if isinstance(body.get("sdkToken"), str) else None,
        key_id=body.get("id") if isinstance(body.get("id"), str) else None,
        label=body.get("label") if isinstance(body.get("label"), str) else label,
    )
    state = store.load()
    state.register = register
    store.save(state)
    return register


def _safe_json(response: httpx.Response) -> Any:
    try:
        return response.json()
    except Exception:
        return response.text
