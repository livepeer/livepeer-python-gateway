from __future__ import annotations

from dataclasses import dataclass

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from .state import StateStore


@dataclass(frozen=True)
class Ed25519Keypair:
    private_key_hex: str
    public_key_hex: str


def generate_keypair() -> Ed25519Keypair:
    private_key = Ed25519PrivateKey.generate()
    private_bytes = private_key.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_bytes = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return Ed25519Keypair(
        private_key_hex=private_bytes.hex(),
        public_key_hex=public_bytes.hex(),
    )


def load_or_create_keypair(store: StateStore) -> Ed25519Keypair:
    existing = store.read_private_key_hex()
    if existing:
        private_key = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(existing))
        public_bytes = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        return Ed25519Keypair(
            private_key_hex=existing.lower(),
            public_key_hex=public_bytes.hex(),
        )

    keypair = generate_keypair()
    store.write_private_key_hex(keypair.private_key_hex)
    return keypair


def sign_nonce_hex(private_key_hex: str, nonce: str) -> str:
    """Sign UTF-8 nonce bytes with Ed25519; return 64-byte signature as hex."""
    private_key = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(private_key_hex))
    signature = private_key.sign(nonce.encode("utf-8"))
    return signature.hex()


def public_key_from_private_hex(private_key_hex: str) -> str:
    private_key = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(private_key_hex))
    public_bytes = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return public_bytes.hex()


def verify_signature_hex(public_key_hex: str, nonce: str, signature_hex: str) -> bool:
    public_key = Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_key_hex))
    try:
        public_key.verify(bytes.fromhex(signature_hex), nonce.encode("utf-8"))
        return True
    except Exception:
        return False
