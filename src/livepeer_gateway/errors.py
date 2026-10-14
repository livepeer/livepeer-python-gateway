from __future__ import annotations

from dataclasses import dataclass


class LivepeerGatewayError(RuntimeError):
    """Base error for the library."""


class LivepeerHTTPError(LivepeerGatewayError):
    """Raised when an HTTP endpoint returns a non-success status."""

    def __init__(self, status_code: int, url: str, body: str = "", message: str | None = None) -> None:
        self.status_code = int(status_code)
        self.url = url
        self.body = body
        super().__init__(message or f"HTTP {status_code} from endpoint (url={url})")


@dataclass
class RunnerRejection:
    """Records a single runner that was tried and rejected."""
    url: str
    reason: str


class NoRunnerAvailableError(LivepeerGatewayError):
    """Raised when no runner could be selected."""

    def __init__(self, message: str, rejections: list[RunnerRejection] | None = None) -> None:
        super().__init__(message)
        self.rejections: list[RunnerRejection] = rejections or []

    def __str__(self) -> str:
        message = super().__str__()
        if not self.rejections:
            return message
        reasons = "; ".join(f"{r.url}: {r.reason}" for r in self.rejections)
        return f"{message}: {reasons}"


class SignerRefreshRequired(LivepeerGatewayError):
    """Raised when the remote signer returns HTTP 480 and a refresh is required."""


class SkipPaymentCycle(LivepeerGatewayError):
    """Raised when the signer returns HTTP 482 to skip a payment cycle."""


class PaymentError(LivepeerGatewayError):
    """Raised when a payment operation fails."""
