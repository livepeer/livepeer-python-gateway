"""Exception types raised by the Livepeer gateway SDK."""

from __future__ import annotations

from dataclasses import dataclass


class LivepeerGatewayError(RuntimeError):
    """Base error for the library."""


class LivepeerHTTPError(LivepeerGatewayError):
    """A non-success HTTP response from an endpoint.

    Attributes:
        status_code: HTTP status code returned by the endpoint.
        url: Endpoint that produced the error.
        body: Raw response body, when available.
    """

    def __init__(self, status_code: int, url: str, body: str = "", message: str | None = None) -> None:
        self.status_code = int(status_code)
        self.url = url
        self.body = body
        super().__init__(message or f"HTTP {status_code} from endpoint (url={url})")


@dataclass
class OrchestratorRejection:
    """A single orchestrator that was tried during selection and rejected.

    Attributes:
        url: Endpoint URL of the orchestrator that was attempted.
        reason: Human-readable explanation of why it was rejected.
    """
    url: str
    reason: str


@dataclass
class RunnerRejection:
    """A single runner that was tried during selection and rejected.

    Attributes:
        url: Endpoint URL of the runner that was attempted.
        reason: Human-readable explanation of why it was rejected.
    """
    url: str
    reason: str


class NoOrchestratorAvailableError(LivepeerGatewayError):
    """No orchestrator is available; every candidate was rejected during selection.

    Attributes:
        rejections: Each orchestrator that was tried and why it was rejected.
    """

    def __init__(self, message: str, rejections: list[OrchestratorRejection] | None = None) -> None:
        super().__init__(message)
        self.rejections: list[OrchestratorRejection] = rejections or []

    def __str__(self) -> str:
        message = super().__str__()
        if not self.rejections:
            return message
        reasons = "; ".join(f"{r.url}: {r.reason}" for r in self.rejections)
        return f"{message}: {reasons}"


class NoRunnerAvailableError(LivepeerGatewayError):
    """No runner is available; every candidate was rejected during selection.

    Attributes:
        rejections: Each runner that was tried and why it was rejected.
    """

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
    """The remote signer requires a credential refresh.

    Attributes:
        orchestrator_url: Orchestrator whose signer requested the refresh, if known.
    """

    def __init__(
        self,
        message: str,
        *,
        orchestrator_url: str | None = None,
    ) -> None:
        super().__init__(message)
        self.orchestrator_url = orchestrator_url


class SkipPaymentCycle(LivepeerGatewayError):
    """A signer HTTP 482 response requesting that a payment cycle be skipped."""


class PaymentError(LivepeerGatewayError):
    """A failed PaymentSession operation."""
