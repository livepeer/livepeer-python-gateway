from livepeer_gateway.errors import (
    NoOrchestratorAvailableError,
    OrchestratorRejection,
)


def test_no_orchestrator_available_error_without_rejections() -> None:
    error = NoOrchestratorAvailableError("No orchestrators available to select")

    assert str(error) == "No orchestrators available to select"


def test_no_orchestrator_available_error_includes_rejections() -> None:
    error = NoOrchestratorAvailableError(
        "All orchestrators failed (2 tried)",
        rejections=[
            OrchestratorRejection(
                url="https://orch-a.example.com", reason="connection refused"
            ),
            OrchestratorRejection(
                url="https://orch-b.example.com", reason="request timed out"
            ),
        ],
    )

    assert str(error) == (
        "All orchestrators failed (2 tried): "
        "https://orch-a.example.com: connection refused; "
        "https://orch-b.example.com: request timed out"
    )
