# Changelog

All notable changes to this project will be documented in this file. The
format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `call_runner` accepts a raw request body: `content` (bytes) with an optional
  `content_type` (default `application/octet-stream`), as an alternative to the JSON
  `payload`. Use it for apps that take an image, audio or other binary input.

### Removed

- gRPC orchestrator information and selection including protobuf bindings
- LV2V and Scope startup APIs, including 720p pricing
- BYOC (bring your own container) inference and training jobs: the
  `submit_byoc_job`, `submit_training_job`, `refresh_training_payment`,
  `get_training_status`, `wait_for_training`, and `list_capabilities`
  functions, and the `ByocJobRequest`, `ByocJobResponse`,
  `ByocTrainingRequest`, `ByocTrainingResponse`, and `ByocTrainingStatus`
  types. BYOC is being removed from go-livepeer in
  [livepeer/go-livepeer#4064](https://github.com/livepeer/go-livepeer/pull/4064).

## [1.0.0] - 2026-08-11

The first stable release of the Livepeer Python SDK.

### Added

- Live Runner registration, discovery, session reservation, raw calls, proxy
  calls, and session lifecycle events.
- Scope startup for application and serverless runners.
- BYOC inference and training jobs, including signed payments, payment refresh,
  status polling, and completion waits.
- Live video-to-video jobs with capability-aware orchestrator discovery,
  ordered fallback selection, token-based configuration, and remote-signer
  payments.
- Multi-track media publishing and media output APIs for bytes, decoded frames,
  and demuxed packets.
- Trickle channels for control messages, events, JSON Lines, keepalives, and
  observable publisher/subscriber statistics.
- Orchestrator information, capability discovery, TLS trust-on-first-use, and
  typed SDK errors.

### Changed

- Declared the generated gRPC client's actual minimum runtime versions:
  `grpcio>=1.76.0` and `protobuf>=6.31.1`.
- Completed the package metadata and documented installation from PyPI.

[Unreleased]: https://github.com/livepeer/livepeer-python-gateway/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/livepeer/livepeer-python-gateway/releases/tag/v1.0.0
