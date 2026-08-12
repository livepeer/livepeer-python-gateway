# Changelog

All notable changes to this project will be documented in this file. The
format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

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

[1.0.0]: https://github.com/livepeer/livepeer-python-gateway/releases/tag/v1.0.0
