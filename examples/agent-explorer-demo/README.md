# Agent Explorer Demo

End-to-end demo agent that:

1. Provisions an **Alchemy Agent Wallet** (onchain / x402 identity — parallel)
2. Generates a separate **Ed25519** key and **registers** on PymtHouse
3. Authenticates to hosted **Livepeer MCP** with the minted API key
4. Calls `create_signer_session`, then submits an **LV2V** job via `livepeer_gateway`

Alchemy **cannot** sign the PymtHouse Ed25519 challenge. Keep the identities separate;
both are stored under `.agent-demo/`.

## Prerequisites

### livepeer-python-gateway (this repo)

Use a **token-capable** tree (`upstream/main` or equivalent) so `--token` /
`parse_token` / `signer_headers` work. This demo package lives on branch
`demo/agent-explorer` (branched from `upstream/main`).

### PymtHouse

Needs local/staging with:

- Agent network register (`GET/POST /api/v1/network/register*`) always available
  (PR [#334](https://github.com/pymthouse/pymthouse/pull/334); feature flag removed)
- Hosted MCP at `/api/v1/mcp` (PR [#300](https://github.com/pymthouse/pymthouse/pull/300))
- Healthy signer + discovery (`NEXTAUTH_URL`, `SIGNER_INTERNAL_URL`, `DISCOVERY_URL`)

Both PRs conflict with `main` — rebase onto a local demo branch before e2e.

### Alchemy CLI (bootstrap only)

```bash
npm i -g @alchemy/cli@latest
alchemy auth login                 # human once (browser)
# Or device-code (SSH / WSL without a local browser):
alchemy auth login --device-code   # open verificationUriComplete; approve in Alchemy dashboard
alchemy wallet connect --mode session --instance-name agent-explorer-demo
alchemy wallet status --verify
```

#### Arbitrum networks (Agent Wallets)

Agent Wallets support **Arbitrum One** and **Arbitrum Sepolia** (Wallet APIs:
bundler + gas sponsorship). CLI network slugs:

| Network | CLI slug (`-n`) | Admin allowlist ID | Chain ID |
| --- | --- | --- | --- |
| Arbitrum One | `arb-mainnet` | `ARB_MAINNET` | 42161 |
| Arbitrum Sepolia | `arb-sepolia` | `ARB_SEPOLIA` | 421614 |

The demo defaults to `ALCHEMY_NETWORK=arb-mainnet` (aligned with PymtHouse
Livepeer signing on Arbitrum One). Bootstrap checks the selected app allowlist
and fetches the session wallet native balance on that network.

```bash
# Confirm / enable on the selected Alchemy app
alchemy app configured-networks
alchemy app networks <appId> --networks ARB_MAINNET,ARB_SEPOLIA

# Manual balance check (CLI requires explicit -n; ALCHEMY_NETWORK alone is not enough)
alchemy --json --no-interactive evm data balance <0xSessionAddress> -n arb-mainnet
```

**Dashboard:** [dashboard.alchemy.com](https://dashboard.alchemy.com) → **Apps** →
select app (e.g. PymtHouse) → **Networks** → enable **Arbitrum Mainnet** and
optionally **Arbitrum Sepolia**. Agent Wallet session approval itself is
chain-agnostic for EVM; the app network allowlist is what gates RPC/balance.

For a real `publish_url` without local Docker signer, set in `.env`:

```bash
SIGNER_URL=https://signer.pymthouse.com
DISCOVERY_URL=https://signer.pymthouse.com/discover-orchestrators
MODEL_ID=streamdiffusion-sdxl   # production has no noop orch
```

That rewrites the `sdk_token` signer/discovery while register + MCP stay on
`PYMTHOUSE_BASE_URL`. **Payment auth requires matching OIDC issuers**: a local
PymtHouse API key/JWT is rejected by the production signer
(`subject_token is not a valid access token for this issuer`). Until agent
register/MCP ship on `https://pymthouse.com`, use local signer-dmz instead:

```bash
# in pymthouse repo
docker compose up -d signer-dmz
# in demo .env: comment out SIGNER_URL / DISCOVERY_URL
```

Alternatively leave `SIGNER_URL` unset so MCP’s `SIGNER_INTERNAL_URL` is used as-is.

## Setup

```bash
cd examples/agent-explorer-demo
cp .env.example .env
# edit PYMTHOUSE_BASE_URL, optional ORCHESTRATOR / MODEL_ID
uv sync
```

## CLI

```bash
uv run agent-explorer-demo bootstrap      # Alchemy session wallet
uv run agent-explorer-demo register       # Ed25519 challenge → apiKey
uv run agent-explorer-demo mcp-session    # hosted MCP + create_signer_session
uv run agent-explorer-demo job            # LV2V noop via livepeer_gateway --token
uv run agent-explorer-demo run-all        # all steps + status report
uv run agent-explorer-demo status
# optional: uv run agent-explorer-demo --state-dir /path/to/dir status
```

`run-all` prints Alchemy address, `externalUserId`, MCP tools ok, and
`publish_url` / job error.

## Parallel identities

| Identity | Purpose | Material |
| --- | --- | --- |
| Alchemy Agent Wallet | Onchain / x402 via CLI | Privy-backed session (no key to agent) |
| PymtHouse agent | Network API key + MCP + signer | Local Ed25519; `app_<24hex>_<secret>` |

## Job path

Hosted MCP does **not** run jobs. Execution stays in `livepeer_gateway`:

- Prefer `sdk_token` from MCP `create_signer_session`
- Fall back to register-time `sdkToken`
- Default model: `MODEL_ID=noop` (short frame burst, same pattern as `examples/write_frames.py`)

## Out of scope (v1)

- Bridging Alchemy address into PymtHouse identity
- Funding Alchemy / x402 settlement against Livepeer
- Local `comfypeer-mcp` execution tools (hosted MCP + gateway is enough)
