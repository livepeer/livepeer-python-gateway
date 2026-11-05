# Fleet CI and deployment freeze

CI tests source changes. The fleet builder in `livepeer/simple-infra` builds and publishes deployment images.
This PR does not authorize a staging or production deployment.

## Source contract

The receiver is `agent-fleet-build.yaml` on `livepeer/simple-infra` main.
Inputs are `source_repository`, `source_sha` (full commit SHA), and `ci_run_id`.
`.github/fleet-ci.json` identifies this repository's trusted source workflow.
The receiver must independently check that the run succeeded for this repository, workflow, SHA, main branch, and push event.
Use the exact SHA from the run. A newer main commit does not invalidate an older successful run.

## Freeze

`FLEET_CI_NOTIFY_ENABLED` is false in this repository.
Missing or false disables notification before the GitHub App receives a token.
`FLEET_DEPLOYMENTS_ENABLED` is false in simple-infra. The new CD workflows must enforce it.
The new flag does not disable existing manual workflows. Any Grafana staging work remains separately authorized.
Keep this PR in draft. Do not merge it or enable notifications during the freeze.

## Activation after the freeze

1. Merge the reviewed CI PR after its required checks pass.
2. Check that post-merge CI reports success for that exact commit.
3. Confirm Pulumi's receiver validates the source contract and refuses deployment while its gate is false.
4. Configure `FLEET_CI_APP_ID` and `FLEET_CI_APP_PRIVATE_KEY`. The App needs Actions write on simple-infra.
5. Enable source notifications only after Emran authorizes activation. Production remains manual in the CD workflow.


For other repositories, retain their existing required checks until the new check is observed on a PR.
Do not enable auto-merge or change release tag rules as part of this freeze preparation.
