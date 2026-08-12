# Publishing releases

Releases are built and published by
[`publish-to-pypi.yml`](.github/workflows/publish-to-pypi.yml). The workflow
uses PyPI Trusted Publishing, so no PyPI API token is stored in GitHub.

## Production release

1. Update the version in `pyproject.toml` and `uv.lock`.
2. Add the dated release notes to `CHANGELOG.md`.
3. Merge the release commit into `main` and ensure the test workflow passes.
4. Create and push an annotated tag matching the package version:

   ```bash
   git switch main
   git pull --ff-only github main
   git tag -a v1.0.0 -m "Livepeer Python SDK 1.0.0"
   git push github v1.0.0
   ```

5. Approve the deployment in the protected `pypi` GitHub Environment.
6. Verify the release at <https://pypi.org/project/livepeer-gateway/>.

The workflow rejects tags that do not exactly match `v` followed by the
version in `pyproject.toml`. The publishing action generates and uploads PyPI
digital attestations for both release files.

## TestPyPI

Run the `Publish Python release` workflow manually from the Actions tab.
Manual runs build and upload only to TestPyPI. Each package version can be
uploaded only once, so bump to a new pre-release version before repeating a
TestPyPI upload.

<details>
<summary>Initial repository setup (maintainers only)</summary>

Create these GitHub Environments in the repository settings:

- `pypi`: add a required reviewer and restrict deployment to version tags.
- `testpypi`: manual approval is optional because this target is only available
  through a manually dispatched workflow.

Register a pending GitHub Trusted Publisher at
<https://pypi.org/manage/account/publishing/> with:

| Field | Value |
|---|---|
| PyPI project name | `livepeer-gateway` |
| Owner | `livepeer` |
| Repository | `livepeer-python-gateway` |
| Workflow | `publish-to-pypi.yml` |
| Environment | `pypi` |

Optionally register the same pending publisher at
<https://test.pypi.org/manage/account/publishing/>, using the `testpypi`
environment. PyPI and TestPyPI require separate accounts and publisher setup.

The workflow file must be present on the repository's default branch before a
Trusted Publisher can use it.

</details>
