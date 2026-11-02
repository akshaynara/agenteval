# Releasing

## One-time setup (about 5 minutes)

1. **PyPI trusted publisher.** Sign in at [pypi.org](https://pypi.org), go to *Your account → Publishing → Add a new pending publisher*, and enter:
   - PyPI project name: `agenteval-cx`
   - Owner: `akshaynara` · Repository: `agenteval`
   - Workflow name: `release.yml` · Environment: `pypi`
2. **GitHub environment.** In the repo: *Settings → Environments → New environment* named `pypi`. Optionally add yourself as a required reviewer so every publish waits for your click.

No API token is stored anywhere; PyPI trusts this workflow directly.

## Each release

1. Bump `version` in `pyproject.toml` and `__version__` in `src/agenteval/__init__.py`.
2. Add a `## X.Y.Z — date` section at the top of `CHANGELOG.md`.
3. Merge to `main`, then tag and push:

   ```bash
   git tag v1.0.0 && git push origin v1.0.0
   ```

The Release workflow checks the tag matches the version, builds, publishes to PyPI, creates the GitHub release with the changelog section as notes, and moves the `v1` tag so `uses: akshaynara/agenteval@v1` picks up the new version.

## Listing the GitHub Action on the Marketplace (optional)

Open the GitHub release, tick *Publish this Action to the GitHub Marketplace*, and accept the terms. `action.yml` already has the name, description and branding.
