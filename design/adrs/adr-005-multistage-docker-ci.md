# ADR-005: Multi-stage Dockerfile for CI testing vs production

**Status:** Accepted
**Date:** 2026-02-16
**Deciders:** Edge AI Engineering Team

## Context

We need to run `pytest` in CI inside the same Docker environment that ships
to edge devices, ensuring tests validate the real dependency tree. However,
the production image must not contain test tooling (`pytest`, test fixtures).

## Decision

Use a **multi-stage Dockerfile** with three named stages:

```
base  → shared OS deps + pip packages + application code
test  → base + pytest + tests/  (CI only, never shipped)
prod  → base + ENTRYPOINT       (deployed to edge)
```

CI builds and runs the `test` target. Production builds use `--target prod`
(or default, since `prod` is last).

## Rationale

- **Parity.** The `test` stage inherits the exact same `base` layer as
  `prod`. If a test passes in CI, the same Python packages and OS libs are
  present on the edge device.
- **Image size.** `pytest` and its transitive dependencies (~5 MB) never
  appear in the production image.
- **Caching.** The `base` layer is shared across both targets. Docker layer
  caching means CI re-builds are fast — only the test layer needs to be
  rebuilt when test files change.

## Trade-offs accepted

- **Slightly more complex Dockerfile.** Three stages instead of one, but
  the complexity is well-understood and documented in comments.
- **CI must specify `--target test` explicitly.** Forgetting this builds
  the `prod` image, which has no `pytest`. Mitigated by the GitHub Actions
  workflow hardcoding the target.

## Consequences

- `Dockerfile` defines `base`, `test`, and `prod` stages.
- `.github/workflows/ci.yml` runs `docker build --target test` then
  `docker run` to execute tests.
- `requirements-test.txt` extends `requirements.txt` with `pytest`.
- The `prod` stage contains zero test-related files or packages.
