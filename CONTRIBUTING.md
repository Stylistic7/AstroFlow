# Contributing to AstroFlow

AstroFlow is licensed under the Apache License 2.0 (see `LICENSE`). By submitting a
contribution you agree that it is licensed under the same terms, as section 5 of the
license says. Sign each commit off (`git commit -s`) to certify the
[Developer Certificate of Origin](https://developercertificate.org/). A pseudonym is
fine for the sign-off and for your GitHub account; the maintainer uses one too.

## Setting up

    uv sync --frozen                 # Python 3.12 environment, pinned by uv.lock
    cd core/web_server/frontend && npm ci && npm run build

The app itself runs on macOS and Windows with the tools it wraps installed
(`make start` in `core/`); see the node settings page in the app for what it finds.

## Tests

Three tiers, from `core/`:

| tier | command | needs |
|---|---|---|
| unit | `make tests-unit` | nothing beyond the environment |
| integration | `python3 testdata/fetch.py` then `make tests` | Siril 1.4 and the fixtures (3.1 GB, fetched once into a local cache) |
| studio | `make tests-exhaustive` | PixInsight and the RC Astro tools; the maintainer's machine |

Pull requests run the unit tier and the frontend build; pushes to `main` and a nightly
run add the integration tier. Never run two pytest invocations at once: the suite shares
a Ray cluster.

## How the code is written

- Every line must justify itself. No backward-compatibility shims, no dead code, no
  flags that keep an old path alive.
- Objects are identified by content hash, never by name. A node's cache key is its
  body plus its inputs, so editing a `@node` body re-keys every product downstream of
  it; say so in the commit message when you do it on purpose.
- `@node` functions are pure and cached; `@ray.remote` builders may read the context.
- Comments and commit messages describe the decision and the date it was made, not the
  person who made it.
