# Enterprise Git Workflow & Collaboration Rules

> **Standard operating procedures for version control, commit conventions, branch governance, code reviews, and release lifecycle in the ML-Autonomus engineering team.**

---

## 1. Branching Strategy: Trunk-Based Development

We use **Trunk-Based Development** with short-lived feature branches. All development merges into `main` through Pull Requests backed by continuous integration.

```text
main ───────────────────────●──────────────●──────────────●─── (Always Deployable / Tagged)
                             ▲              ▲              ▲
                              \ [PR Merge]   \ [PR Merge]   \ [PR Merge]
feat/nlp-sentiment ────────────┘              │              │
fix/oov-tokenizer ─────────────────────────────┘              │
docs/roadmap-v3 ──────────────────────────────────────────────┘
```

### Branch Naming Conventions
All branch names must be lowercase, hyphen-separated, and prefixed with the appropriate category:

| Prefix | Usage | Example |
|---|---|---|
| `feat/` | New functional features | `feat/nlp-text-profiler`, `feat/word-attribution` |
| `fix/` | Bug fixes and logical corrections | `fix/frequency-encoder-nan`, `fix/login-seed-401` |
| `refactor/` | Code refactoring without behavior change | `refactor/storage-adapter-sigv4` |
| `perf/` | Performance optimizations | `perf/cv-multiprocessing-speed` |
| `test/` | Adding or refactoring test suites | `test/api-contract-sentiment` |
| `docs/` | Documentation additions or updates | `docs/roadmap-v3-sentiment` |
| `chore/` | CI/CD, dependency bumps, tooling | `chore/github-actions-ci` |
| `hotfix/` | Urgent production fixes directly branched | `hotfix/cors-origin-header` |

*Branches should live no longer than 2–3 days to prevent merge conflicts.*

---

## 2. Commit Message Standard: Conventional Commits v1.0.0

Every commit in this repository must strictly adhere to the [Conventional Commits](https://www.conventionalcommits.org/) specification.

### Commit Format
```text
<type>(<optional scope>): <imperative summary>

[optional body explaining motivation, evidence, and architectural tradeoffs]

[optional footer: e.g. BREAKING CHANGE: or Closes #123]
```

### Types
* `feat`: A new user-facing or platform feature.
* `fix`: A bug fix or error correction.
* `docs`: Documentation changes only (e.g. Markdown files in `docs/`).
* `style`: Changes that do not affect code logic (whitespace, formatting, missing semi-colons).
* `refactor`: Code changes that neither fix a bug nor add a feature.
* `perf`: Code changes improving computational performance or latency.
* `test`: Adding missing tests or correcting existing tests.
* `chore`: Maintenance tasks, dependencies, build configurations (`pyproject.toml`, `package.json`, CI).

### Golden Rules for Commit Messages
1. **Imperative Mood**: Use `"add"`, `"fix"`, `"update"`, `"remove"`, **not** `"added"`, `"fixing"`, or `"adds"`.
2. **Capitalization**: Lowercase type and summary (e.g. `feat(storage): support minio s3 sigv4`).
3. **No Trailing Period**: Do not end the summary line with a period.
4. **Character Limit**: Keep the first line under 72 characters.
5. **Atomic Commits**: One commit per cohesive logical change. Do not bundle unrelated frontend CSS changes with backend database schema updates.

### Good vs Bad Examples

```text
# GOOD
feat(nlp): implement text profiler with lexical diversity metrics
fix(auth): auto-seed default admin on startup to prevent 401
docs(git): create enterprise git rules and branching guidelines
test(jobs): add unit tests for async worker cancellation

# BAD
Fixed stuff
wip
update code
feat: made the sentiment analysis work and also changed the button color and fixed database bug.
```

---

## 3. Pull Request (PR) & Code Review Guidelines

Direct pushes to `main` are strictly forbidden. All code enters `main` via reviewed Pull Requests.

### Pull Request Checklist
Before requesting review, the author must ensure:
1. **Target Branch**: PR targets `main`.
2. **Title**: Follows Conventional Commits (e.g. `feat(sentiment): add text cleaner and tfidf pipeline`).
3. **PR Description**: Must include:
   - **Problem / Context**: Why is this change necessary?
   - **Solution**: What architectural approach was chosen?
   - **Verification**: Exact test commands executed and results observed.
   - **Breaking Changes**: Any database migrations or contract changes?
4. **CI Status**: All automated CI checks (FastAPI pytest suite, React Vite build, linting) must pass with green status.
5. **No Secrets / Artifacts**: Ensure no `.env`, SQLite DB files (`*.db`), or training artifacts are staged.

### Reviewer Checklist (Code Review Standard)
Reviewers must evaluate PRs against:
* **Correctness**: Does the algorithm handle edge cases, empty datasets, and NaN values?
* **Zero Leakage**: Are scikit-learn transformers fitted strictly inside training folds?
* **Maintainability**: Boring design over premature abstraction; no dead code or unnecessary wrappers.
* **Typing Precision**: No usage of bare `any` in TypeScript or untyped signatures in Python APIs.

### Merge Policy
* Use **Squash and Merge** for feature and bugfix branches to maintain a clean, linear git history on `main`.
* Rebase feature branches before merge (`git pull --rebase origin main`) to resolve any divergent history cleanly.

---

## 4. Git Hygiene & Protected Files

### Never Commit (`.gitignore` Enforced)
* Environment secrets: `.env`, `.env.local`, API keys, private certificates.
* Local databases: `automl.db`, `mlflow.db`, `*.sqlite`.
* Storage blobs: `storage/artifacts/`, `storage/datasets/`, `mlartifacts/`, `mlruns/`.
* Dependency directories: `node_modules/`, `__pycache__/`, `.pytest_cache/`, `dist/`.

### Secret Leak Protocol
If a secret or credential is inadvertently committed:
1. Immediately rotate the credential at the provider level (MinIO, JWT secret, AWS key).
2. Do **not** merely commit a deletion; use `git filter-repo` or BFG Repo-Cleaner to expunge the hash from history.
3. Notify the engineering team.

---

## 5. Release & Tagging Lifecycle (Semantic Versioning 2.0.0)

Production releases are tagged using `vMAJOR.MINOR.PATCH`:
* **MAJOR** (`v2.0.0`): Incompatible API contract changes or fundamental architecture overhauls.
* **MINOR** (`v1.1.0`, `v1.2.0`): Backward-compatible new capabilities (e.g. NLP Sentiment Analysis, Multi-Tenancy).
* **PATCH** (`v1.1.1`): Backward-compatible bug fixes and stability hardening.

### Tagging Commands
```bash
# Create annotated tag
git tag -a v1.1.0 -m "Release v1.1.0: Enterprise Multi-Tenancy, S3 Storage, Async Jobs & Meta-Ranker"

# Push tag to remote
git push origin v1.1.0
```
