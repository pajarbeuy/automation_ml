# Security and Execution Model

## Threat model

Uploaded datasets are untrusted input.

The platform must assume:
- malformed files
- extremely large files
- malicious filenames
- unexpected encodings
- pathological data distributions

## Rules

1. Never execute user-uploaded Python.
2. Validate file type and size.
3. Store uploads outside executable directories.
4. Generate safe internal filenames.
5. Isolate ML execution where possible.
6. Restrict filesystem access for workers.
7. Keep secrets outside project artifacts.
8. Authorize artifact access by project ownership.
9. Do not send raw datasets to an external LLM by default.

## LLM privacy

If an LLM is enabled:
- send structured summaries by default
- do not send full datasets unless explicitly configured
- redact sensitive columns where possible
- document external data processing
