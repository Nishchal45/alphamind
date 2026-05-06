# 4. Storage layer for filing bodies

- **Status**: accepted
- **Date**: 2026-04-29

## Context

Phase 1 ingests SEC filing *metadata* — one row per filing in the `filings`
table — but not the document bodies. Phase 2 needs the bodies in order to
chunk, embed, and retrieve them.

Bodies are awkward:

- Size varies wildly. An 8-K is a few kilobytes. A 10-K with exhibits can
  be tens of megabytes.
- They're immutable for our purposes — once an accession number exists, the
  body at that URL doesn't change. (Amendments produce *new* accession
  numbers, which we treat as separate filings.)
- Postgres can technically hold them in a `BYTEA` or `TEXT` column. Doing
  so bloats the table page cache and makes every schema migration slower
  than it has any business being.

## Decision

Bodies live in object storage behind a narrow `StorageBackend` protocol —
``put``, ``get``, ``exists``. A small `FilingDocument` row in Postgres
records *where* the bytes are (URI), *what* they are (content hash, byte
size, MIME type, source URL, fetch timestamp). The bytes themselves never
touch a Postgres row.

Two backend implementations planned:

| Backend | Purpose | Status |
| --- | --- | --- |
| `LocalFilesystemStorage` | Dev, tests, single-machine deployments | Implemented |
| `S3CompatibleStorage` | Production (S3, MinIO, R2) | Deferred to Phase 3 |

The factory in `alphamind.storage.factory` reads `STORAGE_BACKEND` from
config and returns the right instance. Call sites depend on the
`StorageBackend` protocol, not the concrete class — swapping backends
later is a config change.

### Content-addressable keys

Keys are SHA-256 hex digests of the body. Three reasons:

1. **Deduplication.** If two filings somehow share an identical body
   (rare but possible with exhibits), we store one copy.
2. **Idempotency.** Re-fetching a filing whose body hasn't changed is
   a no-op: same hash → same key → `put` overwrites the same bytes.
3. **Tamper detection.** Comparing the stored hash against a re-hash of
   the bytes catches storage corruption without hitting EDGAR again.

The local backend shards the tree by the first two characters of the key
(`<root>/<key[:2]>/<key>`) so no single directory accumulates a million
entries when the corpus grows.

### Why not just use Postgres `BYTEA`

I prototyped this. Three problems showed up immediately:

- `pg_dump` and `pg_restore` time grows linearly with body bytes. Even at
  Phase 2 scale (low thousands of filings), backups become annoying.
- `ALTER TABLE` on the bodies table acquires a lock long enough to be
  user-visible.
- The page cache fills up with body bytes that the planner doesn't need
  for query work, evicting pages that would actually help retrieval.

Object storage avoids all three. The cost is a second system to manage —
mitigated by keeping the protocol narrow and the local backend trivial.

### Why not store on the filesystem without an abstraction

A direct `Path("data/filings/<accession>.htm").write_bytes(body)` would
work for dev. It would not work for prod. Keeping the abstraction from
day one means the call sites in the ingestion service don't need to be
rewritten when we move to S3.

## Consequences

- One small migration adds the `filing_documents` table. The bodies table
  pattern is reusable for future blob types (earnings transcripts, news
  articles).
- The retrieval layer (Phase 2 chunker) reads bodies via the same
  `StorageBackend` regardless of where they live.
- A stored hash means we can cheaply detect when a filing was amended
  in place (rare on EDGAR, but it happens) — refetch, recompute hash,
  upsert if it changed.
- Production deployment now has two stateful components to back up:
  Postgres and the object store. Acceptable; the alternative is one
  bigger Postgres that's harder to operate.
