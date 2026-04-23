-- Ensure the pgvector extension is available when the container starts up.
-- Alembic's baseline migration creates the extension inside a transaction;
-- this file only validates that the shared library ships with the image.
CREATE EXTENSION IF NOT EXISTS vector;
