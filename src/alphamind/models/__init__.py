"""ORM models for AlphaMind.

Importing this package eagerly imports every model module so that each
``Base`` subclass is registered on ``Base.metadata``. Alembic's environment
imports this package to discover the full schema for autogenerate.
"""

from alphamind.models.company import Company
from alphamind.models.filing import Filing
from alphamind.models.filing_chunk import EMBEDDING_DIM, FilingChunk
from alphamind.models.filing_document import FilingDocument

__all__ = [
    "EMBEDDING_DIM",
    "Company",
    "Filing",
    "FilingChunk",
    "FilingDocument",
]
