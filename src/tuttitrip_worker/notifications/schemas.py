"""Internal payloads of the notifications domain (pure)."""

from pydantic import BaseModel, ConfigDict, Field


class PurgeResult(BaseModel):
    """What one daily purge deleted."""

    model_config = ConfigDict(frozen=True)

    deleted: int = Field(ge=0)
    batches: int = Field(ge=0)
