"""Internal models of the demo domain."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class DemoResetResult(BaseModel):
    """The backend's answer to the internal reset request.

    ``disabled`` means the demo is switched off there (empty
    ``TUTTITRIP_DEMO__TOKEN_SHA256``) and nothing was touched.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    status: Literal["reset", "disabled"]
    trips: int = Field(default=0, ge=0)
