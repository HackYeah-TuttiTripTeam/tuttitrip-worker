"""Helpers for reading the answers of decision models (basal, Laya, JEV)."""

from pydantic_ai import ModelResponse
from pydantic_ai.exceptions import (
    FallbackExceptionGroup,
    ModelAPIError,
    UnexpectedModelBehavior,
)


def model_unavailable(error: BaseException) -> bool:
    """Whether a model failure means "no model could answer".

    A ``FallbackModel`` raises a group when every model failed; the group
    counts only if each member is a provider or response failure. Anything else
    (a misconfiguration such as ``UserError``) is a bug and must fail the job.

    Args:
        error: Exception raised by an agent run.

    Returns:
        ``True`` for provider errors and unreadable model answers.
    """
    if isinstance(error, FallbackExceptionGroup):
        return all(model_unavailable(inner) for inner in error.exceptions)
    return isinstance(error, ModelAPIError | UnexpectedModelBehavior)


def reported_confidences(response: ModelResponse | None) -> dict[str, float]:
    """Read the confidence a decision model reported per answered field.

    ``ModelResponse.provider_details["confidence"]`` is a dict keyed by output
    field name (see ``pydantic_ai.models.decision``). For a pick-one it is the
    model's margin, scaled to 0..1; it is not a probability. A language-model
    fallback reports nothing.

    Args:
        response: Last model response of the run.

    Returns:
        Confidence per field name, clamped to 0..1; fields without one are absent.
    """
    details = response.provider_details if response else None
    reported = details.get("confidence") if details else None
    if not isinstance(reported, dict):
        return {}
    return {
        str(name): min(1.0, max(0.0, float(value)))
        for name, value in reported.items()
        if isinstance(value, int | float) and not isinstance(value, bool)
    }
