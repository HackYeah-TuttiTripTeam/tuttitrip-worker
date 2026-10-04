"""Failures a case reports to the runner."""


class InvalidOutputError(Exception):
    """The model answered, but never with a valid structured answer."""


class ProviderError(Exception):
    """No model answered: network, quota or an unreachable endpoint."""


class CaseUnavailableError(Exception):
    """The use case is not in this build of the worker (its PR is not merged)."""
