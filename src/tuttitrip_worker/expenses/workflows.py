"""Expenses workflows: a typed expense and a receipt, read by models."""

from datetime import date
from typing import Any

from dbos import DBOS

from tuttitrip_worker.contracts import (
    ErrorCode,
    ParseExpenseTextInput,
    ParseExpenseTextOutput,
    ReadReceiptInput,
    Workflow,
    contract_failure,
    document_not_found,
    parse_input,
)
from tuttitrip_worker.expenses import steps
from tuttitrip_worker.expenses.agents import judge_reading, read_typed_expense
from tuttitrip_worker.expenses.logic.amounts import (
    currency_in_text,
    names_in_text,
    require_amount,
)
from tuttitrip_worker.expenses.logic.receipt import build_output
from tuttitrip_worker.expenses.schemas import TripDates
from tuttitrip_worker.shared.db import job_results
from tuttitrip_worker.shared.dbos.runtime import PORTABLE, report_progress

DESCRIPTION_MAX = 500
NAME_MAX = 100
NAMES_MAX = 50


async def _keep(workflow: Workflow, output: dict[str, Any]) -> None:
    workflow_id = DBOS.workflow_id
    if workflow_id is not None:
        await job_results.save_job_result(workflow_id, workflow.value, output)


@DBOS.workflow(name=Workflow.PARSE_EXPENSE_TEXT.value, serialization_type=PORTABLE)
async def parse_expense_text(payload: dict[str, Any]) -> dict[str, Any]:
    """Read one typed sentence into the fields of an expense.

    The amount must be written in the text and match the model's number;
    otherwise the job fails, because an amount is never guessed. Names stay as
    written (a name or a currency that is not in the text is dropped); the
    backend matches names to profiles. ``confidence`` stays ``None``: the chat
    model reports none, and the check against the text is binary.

    Args:
        payload: JSON object matching ``ParseExpenseTextInput``.

    Returns:
        JSON object matching ``ParseExpenseTextOutput``.

    Raises:
        ContractError: The text holds no usable amount (``model_output_invalid``
            when the model could not produce one, ``invalid_payload`` shape errors).
    """
    request = parse_input(ParseExpenseTextInput, payload)
    await report_progress("reading", 20)
    reading = await read_typed_expense(
        request.text, str(request.trip_id), request.locale
    )
    text = request.text
    amount = require_amount(text, reading.amount_minor, reading.amount_text)
    if amount is None:
        raise contract_failure(
            ErrorCode.MODEL_OUTPUT_INVALID, "the text holds no clear amount"
        )
    payer = names_in_text(text, [reading.payer_name or ""])
    output = ParseExpenseTextOutput(
        amount_minor=amount,
        # Only what the text says: a currency or a name the model made up is
        # dropped (the trip currency applies; the backend asks about the rest).
        currency=currency_in_text(text, reading.currency),
        description=reading.description.strip()[:DESCRIPTION_MAX],
        payer_name=payer[0][:NAME_MAX] if payer else None,
        included_names=names_in_text(text, reading.included_names)[:NAMES_MAX],
        excluded_names=names_in_text(text, reading.excluded_names)[:NAMES_MAX],
    ).model_dump(mode="json")
    await _keep(Workflow.PARSE_EXPENSE_TEXT, output)
    await report_progress("done", 100)
    return output


@DBOS.workflow(name=Workflow.READ_RECEIPT.value, serialization_type=PORTABLE)
async def read_receipt(payload: dict[str, Any]) -> dict[str, Any]:
    """Read a receipt or a bank screenshot into settlement fields.

    The vision model reads the image inside one step (only the reading is
    stored), the decision model proposes the category, and rules S1 in pure
    code decide whether the host must confirm. No fallback to a cloud model
    for the image: when the GB10 is down the job fails and the user types the
    expense. The result goes to ``job_results`` without the image.

    Args:
        payload: JSON object matching ``ReadReceiptInput``.

    Returns:
        JSON object matching ``ReadReceiptOutput``.

    Raises:
        ContractError: No such image for the trip (``document_not_found``) or
            no amount could be read (``model_output_invalid``).
    """
    request = parse_input(ReadReceiptInput, payload)
    await report_progress("reading", 10)
    raw = await steps.read_evidence(str(request.evidence_id), str(request.trip_id))
    if raw is None:
        raise document_not_found(request.evidence_id, "receipt image")
    reading = steps.to_reading(raw)
    await report_progress("judging", 60)
    dates = await steps.load_trip_dates(str(request.trip_id))
    trip = TripDates(
        start=date.fromisoformat(dates["start"]) if dates["start"] else None,
        end=date.fromisoformat(dates["end"]) if dates["end"] else None,
    )
    judgement = await judge_reading(reading, str(request.evidence_id))
    result = build_output(reading, trip, judgement)
    if result is None:
        raise contract_failure(
            ErrorCode.MODEL_OUTPUT_INVALID, "no amount could be read from the image"
        )
    output = result.model_dump(mode="json")
    await _keep(Workflow.READ_RECEIPT, output)
    await report_progress("done", 100)
    return output
