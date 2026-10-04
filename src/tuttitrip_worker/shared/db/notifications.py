"""Write notifications for users (``notifications`` table, backend#133).

The backend owns the table, the list of types and the live stream: a trigger
on INSERT sends ``NOTIFY`` after the commit. The worker only inserts (and, in
the ``notifications`` domain, deletes old rows); it never updates.

Idempotency: ``(user_sub, dedupe_key)`` is unique and the id is derived from
it, so a retried step, a recovered workflow or a repeated job inserts nothing
new. A ``dedupe_key`` therefore names the thing *and* its version or day
(``plan_ready:<trip>:<workflow>``, ``daily:<trip>:<date>:<kind>``).
"""

from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

from dbos import DBOS
from sqlalchemy.dialects.postgresql import Insert, insert

from tuttitrip_worker.contracts import NotificationDraft
from tuttitrip_worker.shared.db.engine import transaction
from tuttitrip_worker.shared.db.tables import notifications


def notification_id(user_sub: str, dedupe_key: str) -> UUID:
    """Deterministic id of the notification of one user and one key.

    Args:
        user_sub: Auth0 subject of the recipient.
        dedupe_key: The idempotency key.

    Returns:
        A UUIDv5.
    """
    return uuid5(NAMESPACE_URL, f"tuttitrip:notification:{user_sub}:{dedupe_key}")


def build_notification_insert(user_sub: str, draft: NotificationDraft) -> Insert:
    """Insert one notification; a repeat of the same key changes nothing.

    Args:
        user_sub: Auth0 subject of the recipient.
        draft: What to write.

    Returns:
        The statement (not executed).
    """
    return (
        insert(notifications)
        .values(
            id=notification_id(user_sub, draft.dedupe_key),
            user_sub=user_sub,
            type=draft.type,
            trip_id=draft.trip_id,
            params=draft.params,
            actions=[{"code": code, "params": {}} for code in draft.actions],
            dedupe_key=draft.dedupe_key,
        )
        .on_conflict_do_nothing(
            index_elements=[notifications.c.user_sub, notifications.c.dedupe_key]
        )
    )


@DBOS.step(retries_allowed=True, max_attempts=3)
async def notify_user(user_sub: str, draft: dict[str, Any]) -> None:
    """Insert a notification for a user (idempotent per ``dedupe_key``).

    Args:
        user_sub: Auth0 subject of the recipient.
        draft: A ``NotificationDraft`` as JSON.
    """
    statement = build_notification_insert(
        user_sub, NotificationDraft.model_validate(draft)
    )
    async with transaction() as connection:
        await connection.execute(statement)
