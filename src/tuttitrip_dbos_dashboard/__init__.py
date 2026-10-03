"""Read-only DBOS dashboard for tuttitrip-worker (workflows, queues, steps).

Served at https://tuttitrip-dbos.gburek.app behind tuttitrip-gateway and
oauth2-proxy (Auth0 + superadmin allow-list); it never writes to the database.
Not a worker domain: it runs no workflows and is started as its own container
(``tuttitrip-dbos-dashboard``) from the ``tuttitrip-worker:main`` image.
"""
