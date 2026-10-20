# Orbit REST API

The Orbit REST API lets developers automate projects, tasks and workspaces. The current
version is v2, served from `https://api.orbit.example/v2`.

## Authentication

Requests are authenticated with a personal access token or an OAuth 2.0 access token, sent
in the `Authorization: Bearer <token>` header. Personal access tokens expire after 1 year
and can be revoked at any time from Settings > Developer.

## Rate limits

API rate limits depend on the workspace plan:

| Plan       | Requests per minute |
|------------|---------------------|
| Free       | 60                  |
| Team       | 300                 |
| Business   | 1,000               |
| Enterprise | Custom              |

When a client exceeds its rate limit, the API responds with HTTP status 429 and a
`Retry-After` header that says how many seconds to wait before retrying.

## Pagination

List endpoints use cursor-based pagination. The default page size is 50 items and the
maximum page size is 200 items.

## Webhooks

Orbit can send webhooks for events such as `task.created`, `task.updated` and
`project.archived`. Every webhook payload is signed with HMAC-SHA256, and the signature is
sent in the `X-Orbit-Signature` header. Failed deliveries are retried up to 5 times with
exponential backoff over 24 hours.

## SDKs

Lumen Labs maintains official Orbit SDKs for Python, JavaScript and Go.

## Versioning

API v1 was retired on March 31, 2025. All integrations must use v2.
