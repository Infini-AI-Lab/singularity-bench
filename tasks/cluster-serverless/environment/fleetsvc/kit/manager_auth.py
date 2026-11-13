"""The node manager is a private RPC service for the controller and verifier."""
import hmac

from aiohttp import web


def middleware(token: str):
    @web.middleware
    async def authenticate(request, handler):
        supplied = request.headers.get("X-Grader-Token", "")
        if request.path != "/health" and (
            not token or not hmac.compare_digest(supplied.encode(), token.encode())
        ):
            return web.json_response({"error": "manager authentication required"}, status=403)
        return await handler(request)
    return authenticate
