import secrets
import time
from urllib.parse import urlencode

from aiohttp import ClientSession, ClientTimeout, web

DISCORD_API = "https://discord.com/api/v10"
OAUTH_SCOPES = "identify guilds.join"
STATE_TTL_SECONDS = 600
MAX_PENDING_STATES = 5000


class OAuthWebService:
    def __init__(
        self,
        *,
        bot_token: str,
        client_id: str,
        client_secret: str,
        redirect_uri: str,
        public_url: str,
        target_guild_id: int,
    ) -> None:
        self._bot_token = bot_token
        self._client_id = client_id
        self._client_secret = client_secret
        self._redirect_uri = redirect_uri
        self._public_url = public_url.rstrip("/")
        self._target_guild_id = target_guild_id
        self._pending: dict[str, tuple[int, float]] = {}
        self._runner: web.AppRunner | None = None

    def create_consent_url(self, discord_user_id: int) -> str:
        self._remove_expired_states()
        if len(self._pending) >= MAX_PENDING_STATES:
            raise RuntimeError("Muitas autorizações pendentes. Tente novamente mais tarde.")
        state = secrets.token_urlsafe(32)
        self._pending[state] = (discord_user_id, time.monotonic())
        return f"{self._public_url}/consent/{state}"

    async def start(self, host: str, port: int) -> None:
        app = web.Application(client_max_size=16 * 1024)
        app.add_routes(
            [
                web.get("/consent/{state}", self._consent),
                web.post("/oauth/start/{state}", self._oauth_start),
                web.get("/oauth/callback", self._callback),
            ]
        )
        self._runner = web.AppRunner(app, access_log=None)
        await self._runner.setup()
        site = web.TCPSite(self._runner, host, port)
        await site.start()

    async def close(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None

    async def _consent(self, request: web.Request) -> web.Response:
        state = request.match_info["state"]
        self._remove_expired_states()
        if state not in self._pending:
            return self._page(
                "Link expirado",
                "<p>Este link expirou ou já foi usado. Peça um novo link ao bot.</p>",
                status=400,
            )

        content = (
            "<p>Ao continuar, você será enviado ao Discord para autorizar este aplicativo.</p>"
            "<ul>"
            "<li><strong>Identificar você no Discord</strong> "
            "<p>O aplicativo não solicita nem armazena seu IP, e-mail ou senha. "
            "O token OAuth é usado apenas durante a solicitação de entrada e não é "
            "persistido. O link é individual e só funciona para a conta Discord que "
            "solicitou a autorização.</p>"
            f'<form method="post" action="/oauth/start/{state}">'
            '<button type="submit">Continuar para autorização do Discord</button>'
            "</form>"
            "<p>Se não quiser participar, feche esta página.</p>"
        )
        return self._page("Autorização para entrar no servidor", content)

    async def _oauth_start(self, request: web.Request) -> web.Response:
        state = request.match_info["state"]
        self._remove_expired_states()
        if state not in self._pending:
            return self._page(
                "Link expirado",
                "<p>Este link expirou ou já foi usado. Peça um novo link ao bot.</p>",
                status=400,
            )

        query = urlencode(
            {
                "client_id": self._client_id,
                "response_type": "code",
                "redirect_uri": self._redirect_uri,
                "scope": OAUTH_SCOPES,
                "state": state,
            }
        )
        raise web.HTTPSeeOther(f"https://discord.com/oauth2/authorize?{query}")

    async def _callback(self, request: web.Request) -> web.Response:
        state = request.query.get("state", "")
        self._remove_expired_states()
        pending = self._pending.pop(state, None)
        if pending is None:
            return self._page(
                "Autorização inválida",
                "<p>O link expirou, já foi usado ou não corresponde a uma autorização pendente.</p>",
                status=400,
            )

        if request.query.get("error"):
            return self._page(
                "Autorização cancelada",
                "<p>Você cancelou a autorização no Discord. Nenhuma entrada foi solicitada.</p>",
                status=400,
            )

        code = request.query.get("code")
        if not code:
            return self._page(
                "Resposta inválida",
                "<p>O Discord não retornou um código de autorização válido.</p>",
                status=400,
            )

        expected_user_id = pending[0]
        timeout = ClientTimeout(total=15)
        async with ClientSession(timeout=timeout) as session:
            token_response = await session.post(
                f"{DISCORD_API}/oauth2/token",
                data={
                    "client_id": self._client_id,
                    "client_secret": self._client_secret,
                    "grant_type": "authorization_code",
                    "code": code,
                    "redirect_uri": self._redirect_uri,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            if token_response.status != 200:
                return self._page(
                    "Não foi possível concluir",
                    "<p>O Discord recusou a autorização. Solicite um novo link ao bot.</p>",
                    status=502,
                )
            token_data = await token_response.json()
            access_token = token_data.get("access_token")
            if not isinstance(access_token, str) or not access_token:
                return self._page(
                    "Resposta inválida",
                    "<p>O Discord não retornou uma autorização utilizável.</p>",
                    status=502,
                )

            user_response = await session.get(
                f"{DISCORD_API}/users/@me",
                headers={"Authorization": f"Bearer {access_token}"},
            )
            if user_response.status != 200:
                return self._page(
                    "Não foi possível verificar a conta",
                    "<p>Não foi possível confirmar a conta Discord autorizada.</p>",
                    status=502,
                )
            user_data = await user_response.json()
            if str(user_data.get("id", "")) != str(expected_user_id):
                return self._page(
                    "Conta diferente",
                    "<p>Esta autorização foi solicitada por outra conta Discord. "
                    "Nenhuma entrada foi solicitada. Peça seu próprio link ao bot.</p>",
                    status=403,
                )

            join_response = await session.put(
                f"{DISCORD_API}/guilds/{self._target_guild_id}/members/{expected_user_id}",
                json={"access_token": access_token},
                headers={"Authorization": f"Bot {self._bot_token}"},
            )
            if join_response.status not in (201, 204):
                return self._page(
                    "Não foi possível adicionar à comunidade",
                    "<p>A autorização foi confirmada, mas o Discord não permitiu a "
                    "entrada no servidor. Confirme com a equipe se o bot está no "
                    "servidor de destino e configurado corretamente.</p>",
                    status=502,
                )

        return self._page(
            "Concluído",
            "<p>Sua conta Discord foi autorizada e a solicitação de entrada no "
            "servidor foi concluída. Você já pode fechar esta página.</p>",
        )

    def _remove_expired_states(self) -> None:
        now = time.monotonic()
        expired = [
            state
            for state, pending in self._pending.items()
            if now - pending[1] > STATE_TTL_SECONDS
        ]
        for state in expired:
            del self._pending[state]

    @staticmethod
    def _page(title: str, content: str, *, status: int = 200) -> web.Response:
        body = (
            "<!doctype html><html lang=\"pt-BR\"><meta charset=\"utf-8\">"
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>{title}</title><h1>{title}</h1>{content}</html>"
        )
        return web.Response(
            text=body,
            status=status,
            content_type="text/html",
            headers={
                "Cache-Control": "no-store",
                "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; "
                "form-action 'self'; base-uri 'none'; frame-ancestors 'none'",
                "Referrer-Policy": "no-referrer",
                "X-Content-Type-Options": "nosniff",
            },
        )
