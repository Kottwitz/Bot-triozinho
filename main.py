import asyncio
import logging
import os
import re
import time
from collections import defaultdict, deque

import discord
from discord.ext import commands
from dotenv import load_dotenv

from ai_chat import GeminiAPIError, GeminiChat
# from oauth_web import OAuthWebService  # [OAuth desativado temporariamente]

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("valorant_community_bot")


def _optional_id(name: str) -> int | None:
    value = os.getenv(name, "").strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError as error:
        raise RuntimeError(f"{name} precisa ser um ID numérico do Discord.") from error


def _role_ids() -> set[int]:
    values = os.getenv("SELF_ASSIGNABLE_ROLE_IDS", "").split(",")
    try:
        return {int(value.strip()) for value in values if value.strip()}
    except ValueError as error:
        raise RuntimeError(
            "SELF_ASSIGNABLE_ROLE_IDS deve conter IDs numéricos separados por vírgula."
        ) from error


RANKS = {
    "ferro": "Ferro",
    "bronze": "Bronze",
    "prata": "Prata",
    "ouro": "Ouro",
    "platina": "Platina",
    "diamante": "Diamante",
    "ascendente": "Ascendente",
    "imortal": "Imortal",
    "radiante": "Radiante",
}
HIGH_RANKS = {"imortal", "radiante"}


def _rank_role_ids() -> dict[str, int]:
    value = os.getenv("RANK_ROLE_IDS", "").strip()
    if not value:
        return {}

    roles: dict[str, int] = {}
    try:
        for item in value.split(","):
            rank_name, role_id = item.split(":", 1)
            rank_name = rank_name.strip().lower()
            if rank_name not in RANKS or rank_name in roles:
                raise ValueError
            roles[rank_name] = int(role_id.strip())
    except ValueError as error:
        raise RuntimeError(
            "RANK_ROLE_IDS deve usar o formato elo:ID,elo:ID com elos válidos e sem duplicatas."
        ) from error

    if set(roles) != set(RANKS):
        raise RuntimeError("Configure um ID em RANK_ROLE_IDS para cada elo.")
    if len(set(roles.values())) != len(roles):
        raise RuntimeError("Cada elo precisa ter um ID de cargo diferente.")
    return roles


WELCOME_CHANNEL_ID = _optional_id("WELCOME_CHANNEL_ID")
SELF_ASSIGNABLE_ROLE_IDS = _role_ids()
RANK_ROLE_IDS = _rank_role_ids()
HIGH_RANK_VERIFIED_ROLE_ID = _optional_id("HIGH_RANK_VERIFIED_ROLE_ID")

# Configuração da IA local via Ollama
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "ollama").strip() or "ollama"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "llama3").strip()

if set(RANK_ROLE_IDS.values()) & SELF_ASSIGNABLE_ROLE_IDS:
    raise RuntimeError("Cargos de elo não podem estar em SELF_ASSIGNABLE_ROLE_IDS.")
if HIGH_RANK_VERIFIED_ROLE_ID is not None and (
    HIGH_RANK_VERIFIED_ROLE_ID in SELF_ASSIGNABLE_ROLE_IDS
    or HIGH_RANK_VERIFIED_ROLE_ID in RANK_ROLE_IDS.values()
):
    raise RuntimeError("O cargo de verificação deve ser exclusivo e não autogerenciável.")

intents = discord.Intents.default()
intents.members = True
intents.message_content = True

ai_chat = GeminiChat(OPENAI_API_KEY, OPENAI_MODEL, base_url=OLLAMA_BASE_URL)
ai_history: dict[tuple[int, int, int], deque[tuple[str, str]]] = defaultdict(
    lambda: deque(maxlen=8)
)
ai_last_reply: dict[int, float] = {}
AI_USER_COOLDOWN_SECONDS = 5


class CommunityBot(commands.Bot):
    async def setup_hook(self) -> None:
        if ai_chat is not None:
            await ai_chat.start()

    async def close(self) -> None:
        if ai_chat is not None:
            await ai_chat.close()
        await super().close()


bot = CommunityBot(command_prefix=".", intents=intents, help_command=None)


@bot.event
async def on_ready() -> None:
    logger.info("Bot conectado como %s", bot.user)
    logger.info(
        "IA configurada via Ollama: modelo=%s, servidor=%s",
        OPENAI_MODEL,
        OLLAMA_BASE_URL,
    )


@bot.listen("on_message")
async def answer_when_mentioned(message: discord.Message) -> None:
    if message.author.bot or ai_chat is None or bot.user is None:
        return

    mentioned = bot.user in message.mentions
    replied_to_bot = False
    reference = message.reference
    if reference is not None and reference.message_id is not None:
        referenced_message = reference.resolved
        if isinstance(referenced_message, discord.Message):
            replied_to_bot = referenced_message.author.id == bot.user.id
        else:
            try:
                referenced_message = await message.channel.fetch_message(reference.message_id)
                replied_to_bot = referenced_message.author.id == bot.user.id
            except (discord.Forbidden, discord.HTTPException):
                logger.debug("Não foi possível ler a mensagem respondida.")

    if not mentioned and not replied_to_bot:
        return

    now = time.monotonic()
    last_reply = ai_last_reply.get(message.author.id, 0)
    remaining_cooldown = AI_USER_COOLDOWN_SECONDS - (now - last_reply)
    if remaining_cooldown > 0:
        await message.reply(
            f"Calma aí! Aguarde mais {remaining_cooldown:.1f}s antes de fazer outra pergunta.",
            delete_after=4,
            mention_author=False,
            allowed_mentions=discord.AllowedMentions.none(),
        )
        return

    prompt = re.sub(rf"<@!?{bot.user.id}>", "", message.content).strip()
    if not prompt:
        prompt = "Cumprimente a pessoa e pergunte no que pode ajudar com VALORANT."

    history_key = (
        message.guild.id if message.guild is not None else 0,
        message.channel.id,
        message.author.id,
    )
    history = ai_history[history_key]
    ai_last_reply[message.author.id] = now

    # Recolhe o nome de exibição e o username técnico para o controlo rigoroso de identidade
    author_name = message.author.display_name
    author_username = message.author.name

    discord_reply = await message.reply(
        "…",
        mention_author=False,
        allowed_mentions=discord.AllowedMentions.none(),
    )
    accumulated = ""
    last_edit = time.monotonic()

    try:
        async for token in ai_chat.generate_reply_stream(
            list(history),
            prompt,
            author_name=author_name,
            author_username=author_username,
        ):
            accumulated = (accumulated + token)[:1900]
            current_time = time.monotonic()
            if current_time - last_edit >= 0.9 and accumulated.strip():
                await discord_reply.edit(content=accumulated)
                last_edit = current_time

        final_reply = accumulated.strip() or "Não consegui gerar uma resposta agora."
        if discord_reply.content != final_reply:
            await discord_reply.edit(content=final_reply)

    except GeminiAPIError as error:
        logger.warning("Falha na IA local (status %s): %s", error.status, error)
        if error.status == 404:
            notice = f"Não encontrei o modelo `{OPENAI_MODEL}` no Ollama. Confirme com `ollama list`."
        else:
            notice = "A IA local não respondeu. Confira se o Ollama está ativo."
        await discord_reply.edit(content=notice)
        return
    except (discord.HTTPException, discord.Forbidden):
        logger.exception("Não foi possível atualizar a resposta no Discord.")
        return

    history.append(("user", prompt[:1200]))
    history.append(("model", final_reply))


@bot.event
async def on_member_join(member: discord.Member) -> None:
    if WELCOME_CHANNEL_ID is None:
        logger.warning("WELCOME_CHANNEL_ID não configurado; boas-vindas ignoradas.")
        return

    channel = bot.get_channel(WELCOME_CHANNEL_ID)
    if not isinstance(channel, discord.TextChannel):
        logger.warning("Canal de boas-vindas %s não encontrado.", WELCOME_CHANNEL_ID)
        return

    await channel.send(
        f"Boas-vindas, {member.mention}! Confira as regras e encontre seu time "
        "para a próxima partida de VALORANT. Para escolher seu elo, use "
        "`.elo escolher <elo>`."
    )


@bot.command(name="ajuda")
async def ajuda(ctx: commands.Context) -> None:
    await ctx.send(
        "**Comandos da comunidade**\n"
        "`.regras` — regras básicas\n"
        "`.elo escolher <elo>` — escolher seu cargo de elo\n"
        "`.elo verificar @membro` / `.elo revogar @membro` — revisão de Imortal/Radiante por moderador\n"
        "`.rank`, `.agentes`, `.mapas` — informações e links oficiais de VALORANT"
    )


@bot.command(name="regras")
async def regras(ctx: commands.Context) -> None:
    await ctx.send(
        "**Regras da comunidade**\n"
        "1. Respeite todos os membros; sem assédio, discurso de ódio ou provocações pessoais.\n"
        "2. Sem cheats, exploits, venda de contas ou incentivo a trapaças.\n"
        "3. Evite spam e use os canais corretos para cada assunto.\n"
    )


@bot.group(name="elo", invoke_without_command=True)
async def elo(ctx: commands.Context) -> None:
    await ctx.send(
        "Use `.elo escolher <elo>`. Elos disponíveis: "
        + ", ".join(RANKS.values())
        + ". Imortal e Radiante exigem validação manual da equipe."
    )


@elo.command(name="escolher")
@commands.guild_only()
async def elo_escolher(ctx: commands.Context, *, rank_name: str) -> None:
    guild = ctx.guild
    member = ctx.author
    rank_key = rank_name.strip().lower()
    if guild is None or not isinstance(member, discord.Member):
        return
    if rank_key not in RANKS:
        await ctx.send("Elo inválido. Escolha: " + ", ".join(RANKS.values()) + ".")
        return
    if not RANK_ROLE_IDS:
        await ctx.send("Os cargos de elo ainda não foram configurados pela equipe.")
        return

    if rank_key in HIGH_RANKS:
        verification_role = (
            guild.get_role(HIGH_RANK_VERIFIED_ROLE_ID)
            if HIGH_RANK_VERIFIED_ROLE_ID is not None
            else None
        )
        if verification_role is None or verification_role not in member.roles:
            await ctx.send(
                "Imortal e Radiante só podem ser escolhidos após validação manual "
                "da equipe. Envie sua prova aos moderadores."
            )
            return

    selected_role = guild.get_role(RANK_ROLE_IDS[rank_key])
    if selected_role is None:
        logger.error("Cargo configurado para o elo %s não existe no servidor.", rank_key)
        await ctx.send("O cargo desse elo não foi encontrado. Avise a equipe.")
        return

    bot_member = guild.me
    if selected_role.managed or bot_member is None or selected_role >= bot_member.top_role:
        await ctx.send("Não consigo gerenciar esse cargo. Avise um administrador.")
        return

    rank_roles = [
        role
        for role_id in RANK_ROLE_IDS.values()
        if (role := guild.get_role(role_id)) is not None and role in member.roles
    ]
    try:
        if rank_roles:
            await member.remove_roles(*rank_roles, reason="Atualização do cargo de elo")
        await member.add_roles(selected_role, reason="Elo escolhido pelo membro")
    except discord.Forbidden:
        logger.exception("Sem permissão para atualizar o elo do membro %s.", member.id)
        await ctx.send("Não tenho permissão para alterar os cargos de elo.")
        return

    await ctx.send(f"{member.mention}, seu cargo de elo agora é **{RANKS[rank_key]}**.")


async def _set_high_rank_verification(
    ctx: commands.Context, member: discord.Member, *, verified: bool
) -> None:
    guild = ctx.guild
    if guild is None:
        return
    moderator = ctx.author
    if not isinstance(moderator, discord.Member):
        return
    if not moderator.guild_permissions.manage_guild:
        await ctx.send("Somente moderadores com Gerenciar Servidor podem usar isso.")
        return
    if HIGH_RANK_VERIFIED_ROLE_ID is None:
        await ctx.send("Configure HIGH_RANK_VERIFIED_ROLE_ID antes de usar este comando.")
        return

    verification_role = guild.get_role(HIGH_RANK_VERIFIED_ROLE_ID)
    if verification_role is None:
        await ctx.send("O cargo de verificação não foi encontrado no servidor.")
        return
    bot_member = guild.me
    if verification_role.managed or bot_member is None or verification_role >= bot_member.top_role:
        await ctx.send("Não consigo gerenciar o cargo de verificação. Avise um administrador.")
        return

    try:
        if verified:
            await member.add_roles(
                verification_role,
                reason=f"Validação manual de elo por {moderator.id}",
            )
            await ctx.send(
                f"{member.mention} foi marcado como validado. A pessoa poderá escolher "
                "Imortal ou Radiante; o comando não escolhe o elo por ela."
            )
        else:
            high_rank_roles = [
                role
                for rank_key in HIGH_RANKS
                if (role := guild.get_role(RANK_ROLE_IDS.get(rank_key, 0))) is not None
                and role in member.roles
            ]
            if high_rank_roles:
                await member.remove_roles(
                    *high_rank_roles,
                    reason=f"Revogação de validação manual por {moderator.id}",
                )
            if verification_role in member.roles:
                await member.remove_roles(
                    verification_role,
                    reason=f"Revogação de validação manual por {moderator.id}",
                )
            await ctx.send(f"A validação de elo de {member.mention} foi revogada.")
    except discord.Forbidden:
        logger.exception("Sem permissão para alterar validação de %s.", member.id)
        await ctx.send("Não tenho permissão para alterar essa validação.")


@elo.command(name="verificar")
@commands.guild_only()
async def elo_verificar(ctx: commands.Context, member: discord.Member) -> None:
    await _set_high_rank_verification(ctx, member, verified=True)


@elo.command(name="revogar")
@commands.guild_only()
async def elo_revogar(ctx: commands.Context, member: discord.Member) -> None:
    await _set_high_rank_verification(ctx, member, verified=False)


@bot.group(name="cargo", invoke_without_command=True)
async def cargo(ctx: commands.Context) -> None:
    await ctx.send("Use `.cargo adicionar @cargo` ou `.cargo remover @cargo`.")


async def _change_self_assignable_role(
    ctx: commands.Context, role: discord.Role, *, add: bool
) -> None:
    guild = ctx.guild
    member = ctx.author
    if guild is None or not isinstance(member, discord.Member):
        return
    if role.id not in SELF_ASSIGNABLE_ROLE_IDS:
        await ctx.send("Esse cargo não está liberado para autogerenciamento.")
        return
    if role.id in RANK_ROLE_IDS.values() or role.id == HIGH_RANK_VERIFIED_ROLE_ID:
        await ctx.send("Cargos de elo/verificação só podem ser alterados pelo fluxo `.elo`.")
        return

    bot_member = guild.me
    if role.managed or bot_member is None or role >= bot_member.top_role:
        await ctx.send("Não consigo gerenciar esse cargo. Peça ajuda a um administrador.")
        return

    try:
        if add:
            await member.add_roles(role, reason="Cargo autogerenciado pelo membro")
            await ctx.send(f"{member.mention}, cargo **{role.name}** adicionado.")
        else:
            await member.remove_roles(role, reason="Cargo autogerenciado pelo membro")
            await ctx.send(f"{member.mention}, cargo **{role.name}** removido.")
    except discord.Forbidden:
        logger.exception("Sem permissão para alterar o cargo %s.", role.id)
        await ctx.send("Não tenho permissão para alterar esse cargo.")


@cargo.command(name="adicionar")
@commands.guild_only()
async def cargo_adicionar(ctx: commands.Context, role: discord.Role) -> None:
    await _change_self_assignable_role(ctx, role, add=True)


@cargo.command(name="remover")
@commands.guild_only()
async def cargo_remover(ctx: commands.Context, role: discord.Role) -> None:
    await _change_self_assignable_role(ctx, role, add=False)


@bot.command(name="rank")
async def rank(ctx: commands.Context) -> None:
    await ctx.send(
        "Os elos competitivos vão de Ferro a Radiante "
        "(Ferro, Bronze, Prata, Ouro, Platina, Diamante, Ascendente, Imortal e Radiante). "
        "Consulte os detalhes atuais no "
        "<https://support-valorant.riotgames.com/hc/pt-br>."
    )


@bot.command(name="agentes")
async def agentes(ctx: commands.Context) -> None:
    await ctx.send("Veja a lista oficial e atualizada de agentes: <https://playvalorant.com/pt-br/agents/>")


@bot.command(name="mapas")
async def mapas(ctx: commands.Context) -> None:
    await ctx.send("Veja os mapas oficiais de VALORANT: <https://playvalorant.com/pt-br/maps/>")


if __name__ == "__main__":
    token = os.getenv("DISCORD_TOKEN")
    if not token:
        raise SystemExit("Configure DISCORD_TOKEN antes de iniciar o bot.")
    bot.run(token)