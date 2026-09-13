"""Tests for on_app_command_error handler in bot.py.

Verifies that all three error branches (CommandOnCooldown, CheckFailure,
generic Exception) correctly dispatch user-visible feedback via
interaction.followup.send() when the interaction has been deferred, and
via interaction.response.send_message() when it has not.

The error handler under test lives in run_bot() as a nested closure.
Rather than bootstrapping the full Discord client, we extract the handler
logic into a standalone async helper and test it with lightweight fakes
that mirror the patterns used in test_who_is_command.py and
test_social_commands.py.
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock

import pytest
from discord import app_commands

# ---------------------------------------------------------------------------
# Fakes -- mirrors _FakeFollowup / _FakeResponse / _FakeInteraction used
# in test_who_is_command.py and test_social_commands.py, extended with
# send_message tracking and is_done() state.
# ---------------------------------------------------------------------------


class _FakeFollowup:
    """Records messages sent via interaction.followup.send()."""

    def __init__(self) -> None:
        self.messages: list[dict] = []

    async def send(self, content: str | None = None, **kwargs: Any) -> None:
        self.messages.append({"content": content, **kwargs})


class _FakeResponse:
    """Tracks whether the interaction was deferred and records send_message calls."""

    def __init__(self, *, deferred: bool = False) -> None:
        self._deferred = deferred
        self.sent_messages: list[dict] = []

    def is_done(self) -> bool:
        return self._deferred

    async def defer(self, *, ephemeral: bool = False) -> None:
        self._deferred = True

    async def send_message(self, content: str, **kwargs: Any) -> None:
        self.sent_messages.append({"content": content, **kwargs})


class _FakeInteraction:
    """Minimal interaction fake with configurable deferred state."""

    def __init__(self, *, deferred: bool = False) -> None:
        self.response = _FakeResponse(deferred=deferred)
        self.followup = _FakeFollowup()
        self.user = MagicMock(name="fakeuser", id=12345)
        self.command = MagicMock(name="fakecommand")
        self.command.name = "test-cmd"


# ---------------------------------------------------------------------------
# Handler under test -- extracted from bot.py on_app_command_error.
# This mirrors the exact logic of the production handler so we can test
# each branch without needing a live Discord client.
# ---------------------------------------------------------------------------


def _stub_format_permission_denied(_config: Any, _cmd_name: str) -> str:
    """Stand-in for format_slash_command_permission_denied."""
    return "Permission denied for this command."


async def _on_app_command_error(
    interaction: _FakeInteraction,
    error: app_commands.AppCommandError,
    *,
    config: Any = None,
    format_denied: Any = _stub_format_permission_denied,
) -> None:
    """Reproduce the on_app_command_error handler logic from bot.py.

    This must stay in sync with the production handler.  Any divergence
    means the tests are no longer validating the real code path.
    """
    logger = logging.getLogger("ghdcbot.bot.test")

    if isinstance(error, app_commands.CommandOnCooldown):
        retry_after = int(error.retry_after) + 1
        message = f"This command is on cooldown. Try again in {retry_after}s."
        try:
            if interaction.response.is_done():
                await interaction.followup.send(message, ephemeral=True)
            else:
                await interaction.response.send_message(message, ephemeral=True)
        except Exception:
            logger.exception("Failed to send cooldown message")
        return

    if isinstance(error, app_commands.CheckFailure):
        try:
            cmd_name = interaction.command.name if interaction.command else "unknown"
            error_message = format_denied(config, cmd_name)
            logger.info(
                "Check failure for user %s (%s) on command %s.",
                interaction.user.name,
                interaction.user.id,
                cmd_name,
            )

            # Use followup if response was already deferred, else send_message
            if interaction.response.is_done():
                await interaction.followup.send(error_message, ephemeral=True)
            else:
                await interaction.response.send_message(error_message, ephemeral=True)
        except Exception as e:
            logger.exception("Failed to send permission denied message", exc_info=e)
            # Try one more time with a simple message
            try:
                if interaction.response.is_done():
                    await interaction.followup.send(
                        "You do not have permission to use this command.",
                        ephemeral=True,
                    )
                else:
                    await interaction.response.send_message(
                        "You do not have permission to use this command.",
                        ephemeral=True,
                    )
            except Exception:  # noqa: BLE001
                logger.error("Could not send any error message to user")
        return

    # General / unknown error fallback
    logger.exception("App command error", exc_info=error)
    try:
        if interaction.response.is_done():
            await interaction.followup.send(
                "An unexpected error occurred. Please try again later.",
                ephemeral=True,
            )
        else:
            await interaction.response.send_message(
                "An unexpected error occurred. Please try again later.",
                ephemeral=True,
            )
    except Exception:  # noqa: BLE001
        logger.error("Could not send error message to user")


# ---------------------------------------------------------------------------
# Helper to create discord.py error objects
# ---------------------------------------------------------------------------


def _cooldown_error(retry_after: float = 9.0) -> app_commands.CommandOnCooldown:
    """Create a CommandOnCooldown error with the given retry_after value."""
    cooldown = MagicMock()
    cooldown.per = 30.0
    cooldown.rate = 1
    return app_commands.CommandOnCooldown(cooldown, retry_after)


def _check_failure() -> app_commands.CheckFailure:
    """Create a generic CheckFailure error."""
    return app_commands.CheckFailure("Permission check failed")


def _generic_error() -> app_commands.AppCommandError:
    """Create a generic AppCommandError (not cooldown, not check failure)."""
    return app_commands.AppCommandError("Something went wrong internally")


# ===================================================================
# Tests for the GENERAL / GENERIC error branch (the main bug fix)
# ===================================================================


class TestGenericErrorHandler:
    """Tests for the else branch -- the primary bug that was fixed."""

    @pytest.mark.asyncio
    async def test_deferred_sends_followup(self) -> None:
        """When the interaction was deferred, the error message must go
        through followup.send() instead of response.send_message()."""
        interaction = _FakeInteraction(deferred=True)

        await _on_app_command_error(interaction, _generic_error())

        # followup.send must have been called exactly once
        assert len(interaction.followup.messages) == 1
        # response.send_message must NOT have been called
        assert len(interaction.response.sent_messages) == 0

    @pytest.mark.asyncio
    async def test_not_deferred_sends_response(self) -> None:
        """When the interaction was NOT deferred, the error message must go
        through response.send_message()."""
        interaction = _FakeInteraction(deferred=False)

        await _on_app_command_error(interaction, _generic_error())

        # response.send_message must have been called exactly once
        assert len(interaction.response.sent_messages) == 1
        # followup.send must NOT have been called
        assert len(interaction.followup.messages) == 0

    @pytest.mark.asyncio
    async def test_message_is_ephemeral(self) -> None:
        """The error message must always be ephemeral so only the
        invoking user sees it."""
        # Test deferred path
        interaction = _FakeInteraction(deferred=True)
        await _on_app_command_error(interaction, _generic_error())
        assert interaction.followup.messages[0]["ephemeral"] is True

        # Test non-deferred path
        interaction = _FakeInteraction(deferred=False)
        await _on_app_command_error(interaction, _generic_error())
        assert interaction.response.sent_messages[0]["ephemeral"] is True

    @pytest.mark.asyncio
    async def test_message_content(self) -> None:
        """The error message must contain a user-friendly error string."""
        interaction = _FakeInteraction(deferred=True)
        await _on_app_command_error(interaction, _generic_error())
        content = interaction.followup.messages[0]["content"]
        assert "unexpected error" in content.lower()

    @pytest.mark.asyncio
    async def test_followup_failure_does_not_crash(self, caplog: pytest.LogCaptureFixture) -> None:
        """If followup.send() itself raises, the handler must log the
        error and not propagate the exception."""
        interaction = _FakeInteraction(deferred=True)

        # Make followup.send raise
        async def _raise(**kwargs: Any) -> None:
            raise RuntimeError("Discord API unavailable")

        interaction.followup.send = _raise  # type: ignore[assignment]

        # Must not raise
        await _on_app_command_error(interaction, _generic_error())

        # Verify the error was logged
        assert any(
            "Could not send error message" in rec.message
            for rec in caplog.records
        )


# ===================================================================
# Tests for the COOLDOWN error branch
# ===================================================================


class TestCooldownErrorHandler:
    """Tests for the CommandOnCooldown branch."""

    @pytest.mark.asyncio
    async def test_deferred_sends_followup(self) -> None:
        """Cooldown message must use followup.send() when deferred."""
        interaction = _FakeInteraction(deferred=True)

        await _on_app_command_error(interaction, _cooldown_error(9.0))

        assert len(interaction.followup.messages) == 1
        assert len(interaction.response.sent_messages) == 0

    @pytest.mark.asyncio
    async def test_not_deferred_sends_response(self) -> None:
        """Cooldown message must use response.send_message() when not deferred."""
        interaction = _FakeInteraction(deferred=False)

        await _on_app_command_error(interaction, _cooldown_error(9.0))

        assert len(interaction.response.sent_messages) == 1
        assert len(interaction.followup.messages) == 0

    @pytest.mark.asyncio
    async def test_message_contains_retry_time(self) -> None:
        """The cooldown message must include the retry-after seconds
        (rounded up by 1)."""
        interaction = _FakeInteraction(deferred=True)

        await _on_app_command_error(interaction, _cooldown_error(9.0))

        content = interaction.followup.messages[0]["content"]
        # retry_after = int(9.0) + 1 = 10
        assert "10s" in content

    @pytest.mark.asyncio
    async def test_cooldown_is_ephemeral(self) -> None:
        """Cooldown messages must always be ephemeral."""
        interaction = _FakeInteraction(deferred=True)
        await _on_app_command_error(interaction, _cooldown_error(5.0))
        assert interaction.followup.messages[0]["ephemeral"] is True


# ===================================================================
# Tests for the CHECK FAILURE error branch
# ===================================================================


class TestCheckFailureErrorHandler:
    """Tests for the CheckFailure branch, including its inner retry."""

    @pytest.mark.asyncio
    async def test_deferred_sends_followup(self) -> None:
        """Permission denied message must use followup.send() when deferred."""
        interaction = _FakeInteraction(deferred=True)

        await _on_app_command_error(interaction, _check_failure())

        assert len(interaction.followup.messages) == 1
        assert len(interaction.response.sent_messages) == 0

    @pytest.mark.asyncio
    async def test_not_deferred_sends_response(self) -> None:
        """Permission denied message must use response.send_message()
        when not deferred."""
        interaction = _FakeInteraction(deferred=False)

        await _on_app_command_error(interaction, _check_failure())

        assert len(interaction.response.sent_messages) == 1
        assert len(interaction.followup.messages) == 0

    @pytest.mark.asyncio
    async def test_check_failure_is_ephemeral(self) -> None:
        """Permission denied messages must always be ephemeral."""
        interaction = _FakeInteraction(deferred=True)
        await _on_app_command_error(interaction, _check_failure())
        assert interaction.followup.messages[0]["ephemeral"] is True

    @pytest.mark.asyncio
    async def test_retry_deferred_sends_followup(self) -> None:
        """If the primary CheckFailure handler throws AND the interaction
        was deferred, the retry must use followup.send()."""
        interaction = _FakeInteraction(deferred=True)

        # Make format_denied raise so we fall into the retry branch
        def _bad_format(_config: Any, _cmd: str) -> str:
            raise RuntimeError("format_denied exploded")

        await _on_app_command_error(
            interaction, _check_failure(), format_denied=_bad_format
        )

        # The retry should have sent a simple fallback message via followup
        assert len(interaction.followup.messages) == 1
        content = interaction.followup.messages[0]["content"]
        assert "permission" in content.lower()

    @pytest.mark.asyncio
    async def test_retry_not_deferred_sends_response(self) -> None:
        """If the primary CheckFailure handler throws AND the interaction
        was NOT deferred, the retry must use response.send_message()."""
        interaction = _FakeInteraction(deferred=False)

        def _bad_format(_config: Any, _cmd: str) -> str:
            raise RuntimeError("format_denied exploded")

        await _on_app_command_error(
            interaction, _check_failure(), format_denied=_bad_format
        )

        # The retry should have sent via response.send_message
        assert len(interaction.response.sent_messages) == 1
        content = interaction.response.sent_messages[0]["content"]
        assert "permission" in content.lower()

    @pytest.mark.asyncio
    async def test_uses_command_name(self) -> None:
        """The handler must read interaction.command.name for logging."""
        interaction = _FakeInteraction(deferred=False)
        interaction.command.name = "sync"

        called_with: list[str] = []

        def _tracking_format(_config: Any, cmd_name: str) -> str:
            called_with.append(cmd_name)
            return f"Denied: {cmd_name}"

        await _on_app_command_error(
            interaction, _check_failure(), format_denied=_tracking_format
        )

        assert called_with == ["sync"]
