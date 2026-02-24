"""Text Event Handler for StrandsAgent.

Handles text streaming events from the Strands agent and converts them to AG-UI text events.
"""

from __future__ import annotations

from ag_ui.core import (
    EventType,
    TextMessageContentEvent,
    TextMessageStartEvent,
)

from server.types import StrandsEvent


class TextEventHandler:
    """Handler for text streaming events from the Strands agent.

    Converts Strands text streaming events into AG-UI text message events.

    Tracks accumulated text per message to calculate deltas if the Strands SDK
    sends accumulated text instead of incremental chunks.
    """

    def __init__(self) -> None:
        """Initialize text event handler with message tracking."""
        # Track accumulated text per message_id to calculate deltas
        # Maps message_id -> accumulated_text
        self._accumulated_text: dict[str, str] = {}

    def handle_text_streaming(
        self, event: StrandsEvent, message_id: str, message_started: bool
    ) -> tuple[bool, TextMessageStartEvent | None, TextMessageContentEvent | None]:
        """Handle text streaming events from the agent.

        Args:
            event: Strands event dictionary
            message_id: Current message ID
            message_started: Whether message has been started

        Returns:
            Tuple of (updated message_started flag, start_event or None, content_event or None)

        """
        start_event = None
        content_event = None

        # Check if event contains text data
        if "data" in event and event["data"]:
            # Check if we have accumulated text for this message_id
            # If accumulated text exists, message has already started (even if flag is wrong)
            has_accumulated_text = (
                message_id in self._accumulated_text
                and self._accumulated_text[message_id]
            )

            if not message_started and not has_accumulated_text:
                # Truly starting a new message - initialize accumulated text
                self._accumulated_text[message_id] = ""
                start_event = TextMessageStartEvent(
                    type=EventType.TEXT_MESSAGE_START,
                    message_id=message_id,
                    role="assistant",
                )
                message_started = True
            elif not message_started and has_accumulated_text:
                # message_started flag is False but we have accumulated text
                # This means message_started was incorrectly reset - don't send START again
                # Don't send START event, but mark as started
                message_started = True
            elif message_started:
                # Message already started - ensure accumulated text exists
                if message_id not in self._accumulated_text:
                    self._accumulated_text[message_id] = ""

            # Get the text from the event
            event_text = str(event["data"])

            # Get previously accumulated text for this message
            accumulated = self._accumulated_text.get(message_id, "")

            # Determine if event_text is accumulated text or a delta:
            # - If we have accumulated text AND event_text starts with it and is longer, it's accumulated
            # - Otherwise, treat it as an incremental delta chunk (normal case for Strands SDK)
            if (
                accumulated
                and event_text.startswith(accumulated)
                and len(event_text) > len(accumulated)
            ):
                # Event contains accumulated text - calculate delta
                delta = event_text[len(accumulated) :]
                self._accumulated_text[message_id] = event_text
            else:
                # Event is an incremental delta chunk (normal case for Strands SDK)
                # This handles both the first chunk (accumulated is empty) and subsequent chunks
                delta = event_text
                self._accumulated_text[message_id] = accumulated + delta

            # Only emit content event if delta is non-empty
            if delta:
                content_event = TextMessageContentEvent(
                    type=EventType.TEXT_MESSAGE_CONTENT,
                    message_id=message_id,
                    delta=delta,
                )

        return message_started, start_event, content_event
