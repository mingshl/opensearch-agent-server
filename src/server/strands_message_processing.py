"""Message Processing Utilities for StrandsAgent.

Handles extraction and processing of user messages from multimodal content.
"""

from __future__ import annotations

import base64
from typing import Any

from utils.logging_helpers import get_logger, log_warning_event
from server.constants import DEFAULT_MIME_TYPE, DEFAULT_USER_MESSAGE, TEXT_MIME_TYPES

logger = get_logger(__name__)


def extract_user_message_from_multimodal_content(content: str | list | Any) -> str:
    """Extract user message text from multimodal content (text + file attachments).

    Handles AG-UI protocol BinaryInputContent by decoding base64 data for text files
    and including file content in the message, similar to Chainlit's file handling.

    Args:
        content: Message content which can be:
                - A string (simple text message)
                - A list of content items (multimodal: text + BinaryInputContent)
                - Other types (converted to string)

    Returns:
        Extracted text message with file content included for text files

    """
    if isinstance(content, str):
        # Simple case: content is already a string
        return content

    # Check if content is a list
    if isinstance(content, list):
        # Initialize empty list to collect text parts
        text_parts = []
        # Iterate over each item in the list
        for item in content:
            # Check if item is a dictionary
            if isinstance(item, dict):
                if item.get("type") == "text":
                    # Text content - extract text field
                    text_parts.append(item.get("text", ""))
                elif item.get("type") == "binary":
                    # File attachment - decode and include content for text files
                    filename = item.get("filename", "uploaded_file")
                    mime_type = item.get("mimeType", DEFAULT_MIME_TYPE)

                    # Try to decode base64 data and include file content
                    file_data = item.get("data")
                    if file_data:
                        try:
                            decoded_data = base64.b64decode(file_data)

                            # For text-based MIME types, decode and include content
                            is_text_file = any(
                                mime_type.startswith(mt) for mt in TEXT_MIME_TYPES
                            )

                            if is_text_file:
                                try:
                                    file_content = decoded_data.decode("utf-8")
                                    text_parts.append(
                                        f"\n\nFile: {filename}\n{file_content}"
                                    )
                                except UnicodeDecodeError:
                                    # If UTF-8 fails, try other encodings or skip content
                                    log_warning_event(
                                        logger,
                                        f"Could not decode file as UTF-8: filename={filename}, mime_type={mime_type}",
                                        "ag_ui.file_utf8_decode_failed",
                                        file_name=filename,
                                        mime_type=mime_type,
                                    )
                                    text_parts.append(
                                        f"\n[File attachment: {filename} ({mime_type}) - binary content]"
                                    )
                            else:
                                # For binary files, just include metadata
                                text_parts.append(
                                    f"\n[File attachment: {filename} ({mime_type}) - binary file]"
                                )
                        except Exception as e:
                            # If decoding fails, include metadata only
                            log_warning_event(
                                logger,
                                f"Error processing file: filename={filename}, mime_type={mime_type}, error={e}",
                                "ag_ui.file_processing_error",
                                exc_info=True,  # Include full exception traceback
                                file_name=filename,
                                mime_type=mime_type,
                                error=str(e),  # Convert exception to string
                            )
                            text_parts.append(
                                f"\n[File attachment: {filename} ({mime_type})]"
                            )
                    else:
                        # No data field, just include metadata
                        text_parts.append(
                            f"\n[File attachment: {filename} ({mime_type})]"
                        )
            elif isinstance(item, str):
                # Item is already a string - append directly
                text_parts.append(item)

        # Join list items with newline separator
        return "\n".join(text_parts) if text_parts else DEFAULT_USER_MESSAGE

    # Fallback for other types
    return str(content)
