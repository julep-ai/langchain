from __future__ import annotations

from typing import Any, Dict, Mapping

from langchain.schema.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
)


def convert_message_to_dict(message: BaseMessage) -> dict:
    """Convert a LangChain message to a dictionary.

    Args:
        message: The LangChain message.

    Returns:
        The dictionary.
    """
    message_dict: Dict[str, Any]
    name = getattr(message, "name", None)
    continue_ = getattr(message, "continue_", None)
    if isinstance(message, ChatMessage):
        message_dict = {
            "role": message.role, 
            "name": name,
            "content": message.content,
            "continue": continue_,
        }
    elif isinstance(message, AIMessage):
        message_dict = {
            "role": "assistant", 
            "name": name,
            "content": message.content,
            "continue": continue_,
        }
    elif isinstance(message, HumanMessage):
        message_dict = {
            "role": "user", 
            "name": name,
            "content": message.content,
            "continue": continue_,
        }
    elif isinstance(message, SystemMessage):
        message_dict = {
            "role": "system", 
            "name": name,
            "content": message.content,
            "continue": continue_,
        }
    elif isinstance(message, FunctionMessage):
        message_dict = {
            "role": "function_call", 
            "content": message.content,
            "continue": continue_,
        }
    else:
        raise TypeError(f"Got unknown type {message}")
    if "name" in message.additional_kwargs and not message_dict.get("name"):
        message_dict["name"] = message.additional_kwargs["name"]

    return {k: v for k, v in message_dict.items() if v is not None}


def convert_dict_to_message(_dict: Mapping[str, Any]) -> BaseMessage:
    """Convert a dictionary to a LangChain message.

    Args:
        _dict: The dictionary.

    Returns:
        The LangChain message.
    """
    role = _dict["role"]
    name = _dict.get("name")
    content = _dict.get("content", "") or ""
    if role == "user":
        return HumanMessage(content=content, name=name)
    elif role == "assistant":
        return AIMessage(content=content, name=name)
    elif role == "system":
        return SystemMessage(content=content, name=name)
    elif role == "function_call":
        return FunctionMessage(content=content)
    else:
        return ChatMessage(content=content, role=role, name=name)
