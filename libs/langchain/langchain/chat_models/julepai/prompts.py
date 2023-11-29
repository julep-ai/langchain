from __future__ import annotations

from typing import Any

from langchain.prompts.chat import BaseStringMessagePromptTemplate
from langchain.schema.messages import BaseMessage, FunctionMessage, SystemMessage


class FunctionCallMessagePromptTemplate(BaseStringMessagePromptTemplate):
    """Function call message prompt template."""

    def format(self, **kwargs: Any) -> BaseMessage:
        """Format the prompt template.

        Args:
            **kwargs: Keyword arguments to use for formatting.

        Returns:
            Formatted message.
        """
        text = self.prompt.format(**kwargs)
        return FunctionMessage(
            content=text,
            role="function_call",
            additional_kwargs=self.additional_kwargs,
        )


class FunctionsMessagePromptTemplate(BaseStringMessagePromptTemplate):
    """Functions message prompt template."""

    def format(self, **kwargs: Any) -> BaseMessage:
        """Format the prompt template.

        Args:
            **kwargs: Keyword arguments to use for formatting.

        Returns:
            Formatted message.
        """
        text = self.prompt.format(**kwargs)
        return SystemMessage(
            content=text,
            role="system",
            name="functions",
            additional_kwargs=self.additional_kwargs,
        )


class InformationMessagePromptTemplate(BaseStringMessagePromptTemplate):
    """Information message prompt template. This is a message sent as an information."""

    def format(self, **kwargs: Any) -> BaseMessage:
        """Format the prompt template.

        Args:
            **kwargs: Keyword arguments to use for formatting.

        Returns:
            Formatted message.
        """
        text = self.prompt.format(**kwargs)
        return SystemMessage(
            role="system",
            name="informarion",
            content=text,
            additional_kwargs=self.additional_kwargs,
        )


class SituationMessagePromptTemplate(BaseStringMessagePromptTemplate):
    """Situation message prompt template. This is a message sent as a situation text."""

    def format(self, **kwargs: Any) -> BaseMessage:
        """Format the prompt template.

        Args:
            **kwargs: Keyword arguments to use for formatting.

        Returns:
            Formatted message.
        """
        text = self.prompt.format(**kwargs)
        return SystemMessage(
            role="system",
            name="situation",
            content=text,
            additional_kwargs=self.additional_kwargs,
        )


class ThoughtMessagePromptTemplate(BaseStringMessagePromptTemplate):
    """Thought message prompt template. This is a message sent as a situation text."""

    def format(self, **kwargs: Any) -> BaseMessage:
        """Format the prompt template.

        Args:
            **kwargs: Keyword arguments to use for formatting.

        Returns:
            Formatted message.
        """
        text = self.prompt.format(**kwargs)
        return SystemMessage(
            role="system",
            name="thought",
            content=text,
            additional_kwargs=self.additional_kwargs,
        )
