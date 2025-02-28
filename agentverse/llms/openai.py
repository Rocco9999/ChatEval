import logging
import numpy as np
import time
import os
import random
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field
import requests

from agentverse.llms.base import LLMResult

from . import llm_registry
from .base import BaseChatModel, BaseCompletionModel, BaseModelArgs
from agentverse.message import Message
logging.getLogger("httpx").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

# Lista dei token (SOSTITUISCI CON I TUOI TOKEN REALI)
tokens = [
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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"
]

current_token_index = 0  # Indice del token corrente (globale)


def set_next_token():
    """Imposta il prossimo token come chiave API."""
    global current_token_index  # Usa la variabile globale
    current_token_index = (current_token_index + 1) % len(tokens)
    openai.api_key = tokens[current_token_index]
    logger.info(f"Switched to token: {openai.api_key}")

try:
    import openai
    from openai import OpenAI, AsyncOpenAI
    
    
    openai.api_key = tokens[current_token_index]
    print(openai.api_key)
    
    #print(openai.api_key)
    client = OpenAI(api_key=openai.api_key)  # Crea il client SENZA specificare la chiave
    #print(client.models.list())
    aclient = AsyncOpenAI(api_key=openai.api_key)  # Crea il client asincrono SENZA
    from openai import OpenAIError
except ImportError:
    is_openai_available = False
    logging.warning("openai package is not installed")
else:

    if openai.api_key is None:
        logging.warning(
            "OpenAI API key is not set. Please set the environment variable OPENAI_API_KEY"
        )
        is_openai_available = False
    else:
        is_openai_available = True


class OpenAIChatArgs(BaseModelArgs):
    model: str = Field(default="gpt-4")
    max_tokens: int = Field(default=2048)
    temperature: float = Field(default=1.0)
    top_p: int = Field(default=1)
    n: int = Field(default=1)
    stop: Optional[Union[str, List]] = Field(default=None)
    presence_penalty: int = Field(default=0)
    frequency_penalty: int = Field(default=0)


class OpenAICompletionArgs(OpenAIChatArgs):
    model: str = Field(default="gpt-4")
    suffix: str = Field(default="")
    best_of: int = Field(default=1)


@llm_registry.register("text-davinci-003")
class OpenAICompletion(BaseCompletionModel):
    args: OpenAICompletionArgs = Field(default_factory=OpenAICompletionArgs)

    def __init__(self, max_retry: int = 3, **kwargs):
        args = OpenAICompletionArgs()
        args = args.dict()
        for k, v in args.items():
            args[k] = kwargs.pop(k, v)
        if len(kwargs) > 0:
            logging.warning(f"Unused arguments: {kwargs}")
        super().__init__(args=args, max_retry=max_retry)

    def generate_response(self, prompt: str, chat_memory: List[Message], final_prompt: str) -> LLMResult:
        try:
            response = client.completions.create(prompt=prompt, **self.args.dict())
            return LLMResult(
                content=response.choices[0].text,
                send_tokens=response.usage.prompt_tokens,
                recv_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )
        except requests.exceptions.HTTPError as e:  # Cattura l'eccezione HTTP
            if e.response.status_code == 429:
                logger.error(f"Rate limit exceeded. Switching token...")
                set_next_token()  # Cambia token
            raise  # Rilancia l'eccezione per farla gestire dallo script principale

    async def agenerate_response(self, prompt: str, chat_memory: List[Message], final_prompt: str) -> LLMResult:
        try:
            response = await aclient.completions.create(prompt=prompt, **self.args.dict())
            return LLMResult(
                content=response.choices[0].text,
                send_tokens=response.usage.prompt_tokens,
                recv_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                logger.error(f"Rate limit exceeded. Switching token...")
                set_next_token()
            raise

@llm_registry.register("llama3-70b-8192")
@llm_registry.register("gpt-3.5-turbo")
@llm_registry.register("gpt-4")
class OpenAIChat(BaseChatModel):
    args: OpenAIChatArgs = Field(default_factory=OpenAIChatArgs)

    def __init__(self, max_retry: int = 3, **kwargs):
        args = OpenAIChatArgs()
        args = args.dict()

        for k, v in args.items():
            args[k] = kwargs.pop(k, v)
        if len(kwargs) > 0:
            logging.warning(f"Unused arguments: {kwargs}")
        super().__init__(args=args, max_retry=max_retry)

    def _construct_messages(self, prompt: str, chat_memory: List[Message], final_prompt: str):
        chat_messages = []
        for item_memory in chat_memory:
            chat_messages.append(str(item_memory.sender) + ": " + str(item_memory.content))
        processed_prompt = [{"role": "user", "content": prompt}]
        for chat_message in chat_messages:
            processed_prompt.append({"role": "assistant", "content": chat_message})
        processed_prompt.append({"role": "user", "content": final_prompt})
        return processed_prompt

    def generate_response(self, prompt: str, chat_memory: List[Message], final_prompt: str) -> LLMResult:
        messages = self._construct_messages(prompt, chat_memory, final_prompt)
        try:
            response = client.chat.completions.create(messages=messages, **self.args.dict())
            return LLMResult(
                content=response.choices[0].message.content,
                send_tokens=response.usage.prompt_tokens,
                recv_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                logger.error(f"Rate limit exceeded. Switching token...")
                set_next_token()
            raise

    async def agenerate_response(self, prompt: str, chat_memory: List[Message], final_prompt: str) -> LLMResult:
        messages = self._construct_messages(prompt, chat_memory, final_prompt)
        try:
            response = await aclient.chat.completions.create(messages=messages, **self.args.dict())
            return LLMResult(
                content=response.choices[0].message.content,
                send_tokens=response.usage.prompt_tokens,
                recv_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                logger.error(f"Rate limit exceeded. Switching token...")
                set_next_token()
            raise


def get_embedding(text: str, attempts=3) -> np.array:
    while attempts > 0:
        try:
            text = text.replace("\n", " ")
            embedding = client.embeddings.create(input=[text], model="llama3-70b-8192")["data"][0]["embedding"]
            return tuple(embedding)
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                logger.error(f"Rate limit exceeded. Switching token...")
                set_next_token()  # Cambia token
            else:
                logger.error(f"HTTP Error: {e}", exc_info=True)
                raise
        except Exception as e:
            logger.error(f"Error {e} when requesting openai models.")
            attempts -= 1
            if attempts > 0:
                logger.error("Retrying...")
                time.sleep(10)  # Aspetta prima di riprovare

    logger.error(f"get_embedding() failed after multiple attempts.")
    raise Exception("Failed to get embedding after multiple attempts")
