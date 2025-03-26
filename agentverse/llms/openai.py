import logging
import numpy as np
import time
import os
import random
from typing import Dict, List, Optional, Union
from collections import deque


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
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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"
]

import time
from collections import deque

# Tracciamento rate per ogni token
request_history = [deque() for _ in tokens]  # una coda per token
token_history = [deque() for _ in tokens]

MAX_REQUESTS_PER_MINUTE = 30
MAX_TOKENS_PER_MINUTE = 6000

current_token_index = 0  # Indice del token corrente (globale)


def set_next_token():
    """Imposta il prossimo token come chiave API."""
    global current_token_index, client, aclient  # Usa la variabile globale
    current_token_index = (current_token_index + 1) % len(tokens)
    openai.api_key = tokens[current_token_index]

    # Ri-crea i client con il nuovo token
    client = OpenAI(api_key=openai.api_key)
    aclient = AsyncOpenAI(api_key=openai.api_key)
    
    logger.info(f"Switched to token: {openai.api_key}")

def check_rate_limit_and_wait(estimated_tokens: int = 1000):
    global request_history, token_history, current_token_index

    now = time.time()
    rq_hist = request_history[current_token_index]
    tk_hist = token_history[current_token_index]

    # Pulisci richieste/token più vecchie di 60 secondi
    while rq_hist and now - rq_hist[0] > 60:
        rq_hist.popleft()
    while tk_hist and now - tk_hist[0][1] > 60:
        tk_hist.popleft()

    total_tokens = sum(t for t, _ in tk_hist)

    if len(rq_hist) >= MAX_REQUESTS_PER_MINUTE and total_tokens > MAX_TOKENS_PER_MINUTE:
        logger.warning("Rate limit reached — switching token...")

        prev_token_index = current_token_index
        set_next_token()

        # Se siamo tornati al primo token, resettiamo gli storici
        if current_token_index == 0 and prev_token_index != 0:
            logger.info("Completed full token cycle — resetting per-minute usage counters.")
            request_history = [deque() for _ in tokens]
            token_history = [deque() for _ in tokens]

        check_rate_limit_and_wait(estimated_tokens)  # Ricontrolla il nuovo token
    else:
        rq_hist.append(now)
        tk_hist.append((estimated_tokens, now))

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
        check_rate_limit_and_wait(estimated_tokens=1000)
        try:
            response = client.completions.create(prompt=prompt, **self.args.dict())
            return LLMResult(
                content=response.choices[0].text,
                send_tokens=response.usage.prompt_tokens,
                recv_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )
        except openai.RateLimitError as e:
            if e.response.status_code == 429:
                logger.error(f"Rate limit exceeded. Switching token...")
                set_next_token()  # Cambia token
            raise  # Rilancia l'eccezione per farla gestire dallo script principale

    async def agenerate_response(self, prompt: str, chat_memory: List[Message], final_prompt: str) -> LLMResult:
        check_rate_limit_and_wait(estimated_tokens=1000)
        try:
            response = await aclient.completions.create(prompt=prompt, **self.args.dict())
            return LLMResult(
                content=response.choices[0].text,
                send_tokens=response.usage.prompt_tokens,
                recv_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )
        except openai.RateLimitError as e:
            if e.response.status_code == 429:
                logger.error(f"Rate limit exceeded. Switching token...")
                set_next_token()
            raise

@llm_registry.register("gemma2-9b-it")
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
        chat_messages = deque(maxlen=8)
        for item_memory in chat_memory:
            chat_messages.append(str(item_memory.sender) + ": " + str(item_memory.content))
        processed_prompt = [{"role": "user", "content": prompt}]
        for chat_message in chat_messages:
            processed_prompt.append({"role": "assistant", "content": chat_message})
        processed_prompt.append({"role": "user", "content": final_prompt})
        return processed_prompt

    def generate_response(self, prompt: str, chat_memory: List[Message], final_prompt: str) -> LLMResult:
        messages = self._construct_messages(prompt, chat_memory, final_prompt)
        estimated_tokens = sum(len(m["content"]) for m in messages)  # stima grezza
        try:
            check_rate_limit_and_wait(estimated_tokens)
            response = client.chat.completions.create(messages=messages, **self.args.dict())
            return LLMResult(
                content=response.choices[0].message.content,
                send_tokens=response.usage.prompt_tokens,
                recv_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )
        except openai.RateLimitError as e:
            if e.response.status_code == 429:
                logger.error(f"Rate limit exceeded. Switching token...")
                set_next_token()
            raise

    async def agenerate_response(self, prompt: str, chat_memory: List[Message], final_prompt: str) -> LLMResult:
        messages = self._construct_messages(prompt, chat_memory, final_prompt)
        estimated_tokens = sum(len(m["content"]) for m in messages)  # stima grezza
        try:
            check_rate_limit_and_wait(estimated_tokens)
            response = await aclient.chat.completions.create(messages=messages, **self.args.dict())
            return LLMResult(
                content=response.choices[0].message.content,
                send_tokens=response.usage.prompt_tokens,
                recv_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )
        except openai.RateLimitError as e:
            if e.response.status_code == 429:
                logger.error(f"Rate limit exceeded. Switching token...")
                set_next_token()
            raise

def get_embedding(text: str, attempts=3) -> np.array:
    check_rate_limit_and_wait(estimated_tokens=1000)
    while attempts > 0:
        try:
            text = text.replace("\n", " ")
            embedding = client.embeddings.create(input=[text], model="gemma2-9b-it")["data"][0]["embedding"]
            return tuple(embedding)
        except openai.RateLimitError as e:
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
