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
    "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"
]

import time
from collections import deque

# Tracciamento rate per ogni token
request_history = [deque() for _ in tokens]  # una coda per token
token_history = [deque() for _ in tokens]

MAX_REQUESTS_PER_MINUTE = 30
MAX_TOKENS_PER_MINUTE = 6000
bad_tokens_seen = set()

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
                set_next_token()

        except openai.OpenAIError as e:
            error_message = str(e)
            if "organization_restricted" in error_message:
                bad_token = tokens[current_token_index]
                double_check = openai.api_key
                if bad_token not in bad_tokens_seen:
                    with open("bad_tokens.txt", "a") as f:
                        f.write(f"{bad_token} and {double_check}\n")
                    bad_tokens_seen.add(bad_token)
                print(f"[!] Token bloccato per restrizione organizzativa: {bad_token}")
                set_next_token()
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

        except openai.OpenAIError as e:
            error_message = str(e)
            if "organization_restricted" in error_message:
                bad_token = tokens[current_token_index]
                double_check = openai.api_key
                if bad_token not in bad_tokens_seen:
                    with open("bad_tokens.txt", "a") as f:
                        f.write(f"{bad_token} and {double_check}\n")
                    bad_tokens_seen.add(bad_token)
                print(f"[!] Token bloccato per restrizione organizzativa: {bad_token}")
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

        except openai.OpenAIError as e:
            error_message = str(e)
            if "organization_restricted" in error_message:
                bad_token = tokens[current_token_index]
                double_check = openai.api_key
                if bad_token not in bad_tokens_seen:
                    with open("bad_tokens.txt", "a") as f:
                        f.write(f"{bad_token} and {double_check}\n")
                    bad_tokens_seen.add(bad_token)
                print(f"[!] Token bloccato per restrizione organizzativa: {bad_token}")
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

        except openai.OpenAIError as e:
            error_message = str(e)
            if "organization_restricted" in error_message:
                bad_token = tokens[current_token_index]
                double_check = openai.api_key
                if bad_token not in bad_tokens_seen:
                    with open("bad_tokens.txt", "a") as f:
                        f.write(f"{bad_token} and {double_check}\n")
                    bad_tokens_seen.add(bad_token)
                print(f"[!] Token bloccato per restrizione organizzativa: {bad_token}")
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
                set_next_token()

        except openai.OpenAIError as e:
            error_message = str(e)
            if "organization_restricted" in error_message:
                bad_token = tokens[current_token_index]
                double_check = openai.api_key
                if bad_token not in bad_tokens_seen:
                    with open("bad_tokens.txt", "a") as f:
                        f.write(f"{bad_token} and {double_check}\n")
                    bad_tokens_seen.add(bad_token)
                print(f"[!] Token bloccato per restrizione organizzativa: {bad_token}")
                set_next_token()
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
