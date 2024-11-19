

import os
import time

from llm_client_base import *

from mistralai import Mistral
from mistralai.models import ChatCompletionResponse, ChatCompletionChoice

# config from .env
# MISTRAL_API_KEY


class Mistral_Client(LlmClientBase):
    support_system_message: bool = True

    server_location = 'west'

    def __init__(self):
        super().__init__()

        api_key = os.getenv('MISTRAL_API_KEY')
        assert api_key is not None

        self.client = Mistral(api_key=api_key)

    async def chat_stream_async(self, model_name, history, model_param, client_param):
        model_param = model_param.copy()
        temperature = model_param.pop('temperature')
        max_tokens = model_param.pop('max_tokens', None)
        tools = model_param.pop('tools', None)

        start_time = time.time()

        async_response = await self.client.chat.stream_async(
            model=model_name,
            messages=history,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
        )

        role = None
        result_buffer = ''
        finish_reason = None
        usage = None
        first_token_time = None

        async for chunk_resp in async_response:
            chunk = chunk_resp.data
            choice0 = chunk.choices[0]

            if choice0.delta.content:
                if first_token_time is None:
                    first_token_time = time.time()
            if choice0.finish_reason:
                finish_reason = choice0.finish_reason
            if chunk.usage:
                usage = chunk.usage.model_dump()

            result_buffer += choice0.delta.content
            # print(choice0.delta.content)

            yield LlmResponseChunk(
                role=choice0.delta.role,
                delta_content=choice0.delta.content,
                accumulated_content=result_buffer,
            )

        completion_time = time.time()

        yield LlmResponseTotal(
            role=role,
            accumulated_content=result_buffer,
            finish_reason=finish_reason,
            usage=usage or {},
            first_token_time=first_token_time - start_time if first_token_time else None,
            completion_time=completion_time - start_time,
        )

    async def chat_async(self, model_name, history, model_param, client_param):
        model_param = model_param.copy()
        temperature = model_param.pop('temperature')
        max_tokens = model_param.pop('max_tokens', None)
        tools = model_param.pop('tools', None)

        start_time = time.time()

        response: ChatCompletionResponse = await self.client.chat.complete_async(
            model=model_name,
            messages=history,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
        )

        completion_time = time.time()

        finish_reason = None
        usage = None

        choice0: ChatCompletionChoice = response.choices[0]

        if choice0.finish_reason:
            finish_reason = str(choice0.finish_reason)
        if response.usage:
            usage = response.usage.model_dump()

        tool_call_args_result = []
        for tool_call in choice0.message.tool_calls:
            args_str = tool_call.function.arguments
            if isinstance(args_str, dict):
                args_str = json.dumps(args_str, ensure_ascii=False)
            tool_call_args_result.append(LlmToolCallInfo(
                tool_call_id=tool_call.id,
                tool_name=tool_call.function.name,
                tool_args_json=args_str,
            ))

        yield LlmResponseTotal(
            role=choice0.message.role,
            accumulated_content=choice0.message.content,
            finish_reason=finish_reason,
            tool_calls=tool_call_args_result if tool_call_args_result else None,
            usage=usage or {},
            first_token_time=None,
            completion_time=completion_time - start_time,
        )


if __name__ == '__main__':
    import asyncio
    import os

    client = Mistral_Client()
    model_name = "mistral-small-latest"
    history = [{"role": "user", "content": "Hello, how are you?"}]

    model_param = {
        'temperature': 0.01,
    }

    async def main():
        async for chunk in client.chat_stream_async(model_name, history, model_param, client_param={}):
            print(chunk)

    asyncio.run(main())
