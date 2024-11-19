

import os
import time

from llm_client_base import *
from typing import List
from openai import AsyncOpenAI
import openai
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall, Function
# config from .env
# OPENAI_API_KEY


class OpenAI_Client(LlmClientBase):
    support_system_message: bool = True
    support_image_message: bool = True

    server_location = 'west'

    def __init__(self,
                 api_base_url=None,
                 api_key=None,
                 max_retries=None,
                 ):
        super().__init__()
        self.client = AsyncOpenAI(
            base_url=api_base_url,
            api_key=api_key,
            max_retries=max_retries if max_retries is not None else openai._constants.DEFAULT_MAX_RETRIES,
        )

    async def close(self):
        await self.client.close()
        await super().close()

    def _extract_args(self, model_name, model_param, client_param):
        model_param = model_param.copy()
        temperature = model_param.pop('temperature', None)
        max_tokens = model_param.pop('max_tokens', None)
        tools = model_param.pop('tools', None)
        functions = model_param.pop('functions', None)
        json_mode = model_param.get('json_mode', False)

        req_args = dict(
            model=model_name,
            stream=True,
            stream_options={'include_usage': True},
        )
        if temperature:
            req_args['temperature'] = temperature
        if json_mode:
            req_args['response_format'] = {"type": "json_object"}
        if max_tokens:
            req_args['max_tokens'] = max_tokens
        if tools:
            req_args['tools'] = tools
        elif functions:
            tools = [
                {
                    'type': 'function',
                    'function': f,
                }
                for f in functions
            ]
            req_args['tools'] = tools

        return req_args, model_param, client_param

    async def chat_response_callback(self, response: ChatCompletion):
        return response
    
    async def chat_async(self, model_name, history, model_param, client_param):
        req_args, left_model_param, left_client_param = self._extract_args(model_name, model_param, client_param)
        req_args['messages'] = history
        if left_model_param:
            req_args['extra_body'] = left_model_param

        req_args['stream'] = False
        req_args.pop('stream_options')

        start_time = time.time()

        response: ChatCompletion = await self.client.chat.completions.create(**req_args)
        response = await self.chat_response_callback(response)
        
        assert response.choices, f'No choices in response {response}'

        choice0 = response.choices[0]
        message = choice0.message
        completion_time = time.time()

        function_call_info_list = []
        if message.tool_calls:
            if isinstance(message.tool_calls, dict):
                # for xunfei
                tool_call = message.tool_calls
                call_info = {
                    'id': tool_call.get('id', ''),
                    'name': tool_call['function']['name'],
                    'arguments': tool_call['function']['arguments'],
                }
                message.tool_calls = [
                    ChatCompletionMessageToolCall(
                        id=call_info['id'],
                        function=Function(
                            name=call_info['name'],
                            arguments=call_info['arguments'],
                        ),
                        type='function',
                    )
                ]

            for tool_call in message.tool_calls:
                assert tool_call.function, f'No function in tool_call {tool_call}'
                call_info = {
                    'id': tool_call.id,
                    'name': tool_call.function.name,
                    'arguments': tool_call.function.arguments,
                }
                function_call_info_list.append(call_info)

        tool_call_args_result = []
        for function_call_args in function_call_info_list:
            tool_call_args_result.append(LlmToolCallInfo(
                tool_call_id=function_call_args.get('id', None),
                tool_name=function_call_args.get('name') or '',
                tool_args_json=function_call_args.get('arguments') or '',
            ))

        usage = {}
        if response.usage:
            usage = response.usage.model_dump()
        if usage.get('completion_tokens_details') and 'reasoning_tokens' in usage['completion_tokens_details']:
            usage['completion_tokens_details']['response_tokens'] = usage['completion_tokens'] - (usage['completion_tokens_details']['reasoning_tokens'] or 0)

        raw_response_message = response.choices[0].message
        raw_response_message_obj = raw_response_message.model_dump()

        return LlmResponseTotal(
            role=message.role,
            accumulated_content=message.content,
            finish_reason=choice0.finish_reason,
            tool_calls=tool_call_args_result if tool_call_args_result else None,
            raw_response_message=raw_response_message,
            raw_response_message_obj=raw_response_message_obj,
            system_fingerprint=response.system_fingerprint,
            real_model=response.model,
            usage=usage,
            first_token_time=None,
            completion_time=completion_time - start_time,
        )
    
    def stream_chunk_callback(self, chunk: ChatCompletionChunk):
        return chunk

    async def chat_stream_async(self, model_name, history, model_param, client_param):
        req_args, left_model_param, left_client_param = self._extract_args(model_name, model_param, client_param)
        req_args['messages'] = history
        if left_model_param:
            req_args['extra_body'] = left_model_param

        start_time = time.time()

        system_fingerprint = None
        role = None
        result_buffer = ''
        finish_reason = None
        usage = None
        first_token_time = None
        real_model = None
        function_call_info_list = []
        tool_calls_raw = None
        async with await self.client.chat.completions.create(**req_args) as response:
            async for chunk in response:
                chunk = self.stream_chunk_callback(chunk)
                system_fingerprint = chunk.system_fingerprint
                if chunk.choices:
                    finish_reason = chunk.choices[0].finish_reason
                    delta_info = chunk.choices[0].delta
                    if delta_info:
                        if delta_info.role:
                            role = delta_info.role
                        if delta_info.content:
                            result_buffer += delta_info.content

                            if first_token_time is None:
                                first_token_time = time.time()

                            yield LlmResponseChunk(
                                role=role or 'assistant',
                                delta_content=delta_info.content,
                                accumulated_content=result_buffer,
                            )
                        
                        if delta_info.function_call:
                            if first_token_time is None:
                                first_token_time = time.time()

                            if not function_call_info_list:
                                function_call_info_list = [{}]
                            if delta_info.function_call.name:
                                function_call_info_list[0]['name'] = delta_info.function_call.name
                            elif delta_info.function_call.arguments:
                                function_call_info_list[0]['arguments'] = function_call_info_list[0].get('arguments', '') + delta_info.function_call.arguments
                        elif delta_info.tool_calls:
                            tool_calls_raw = delta_info.tool_calls
                            if first_token_time is None:
                                first_token_time = time.time()
                            
                            for tool_call in delta_info.tool_calls:
                                tool_call_idx = tool_call.index
                                if tool_call_idx is None:
                                    if tool_call.id or tool_call.function.name:
                                        tool_call_info = {
                                            'id': tool_call.id,
                                            'name': tool_call.function.name,
                                            'arguments': tool_call.function.arguments,
                                        }
                                        function_call_info_list.append(tool_call_info)
                                    elif tool_call.function.arguments is not None:
                                        function_call_info_list[-1]['arguments'] = function_call_info_list[-1].get('arguments', '') + tool_call.function.arguments
                                else:
                                    while len(function_call_info_list) <= tool_call_idx:
                                        function_call_info_list.append({})

                                    if tool_call.id:
                                        function_call_info_list[tool_call_idx]['id'] = tool_call.id
                                    if tool_call.function.name:
                                        function_call_info_list[tool_call_idx]['name'] = tool_call.function.name
                                    if tool_call.function.arguments:
                                        function_call_info_list[tool_call_idx]['arguments'] = function_call_info_list[tool_call_idx].get('arguments', '') + tool_call.function.arguments

                if chunk.usage:
                    usage = chunk.usage.dict()
                if chunk.model:
                    real_model = chunk.model

        completion_time = time.time()

        tool_call_args_result = []
        for function_call_args in function_call_info_list:
            tool_call_args_result.append(LlmToolCallInfo(
                tool_call_id=function_call_args.get('id', None),
                tool_name=function_call_args.get('name', None),
                tool_args_json=function_call_args.get('arguments', None),
            ))

        yield LlmResponseTotal(
            role=role or 'assistant',
            accumulated_content=result_buffer,
            finish_reason=finish_reason,
            tool_calls=tool_call_args_result if tool_call_args_result else None,
            tool_calls_raw=tool_calls_raw,
            system_fingerprint=system_fingerprint,
            real_model=real_model,
            usage=usage or {},
            first_token_time=first_token_time - start_time if first_token_time else None,
            completion_time=completion_time - start_time,
        )

    def convert_multimodal_message(self, content: list, image_detail_config=None):
        assert MultimodalMessageUtils.check_content_valid(content), f"Invalid content {content}"

        content_part_list = []
        for content_part in content:
            if isinstance(content_part, MultimodalMessageContentPart_Text):
                content_part_list.append({
                    'type': 'text',
                    'text': content_part.text,
                })
            elif isinstance(content_part, MultimodalMessageContentPart_ImageUrl):
                part_info = {
                    'type': 'image_url',
                    'image_url': {
                        'url': content_part.url,
                    }
                }
                if image_detail_config:
                    part_info['image_url']['detail'] = image_detail_config
                content_part_list.append(part_info)
            elif isinstance(content_part, MultimodalMessageContentPart_ImagePath):
                image = ImageFile.load_from_path(content_part.image_path)
                data_str = f'data:{image.mime_type};base64,{image.image_base64}'
                part_info = {
                    'type': 'image_url',
                    'image_url': {
                        'url': data_str,
                    }
                }
                if image_detail_config:
                    part_info['image_url']['detail'] = image_detail_config
                content_part_list.append(part_info)

        return content_part_list

    async def multimodal_chat_stream_async(self, model_name, history: List, model_param, client_param):
        # https://platform.openai.com/docs/guides/vision
        # https://openai.com/api/pricing/

        req_args, left_model_param, left_client_param = self._extract_args(model_name, model_param, client_param)
        if left_model_param:
            req_args['extra_body'] = left_model_param

        message_list = []
        for message in history:
            if isinstance(message['content'], str):
                message_list.append({
                    'role': message['role'],
                    'content': message['content'],
                })
            elif isinstance(message['content'], list):
                message_list.append({
                    'role': message['role'],
                    'content': self.convert_multimodal_message(message['content'], image_detail_config='high'),
                })
        req_args['messages'] = message_list


        start_time = time.time()

        system_fingerprint = None
        role = None
        result_buffer = ''
        finish_reason = None
        usage = None
        first_token_time = None
        real_model = None

        async with await self.client.chat.completions.create(**req_args) as response:
            async for chunk in response:
                # print(chunk)
                system_fingerprint = chunk.system_fingerprint
                if chunk.choices:
                    finish_reason = chunk.choices[0].finish_reason
                    delta_info = chunk.choices[0].delta
                    if delta_info:
                        if delta_info.role:
                            role = delta_info.role
                        if delta_info.content:
                            result_buffer += delta_info.content

                            if first_token_time is None:
                                first_token_time = time.time()

                            yield LlmResponseChunk(
                                role=role or 'assistant',  # for yi-vision
                                delta_content=delta_info.content,
                                accumulated_content=result_buffer,
                            )

                if chunk.usage:
                    usage = chunk.usage.dict()
                if chunk.model:
                    real_model = chunk.model


        completion_time = time.time()

        yield LlmResponseTotal(
            role=role or 'assistant',  # for yi-vision
            accumulated_content=result_buffer,
            finish_reason=finish_reason,
            system_fingerprint=system_fingerprint,
            real_model=real_model,
            usage=usage or {},
            first_token_time=first_token_time - start_time if first_token_time else None,
            completion_time=completion_time - start_time,
        )


if __name__ == '__main__':
    import asyncio
    import os

    os.environ['HTTP_PROXY'] = "http://127.0.0.1:7890/"
    os.environ['HTTPS_PROXY'] = "http://127.0.0.1:7890/"

    client = OpenAI_Client(api_key=os.getenv('OPENAI_API_KEY'))

    def test_text():
        model_name = "gpt-4o-mini"
        history = [{"role": "user", "content": "Hello, how are you?"}]

        model_param = {
            'temperature': 0.01,
            # 'json_mode': True,
        }

        async def main():
            async for chunk in client.chat_stream_async(model_name, history, model_param, client_param={}):
                print(chunk)

        asyncio.run(main())


    def test_image():
        model_name = "gpt-4o-mini"

        prompt = '''这是什么？'''

        history = [{"role": "user", "content": [
            MultimodalMessageContentPart_Text(text=prompt),
            MultimodalMessageContentPart_ImagePath(image_path=r".\image_test\case2.png"),
        ]}]

        model_param = {
            'temperature': 0.01,
            # 'json_mode': True,
        }

        async def main():
            async for chunk in client.multimodal_chat_stream_async(model_name, history, model_param, client_param={}):
                print(chunk.get('delta_content', None), end='')
            print()

        asyncio.run(main())

    test_image()
