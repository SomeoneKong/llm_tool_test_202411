

import os
import time
import aiohttp
import json

from openai.types.chat.chat_completion import ChatCompletion

from llm_client_base import *

from .openai_impl import OpenAI_Client

# config from .env
# MINIMAX_API_KEY

class Minimax_Client(OpenAI_Client):
    support_system_message: bool = True
    support_chat_with_bot_profile_simple: bool = True
    support_multi_bot_chat: bool = True

    server_location = 'china'

    def __init__(self):
        api_key = os.getenv('MINIMAX_API_KEY')

        super().__init__(
            api_base_url="https://api.minimax.chat/v1/",
            api_key=api_key,
        )

        self.group_id = os.getenv('MINIMAX_GROUP_ID')
        self.pro_api_key = api_key

    def fix_model_param(self, model_param):
        # deep copy
        temp_model_param = json.loads(json.dumps(model_param))
        
        if temp_model_param['temperature'] is not None:
            temp_model_param['temperature'] = max(0.01, temp_model_param['temperature'])
            
        if 'max_tokens' not in temp_model_param:
            temp_model_param['max_tokens'] = 1024*4  # 官方默认值为256，太短
        
        for tool in temp_model_param.get('tools', []):
            tool['function']['parameters'] = json.dumps(tool['function']['parameters'])
        
        return temp_model_param
    
    async def chat_stream_async(self, model_name, history, model_param, client_param):
        temp_model_param = self.fix_model_param(model_param)

        has_delta_chunk = False

        async for chunk in super().chat_stream_async(model_name, history, temp_model_param, client_param):
            if chunk.is_end:
                assert has_delta_chunk, f"minimax return empty"
            else:
                has_delta_chunk = True

            yield chunk
    
    async def chat_async(self, model_name, history, model_param, client_param):
        temp_model_param = self.fix_model_param(model_param)

        return await super().chat_async(model_name, history, temp_model_param, client_param)
    async def _chat_pro_stream_async(self, model_name, history, bot_profile_dict, model_param, client_param):
        model_param = model_param.copy()
        temperature = model_param.pop('temperature')
        if temperature is not None:
            temperature = max(0.01, temperature)

        max_tokens = model_param.pop('max_tokens', 2048)  # 官方默认值为256，太短

        url = "https://api.minimax.chat/v1/text/chatcompletion_pro?GroupId=" + self.group_id
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.pro_api_key}',
        }

        bot_profile = bot_profile_dict['bot']
        user_profile = bot_profile_dict['user']

        message_list = []
        for message in history:
            if message['role'] == 'user':
                message_list.append({
                    "sender_type": "USER",
                    "sender_name": user_profile['name'],
                    "text": message['content'],
                })
            elif message['role'] == 'assistant':
                message_list.append({
                    "sender_type": "BOT",
                    "sender_name": bot_profile['name'],
                    "text": message['content'],
                })

        payload = {
            "bot_setting": [
                {
                    "bot_name": bot_profile['name'],
                    "content": bot_profile['content'],
                }
            ],
            "messages": message_list,
            "reply_constraints": {"sender_type": "BOT", "sender_name": bot_profile['name']},
            "model": model_name,
            "stream": True,
            # "top_p": 0.95,
        }
        if temperature is not None:
            payload['temperature'] = temperature
        if max_tokens is not None:
            payload['tokens_to_generate'] = max_tokens

        start_time = time.time()

        role = None
        result_buffer = ''
        finish_reason = None
        usage = None
        first_token_time = None

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as response:
                async for chunk in self._parse_sse_response(response):
                    choice0 = chunk['choices'][0]
                    # print(choice0)
                    role = choice0['messages'][0]['sender_type']
                    if 'finish_reason' in choice0:
                        finish_reason = choice0['finish_reason']
                        delta_data = ''
                    else:
                        delta_data = choice0['messages'][0]['text']
                        result_buffer += delta_data
                    if 'usage' in chunk:
                        usage = chunk['usage']

                    if role == 'BOT':
                        role = 'assistant'
                    elif role == 'USER':
                        role = 'user'

                    if delta_data:
                        if first_token_time is None:
                            first_token_time = time.time()

                        yield LlmResponseChunk(
                            role=role,
                            delta_content=delta_data,
                            accumulated_content=result_buffer,
                        )

        completion_time = time.time()

        yield LlmResponseTotal(
            role=role,
            accumulated_content=result_buffer,
            finish_reason=finish_reason,
            usage=usage,
            first_token_time=first_token_time - start_time if first_token_time else None,
            completion_time=completion_time - start_time,
        )


    async def _chat_with_bot_profile_simple(self, model_name, history, bot_profile_dict, model_param, client_param):
        async for chunk in self._chat_pro_stream_async(model_name, history, bot_profile_dict, model_param, client_param):
            yield chunk

    async def two_bot_chat_pro(
            self,
            model_name,
            bot_chat_history,
            group_chat_param,
            model_param,
            client_param,
        ):
        async for chunk in self.two_bot_chat_multi_turns_simple(model_name, bot_chat_history, group_chat_param, model_param, client_param):
            yield chunk

    async def _multi_bot_chat(
            self,
            model_name,
            bot_chat_history,
            group_chat_param,
            model_param,
            client_param,
        ):
        # https://platform.minimaxi.com/document/ChatCompletion%20v2?key=66701d281d57f38758d581d0#f88XYJdDmLFMmrWk1YRmDrSF

        assert model_name in ['abab6.5t-chat'], "model_name must be in ['abab6.5t-chat']"

        bot_profile_dict = group_chat_param['bot_profile_dict']
        group_name = group_chat_param['group_name']
        group_content = group_chat_param['group_content']
        next_bot = group_chat_param['next_bot']

        assert len(bot_chat_history) > 0, "bot_chat_history must have at least one message"

        for message in bot_chat_history:
            assert message['bot_name'] in bot_profile_dict, f"bot_name {message['bot_name']} not in bot_profile_dict"

        accumulated_history = []

        # Minimax的Group chat配置
        for bot, content in bot_profile_dict.items():
            accumulated_history.append({
                "role": "system",
                "name": bot,
                "content": content,
            })

        # 似乎也是必填，但文档没说
        accumulated_history.append({
            "role": "user_system",
            "name": "用户",
            "content": "用户",
        })

        # 必填
        accumulated_history.append({
            "role": "group",
            "name": group_name,
            "content": group_content,
        })

        # 历史消息
        for message in bot_chat_history:
            accumulated_history.append({
                "role": "assistant",
                "name": message['bot_name'],
                "content": message['content'],
            })

        # 指定下一个bot
        accumulated_history.append({
            "role": "assistant",
            "name": next_bot,
            "content": "",
        })

        model_param = model_param.copy()
        temperature = model_param.pop('temperature')
        max_tokens = model_param.pop('max_tokens', 2048)

        start_time = time.time()

        result_buffer = ''
        finish_reason = None
        first_token_time = None
        real_model = None

        req_args = dict(
            model=model_name,
            messages=accumulated_history,
            temperature=temperature,
            stream=True,
        )
        if max_tokens:
            req_args['max_tokens'] = max_tokens
        if model_param:
            req_args['extra_body'] = model_param

        async with await self.client.chat.completions.create(**req_args) as response:
            async for chunk in response:
                # print(chunk)
                if chunk.choices:
                    finish_reason = chunk.choices[0].finish_reason
                    delta_info = chunk.choices[0].delta
                    if delta_info:
                        if delta_info.content:
                            result_buffer += delta_info.content

                            if first_token_time is None:
                                first_token_time = time.time()

                            yield {
                                'bot_name': next_bot,
                                'delta_content': delta_info.content,
                                'accumulated_content': result_buffer,
                            }
                if chunk.model:
                    real_model = chunk.model


        completion_time = time.time()

        yield {
            'bot_name': next_bot,
            'content': result_buffer,
            'call_detail': {
                'finish_reason': finish_reason,
                'real_model': real_model,
                'first_token_time': first_token_time - start_time if first_token_time else None,
                'completion_time': completion_time - start_time,
            }
        }


if __name__ == '__main__':
    import asyncio
    import os

    client = Minimax_Client()
    model_name = "abab6.5s-chat"

    model_param = {
        'temperature': 0.01,
    }

    async def main():
        history = [{"role": "user", "content": "Hello, how are you?"}]
        async for chunk in client.chat_stream_async(model_name, history, model_param, client_param={}):
            print(chunk)

    asyncio.run(main())
