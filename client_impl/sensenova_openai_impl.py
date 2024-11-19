

import os
import time
import jwt

if __name__ == '__main__':
    import sys
    from pathlib import Path

    ROOT_DIR = str(Path(__file__).parent.parent)
    if ROOT_DIR not in sys.path:
        sys.path.append(ROOT_DIR)
    
    from openai_impl import OpenAI_Client
else:
    import os
    from .openai_impl import OpenAI_Client

from llm_client_base import *
import openai
from openai.types.chat.chat_completion import ChatCompletion

# config from .env
# SENSENOVA_KEY_ID
# SENSENOVA_SECRET_KEY

# 模型列表
# https://console.sensecore.cn/micro/help/docs/model-as-a-service/nova/overview/compatible-mode


class SensenovaResponseInterpreterError(Exception):
    pass

def encode_jwt_token():
    ak = os.getenv('SENSENOVA_KEY_ID')
    sk = os.getenv('SENSENOVA_SECRET_KEY')

    headers = {
        "alg": "HS256",
        "typ": "JWT"
    }
    payload = {
        "iss": ak,
        "exp": int(time.time()) + 3600 * 24, # 填写您期望的有效时间
        "nbf": int(time.time()) - 5 # 填写您期望的生效时间，此处示例代表当前时间-5秒
    }
    token = jwt.encode(payload, sk, headers=headers)
    return token

class SenseNovaOpenAI_Client(OpenAI_Client):
    support_system_message: bool = True

    server_location = 'west'

    def __init__(self):
        api_key = encode_jwt_token()

        super().__init__(
            api_base_url="https://api.sensenova.cn/compatible-mode/v1/",
            api_key=api_key,
        )

    async def chat_stream_async(self, model_name, history, model_param, client_param):
        try:
            async for chunk in super().chat_stream_async(model_name, history, model_param, client_param):
                yield chunk
        except openai.BadRequestError as e:
            if 'security reasons' in e.message:
                raise SensitiveBlockError() from e

            raise
    
    async def chat_response_callback(self, response: ChatCompletion):
        # assert response.choices, f'No choices in response {response}'
        # assert response.choices[0].message, f'No message in response {response}'
        interpreter_list = []
        if response.choices[0].message.tool_calls:
            for tool_call in response.choices[0].message.tool_calls:
                if tool_call.type == 'interpreter':
                    interpreter_list.append(tool_call)
                
        if interpreter_list:
            raise SensenovaResponseInterpreterError(interpreter_list[0].model_dump_json())
        
        return await super().chat_response_callback(response)
    
    async def chat_async(self, model_name, history, model_param, client_param):
        try:
            return await super().chat_async(model_name, history, model_param, client_param)
        except openai.BadRequestError as e:
            if 'security reasons' in e.message:
                raise SensitiveBlockError() from e
            raise

if __name__ == '__main__':
    import asyncio
    import os

    client = SenseNovaOpenAI_Client()
    model_name = "SenseChat-5"
    history = [{"role": "user", "content": "Hello, how are you?"}]

    model_param = {
        'temperature': 0.01,
    }

    async def main():
        async for chunk in client.chat_stream_async(model_name, history, model_param, client_param={}):
            print(chunk)

    asyncio.run(main())
