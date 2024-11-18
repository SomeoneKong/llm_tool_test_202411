

import os

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

import asyncio
import aiohttp

from llm_client_base import *

# pip install baidu-bce-auth
from bceauth.auth import make_auth

# config from .env
# QIANFAN_ACCESS_KEY
# QIANFAN_SECRET_KEY

# 对于openai兼容接口的模型名列表：
# https://cloud.baidu.com/doc/WENXINWORKSHOP/s/Fm2vrveyu


class Baidu_Client(OpenAI_Client):
    support_system_message: bool = True

    server_location = 'china'

    def __init__(self):
        super().__init__(
            api_key='placeholder',
            api_base_url="https://qianfan.baidubce.com/v2",
        )
        self.api_key = None
        self.lock = asyncio.Lock()
    
    async def _get_auth(self):
        path = '/v1/BCE-BEARER/token'
        host = 'iam.bj.baidubce.com'
        url = f"https://{host}{path}"
        
        expireInSeconds = 86400
        ak = os.getenv('QIANFAN_ACCESS_KEY')
        sk = os.getenv('QIANFAN_SECRET_KEY')
        assert ak and sk, "QIANFAN_ACCESS_KEY and QIANFAN_SECRET_KEY must be set"

        params = {
            'expireInSeconds': expireInSeconds,
        }
        auth = make_auth(
            ak=ak,
            sk=sk,
            method='GET',
            path=path,
            params=params,
            headers={
                'Host': host,
            },
        )

        headers={
            'Authorization': auth,
            'Host': host,
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, params=params) as response:
                response_json = await response.json()
                return response_json['token']

    async def chat_stream_async(self, model_name, history, model_param, client_param):
        async with self.lock:
            if self.api_key is None:
                self.api_key = await self._get_auth()
                self.client.api_key = self.api_key
        async for chunk in super().chat_stream_async(model_name, history, model_param, client_param):
            yield chunk

    async def chat_async(self, model_name, history, model_param, client_param):
        async with self.lock:
            if self.api_key is None:
                self.api_key = await self._get_auth()
                self.client.api_key = self.api_key
        return await super().chat_async(model_name, history, model_param, client_param)

if __name__ == '__main__':
    # Load environment variables from .env file
    from dotenv import load_dotenv
    load_dotenv()

    import asyncio
    import os

    client = Baidu_Client()
    model_name = "ernie-3.5-8k"
    history = [{"role": "user", "content": "Hello, how are you?"}]

    model_param = {
        'temperature': 0.01,
    }

    async def main():
        async for chunk in client.chat_stream_async(model_name, history, model_param, client_param={}):
            print(chunk)

    asyncio.run(main())
