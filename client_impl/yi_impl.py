

import os

from llm_client_base import *
from typing import List

from .openai_impl import OpenAI_Client

# config from .env
# YI_API_KEY


class Yi_Client(OpenAI_Client):
    support_system_message: bool = True
    support_image_message: bool = True

    server_location = 'china'

    def __init__(self):
        api_key = os.getenv('YI_API_KEY')

        super().__init__(
            api_base_url="https://api.lingyiwanwu.com/v1",
            api_key=api_key,
        )

    async def multimodal_chat_stream_async(self, model_name, history: List, model_param, client_param):
        model_param = model_param.copy()
        assert model_name in ['yi-vision'], f'Model {model_name} not support vl'

        async for chunk in super().multimodal_chat_stream_async(model_name, history, model_param, client_param):
            yield chunk


if __name__ == '__main__':
    import asyncio
    import os

    client = Yi_Client()
    model_name = "yi-spark"
    history = [{"role": "user", "content": "Hello, how are you?"}]

    model_param = {
        'temperature': 0.01,
    }

    async def main():
        async for chunk in client.chat_stream_async(model_name, history, model_param, client_param={}):
            print(chunk)

    asyncio.run(main())
