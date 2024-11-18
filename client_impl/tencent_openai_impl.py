

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

from llm_client_base import *

# config from .env
# TENCENT_OPENAI_API_KEY


class TencentOpenAI_Client(OpenAI_Client):
    support_system_message: bool = True

    server_location = 'west'

    def __init__(self):
        api_key = os.getenv('TENCENT_OPENAI_API_KEY')

        super().__init__(
            api_base_url="https://api.hunyuan.cloud.tencent.com/v1",
            api_key=api_key,
        )

if __name__ == '__main__':
    import asyncio
    import os

    client = TencentOpenAI_Client()
    model_name = "hunyuan-functioncall"
    history = [{"role": "user", "content": "Hello, how are you?"}]

    model_param = {
        'temperature': 0.01,
    }

    async def main():
        async for chunk in client.chat_stream_async(model_name, history, model_param, client_param={}):
            print(chunk)

    asyncio.run(main())
