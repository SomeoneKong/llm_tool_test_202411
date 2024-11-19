

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

import json

# config from .env
# OPENROUTER_API_KEY

# https://openrouter.ai/docs/quick-start
# 模型列表 https://openrouter.ai/models


class OpenRouter_Client(OpenAI_Client):
    support_system_message: bool = True
    support_image_message: bool = True

    server_location = 'west'

    def __init__(self):
        api_key = os.getenv('OPENROUTER_API_KEY')
        assert api_key is not None
        self.api_key = api_key

        super().__init__(
            api_base_url="https://openrouter.ai/api/v1/",
            api_key=api_key,
        )
    
    def update_model_param(self, model_name, model_param):
        # deepcopy
        model_param_temp = json.loads(json.dumps(model_param))
        
        model_name_temp = model_name
        if '|' in model_name:
            chunks = model_name.split('|')
            model_name_temp = chunks[0]
            provider = chunks[1]
            model_param_temp['provider'] = {'order': [provider]}
        return model_name_temp, model_param_temp

    async def chat_stream_async(self, model_name, history, model_param, client_param):
        model_name_temp, model_param_temp = self.update_model_param(model_name, model_param)
        
        async for chunk in super().chat_stream_async(model_name_temp, history, model_param_temp, client_param):
            yield chunk

    async def chat_async(self, model_name, history, model_param, client_param):
        model_name_temp, model_param_temp = self.update_model_param(model_name, model_param)
        return await super().chat_async(model_name_temp, history, model_param_temp, client_param)

if __name__ == '__main__':
    import asyncio
    import os

    client = OpenRouter_Client()
    model_name = "anthropic/claude-3.5-sonnet"
    history = [{"role": "user", "content": "Hello, how are you?"}]

    model_param = {
        'temperature': 0.01,
    }

    async def main():
        async for chunk in client.chat_stream_async(model_name, history, model_param, client_param={}):
            print(chunk)

    asyncio.run(main())
