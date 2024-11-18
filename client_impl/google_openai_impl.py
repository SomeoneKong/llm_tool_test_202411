

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

from openai.types.chat.chat_completion import ChatCompletion
# config from .env
# GOOGLE_API_KEY


class ResponseNoChoices(Exception):
    pass

class GeminiOpenAI_Client(OpenAI_Client):
    support_system_message: bool = True

    server_location = 'west'

    def __init__(self):
        api_key = os.getenv('GOOGLE_API_KEY')

        super().__init__(
            api_base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=api_key,
        )
    
    async def chat_response_callback(self, response: ChatCompletion):
        if not response.choices:
            raise ResponseNoChoices(f'no choices, {response}')
        return response
    
    async def chat_async(self, model_name, history, model_param, client_param={}):
        for retry in range(3):
            try:
                response = await super().chat_async(model_name, history, model_param, client_param)
                if not response.accumulated_content or response.tool_calls:
                    print(f'{model_name} has no content or tool call, retry {retry}')
                    continue
                return response
            except ResponseNoChoices as e:
                if retry == 2:
                    raise
                print(f'{model_name} no choices, retry {retry}')
            
        return None

if __name__ == '__main__':
    import asyncio
    import os

    os.environ['HTTP_PROXY'] = "http://127.0.0.1:7890/"
    os.environ['HTTPS_PROXY'] = "http://127.0.0.1:7890/"

    client = GeminiOpenAI_Client()
    model_name = "gemini-1.5-flash"
    history = [{"role": "user", "content": "Hello, how are you?"}]

    model_param = {
        'temperature': 0.01,
    }

    async def main():
        async for chunk in client.chat_stream_async(model_name, history, model_param, client_param={}):
            print(chunk)

    asyncio.run(main())
