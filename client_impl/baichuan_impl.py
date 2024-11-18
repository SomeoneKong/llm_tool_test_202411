

import os

import llm_client_base

from .openai_impl import OpenAI_Client

# config from .env
# BAICHUAN_API_KEY


class Baichuan_Client(OpenAI_Client):
    support_system_message: bool = True
    support_chat_with_bot_profile_simple: bool = True

    server_location = 'china'

    def __init__(self):
        api_key = os.getenv('BAICHUAN_API_KEY')

        super().__init__(
            api_base_url="https://api.baichuan-ai.com/v1/",
            api_key=api_key,
        )

    async def chat_stream_async(self, model_name, history, model_param, client_param):
        async for chunk in super().chat_stream_async(model_name, history, model_param, client_param):
            if isinstance(chunk, llm_client_base.LlmResponseTotal):
                usage = chunk.usage
                if usage.get('completion_tokens', None) == 0:
                    assert False, f"Baichuan return empty"

            yield chunk

    async def _chat_with_bot_profile_simple(self, model_name, history, bot_profile_dict, model_param, client_param):
        assert model_name in ['Baichuan-NPC-Turbo', 'Baichuan-NPC-Lite'], f"Unsupported model_name: {model_name}"

        bot_setting = {
            'user_name': bot_profile_dict['user']['name'],
            'character_name': bot_profile_dict['bot']['name'],
            'user_info': bot_profile_dict['user']['content'],
            'character_info': bot_profile_dict['bot']['content'],
        }

        raw_model_param = model_param.copy()
        raw_model_param['character_profile'] = bot_setting

        async for chunk in self.chat_stream_async(model_name, history, raw_model_param, client_param):
            yield chunk


if __name__ == '__main__':
    import asyncio
    import os

    client = Baichuan_Client()
    model_name = "Baichuan3-Turbo-128k"
    history = [{"role": "user", "content": "Hello, how are you?"}]

    model_param = {
        'temperature': 0.01,
    }

    async def main():
        async for chunk in client.chat_stream_async(model_name, history, model_param, client_param={}):
            print(chunk)

    asyncio.run(main())
