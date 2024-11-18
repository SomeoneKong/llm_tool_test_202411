
import os

from llm_client_base import *
from typing import List
from .openai_impl import OpenAI_Client
import openai

# config from .env
# ZHIPU_API_KEY

# https://open.bigmodel.cn/dev/api#language


class Zhipu_Client(OpenAI_Client):
    support_system_message: bool = True
    support_image_message: bool = True
    
    support_chat_with_bot_profile_simple: bool = True

    server_location = 'china'

    def __init__(self):
        api_key = os.getenv('ZHIPU_API_KEY')
        assert api_key is not None

        super().__init__(
            api_base_url="https://open.bigmodel.cn/api/paas/v4/",
            api_key=api_key,
        )

    async def chat_stream_async(self, model_name, history, model_param, client_param):
        try:
            async for chunk in super().chat_stream_async(model_name, history, model_param, client_param):
                yield chunk
        except openai.BadRequestError as e:
            if '包含不安全或敏感内容' in str(e.body):
                raise SensitiveBlockError() from e
            raise
    
    async def chat_async(self, model_name, history, model_param, client_param):
        try:
            return await super().chat_async(model_name, history, model_param, client_param)
        except openai.BadRequestError as e:
            if '包含不安全或敏感内容' in str(e.body):
                raise SensitiveBlockError() from e
            raise

    async def multimodal_chat_stream_async(self, model_name, history: List, model_param, client_param):
        model_param = model_param.copy()
        if 'max_tokens' in model_param:
            model_param['max_tokens'] = min(1024, model_param['max_tokens'])

        async for chunk in super().multimodal_chat_stream_async(model_name, history, model_param, client_param):
            yield chunk

    async def _chat_with_bot_profile_simple(self, model_name, history, bot_profile_dict, model_param, client_param):
        assert model_name in ["charglm-3"], f"Unsupported model_name: {model_name}"

        bot_setting = {
            'user_name': bot_profile_dict['user']['name'],
            'bot_name': bot_profile_dict['bot']['name'],
            'user_info': bot_profile_dict['user']['content'],
            'bot_info': bot_profile_dict['bot']['content'],
        }

        raw_model_param = model_param.copy()
        raw_model_param['meta'] = bot_setting

        async for chunk in self.chat_stream_async(model_name, history, raw_model_param, client_param):
            yield chunk

search_prompt = """

# 以下是来自互联网的信息：
{search_result}

# 当前日期: 2024-XX-XX

# 要求：
根据最新发布的信息回答用户问题，当回答引用了参考信息时，必须在句末使用对应的[ref_序号]来标明参考信息来源。

"""


if __name__ == '__main__':
    import asyncio
    import os

    prompt = '''
是否存在一个名词大概叫做 `S code`
如果存在，请提供相关信息。

请通过搜索引擎查找相关信息，然后以json形式返回这个术语的信息：
{
    "found": bool,
    "term": str,
    "definition": "..."
}
'''

    client = Zhipu_Client()
    model_name = "glm-4-plus"
    history = [{"role": "user", "content": prompt}]

    tools = [{
        "type": "web_search",
        "web_search": {
            "enable": True,
            "search_query": "S code",
            "search_result": True,  # 禁用False，启用：True，默认为禁用
            "search_prompt": search_prompt
        }
    }]

    model_param = {
        'temperature': 0.01,
        'tools': tools
    }

    async def main():
        async for chunk in client.chat_stream_async(model_name, history, model_param, client_param={}):
            print(chunk)

    asyncio.run(main())
