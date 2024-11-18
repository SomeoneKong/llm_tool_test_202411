

import os
import time

from llm_client_base import *
from typing import List
import openai
from .openai_impl import OpenAI_Client
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

# config from .env
# 绑定模型
# SPARKAI_HTTP_API_KEY

# for VL, websockets
# SPARKAI_APP_ID
# SPARKAI_API_KEY
# SPARKAI_API_SECRET


class Xunfei_Client(OpenAI_Client):
    # https://www.xfyun.cn/doc/spark/HTTP%E8%B0%83%E7%94%A8%E6%96%87%E6%A1%A3.html

    support_system_message: bool = True
    support_image_message: bool = True

    server_location = 'china'

    def __init__(self):
        self.api_key = os.getenv('SPARKAI_HTTP_API_KEY')
        assert self.api_key is not None

        super().__init__(
            api_base_url='https://spark-api-open.xf-yun.com/v1',
            api_key=self.api_key,
        )

        # for VL
        self.ws_app_id = os.getenv('SPARKAI_APP_ID')
        self.ws_api_key = os.getenv('SPARKAI_API_KEY')
        self.ws_api_secret = os.getenv('SPARKAI_API_SECRET')

    def mapping_model_name(self, model_name):
        assert model_name.startswith('spark-')
        model_version = model_name[len('spark-'):]
        assert model_version in ['lite', 'pro', 'pro-128k', 'max', 'max-32k', '4.0']

        spark_model_name_dict = {
            'lite': 'general',
            'pro': 'generalv3',
            'pro-128k': 'pro-128k',
            'max': 'generalv3.5',
            'max-32k': 'max-32k',
            '4.0': '4.0Ultra',
        }
        return spark_model_name_dict[model_version]

    def stream_chunk_callback(self, chunk: ChatCompletionChunk):
        if chunk.model_extra.get('code'):
            raise Exception(f'Xunfei error: {chunk.model_extra["code"]}, {chunk.model_extra["message"]}')
        return chunk

    async def chat_stream_async(self, model_name, history, model_param, client_param):
        spark_model_name = self.mapping_model_name(model_name)
        try:
            async for chunk in super().chat_stream_async(spark_model_name, history, model_param, client_param):
                yield chunk
        except openai.InternalServerError as e:
            if '根据相关法律法规' in e.message:
                raise SensitiveBlockError() from e
            raise
    
    async def chat_async(self, model_name, history, model_param, client_param):
        spark_model_name = self.mapping_model_name(model_name)
        try:
            return await super().chat_async(spark_model_name, history, model_param, client_param)
        except openai.InternalServerError as e:
            if '根据相关法律法规' in e.message:
                raise SensitiveBlockError() from e
            raise

    def convert_multimodal_message(self, message: list):
        from sparkai.core.messages import ChatMessage, ImageChatMessage

        content = message['content']
        assert MultimodalMessageUtils.check_content_valid(content), f"Invalid content {content}"

        content_part_list = []
        for content_part in content:
            if isinstance(content_part, MultimodalMessageContentPart_Text):
                content_part_list.append(ChatMessage(
                    role=message['role'],
                    content=content_part.text
                ))
            elif isinstance(content_part, MultimodalMessageContentPart_ImageUrl):
                assert False, "Not support ImageUrl"
            elif isinstance(content_part, MultimodalMessageContentPart_ImagePath):
                image = ImageFile.load_from_path(content_part.image_path)
                part_info = ImageChatMessage(
                    role=message['role'],
                    content=image.image_base64,
                )
                content_part_list.append(part_info)

        return content_part_list

    async def multimodal_chat_stream_async(self, model_name, history: List, model_param, client_param):
        # https://www.xfyun.cn/doc/spark/ImageUnderstanding.html

        import sparkai
        from sparkai.llm.llm import ChatSparkLLM
        from sparkai.core.messages import ChatMessage, ImageChatMessage

        model_param = model_param.copy()
        temperature = model_param['temperature']
        max_tokens = model_param.pop('max_tokens', None)

        message_list = []
        for message in history:
            if isinstance(message['content'], str):
                message_list.append(ChatMessage(
                    role=message['role'],
                    content=message['content']
                ))
            elif isinstance(message['content'], list):
                chunks = self.convert_multimodal_message(message)
                for chunk in chunks:
                    message_list.append(chunk)

        assert isinstance(message_list[0], ImageChatMessage), "First message should be image"

        # 官方没有模型名
        assert model_name in ['spark-2.0', 'spark-vl', None]
        model_version = model_name or 'spark-vl'

        url = {
            'spark-2.0': 'wss://spark-api.cn-huabei-1.xf-yun.com/v2.1/image',
            'spark-vl': 'wss://spark-api.cn-huabei-1.xf-yun.com/v2.1/image',
        }[model_version]

        args = {}
        if max_tokens:
            args['max_tokens'] = max_tokens

        start_time = time.time()
        spark = ChatSparkLLM(
            spark_api_url=url,
            spark_app_id=self.ws_app_id,
            spark_api_key=self.ws_api_key,
            spark_api_secret=self.ws_api_secret,
            spark_llm_domain='image',
            temperature=temperature,
            streaming=True,
            **args
        )

        a = spark.astream(message_list)

        role = 'assistant'
        finish_reason = 'stop'
        result_buffer = ''
        usage = None
        first_token_time = None

        try:
            async for message in a:
                # print(message)
                delta = message.content
                if 'token_usage' in message.additional_kwargs:
                    usage = message.additional_kwargs['token_usage']
                    del usage['question_tokens']

                result_buffer += delta
                if delta:
                    if first_token_time is None:
                        first_token_time = time.time()

                    yield LlmResponseChunk(
                        role=role,
                        delta_content=delta,
                        accumulated_content=result_buffer,
                    )


        except sparkai.errors.SparkAIConnectionError as e:
            if e.error_code in [10013, 10014]:
                raise SensitiveBlockError() from e

            raise

        completion_time = time.time()

        yield LlmResponseTotal(
            role=role,
            accumulated_content=result_buffer,
            finish_reason=finish_reason,
            usage=usage,
            first_token_time=first_token_time - start_time if first_token_time else None,
            completion_time=completion_time - start_time,
        )


if __name__ == '__main__':
    import asyncio
    import os

    client = Xunfei_Client()
    model_name = "spark-lite"
    history = [{"role": "user", "content": "Hello, how are you?"}]

    model_param = {
        'temperature': 0.01,
    }

    async def main():
        async for chunk in client.chat_stream_async(model_name, history, model_param, client_param={}):
            print(chunk)

    asyncio.run(main())
