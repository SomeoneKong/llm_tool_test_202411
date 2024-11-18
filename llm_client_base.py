import base64
import json
import imghdr
import logging

from typing import Optional, List, Any, Union
from typing_extensions import Self, Literal

import aiohttp

from pydantic import BaseModel

class MultimodalMessageContentPart_Text(BaseModel):
    text: str

class MultimodalMessageContentPart_ImageUrl(BaseModel):
    image_url: str

class MultimodalMessageContentPart_ImagePath(BaseModel):
    image_path: str

class MultimodalMessageContentPart_AudioPath(BaseModel):
    audio_path: str

class MultimodalMessageContentPart_VideoPath(BaseModel):
    video_path: str



class ImageFile:
    def __init__(self,
                 image_path: str,
                 image_format: str,
                 mime_type: str,
                 image_base64: str,
                 ):
        self.image_path = image_path
        self.image_format = image_format
        self.mime_type = mime_type
        self.image_base64 = image_base64

    @staticmethod
    def load_from_path(image_path: str):
        image_format = imghdr.what(image_path)
        if image_format is None:
            logging.warning(f'image {image_path} format detected failed')
            if image_path.endswith('.jpg') or image_path.endswith('.jpeg'):
                image_format = 'jpeg'
            else:
                raise Exception(f'image {image_path} format {image_format} not supported')
        
        mime_type = f"image/{image_format}"

        image_bytes = open(image_path, "rb").read()
        image_base64 = base64.b64encode(image_bytes).decode()

        return ImageFile(
            image_path=image_path,
            image_format=image_format,
            mime_type=mime_type,
            image_base64=image_base64,
        )

class MultimodalMessageUtils:
    @staticmethod
    def check_content_valid(content: list):
        for part in content:
            if isinstance(part, MultimodalMessageContentPart_Text):
                continue
            elif isinstance(part, MultimodalMessageContentPart_ImageUrl):
                continue
            elif isinstance(part, MultimodalMessageContentPart_ImagePath):
                continue
            elif isinstance(part, MultimodalMessageContentPart_AudioPath):
                continue
            elif isinstance(part, MultimodalMessageContentPart_VideoPath):
                continue

            return False, f"part {part} is not valid"

        return True, None


class LlmResponseChunk(BaseModel):
    is_end: bool = False
    role: str
    delta_content: str
    accumulated_content: str
    extra: Optional[dict] = None

class LlmToolCallInfo(BaseModel):
    tool_call_id: Optional[str]  # 旧的functions没有这个字段
    tool_name: str
    tool_args_json: str

class LlmResponseTotal(BaseModel):
    is_end: bool = True
    role: str

    accumulated_content: Optional[str]
    finish_reason: Optional[str]

    tool_calls: Optional[List[LlmToolCallInfo]] = None
    tool_calls_raw: Optional[Any] = None

    raw_response_message: Optional[Any] = None
    raw_response_message_obj: Optional[Any] = None

    first_token_time: Optional[float]
    completion_time: float

    usage: Optional[dict]
    real_model: Optional[str] = None
    system_fingerprint: Optional[str] = None

    extra: Optional[dict] = None


class LlmClientBase:
    support_system_message: bool
    support_image_message: bool = False

    support_chat_with_bot_profile_simple: bool = False
    support_multi_bot_chat: bool = False

    server_location: Literal['china', 'west']

    async def chat_stream_async(self, model_name, history, model_param, client_param):
        raise NotImplementedError()

    async def multimodal_chat_stream_async(self, model_name, history: List, model_param, client_param):
        raise NotImplementedError()

    async def close(self):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()

    async def _parse_sse_response(self, response: aiohttp.ClientResponse):
        pending = None
        async for chunk, _ in response.content.iter_chunks():
            if pending is not None:
                chunk = pending + chunk
            lines = chunk.splitlines()
            if lines and lines[-1] and chunk and lines[-1][-1] == chunk[-1]:
                pending = lines.pop()
            else:
                pending = None

            for line in lines:
                if line.startswith(b'data: '):
                    line = line[6:]
                    if line.startswith(b'{') or line.startswith(b'['):
                        chunk = json.loads(line)
                    else:
                        chunk = line.decode()

                    yield chunk
                elif line.startswith(b'{'):
                    chunk = json.loads(line)
                    yield chunk

        if pending and pending.startswith(b'{'):
            chunk = json.loads(pending)
            yield chunk

    async def _chat_with_bot_profile_simple(self, model_name, history, bot_profile_dict, model_param, client_param):
        """
        支持为user和assistant设置profile，然后进行对话
        最多只支持这两个
        """
        raise NotImplementedError()
        yield NotImplementedError()

    async def _multi_bot_chat(self, model_name, history, group_chat_param, model_param, client_param):
        """
        支持多个bot进行对话，每个bot独立设置profile
        """
        raise NotImplementedError()
        yield NotImplementedError()

    async def two_bot_chat_multi_turns_simple(
        self,
        model_name,
        bot_chat_history,
        group_chat_param,
        model_param,
        client_param,
    ):
        assert self.support_chat_with_bot_profile_simple, "two_bot_chat_multi_turns not supported"

        bot_profile_dict = group_chat_param['bot_profile_dict']
        max_turns = group_chat_param['max_turns']

        assert len(bot_profile_dict) == 2, "two_bot_chat must have two bot profile"
        assert len(bot_chat_history) > 0, "bot_chat_history must have at least one message"
        assert max_turns > 0, "max_turns must be greater than 0"

        for message in bot_chat_history:
            assert message['bot_name'] in bot_profile_dict, f"bot_name {message['bot_name']} not in bot_profile_dict"

        def rewrite_history_for_single_call(bot_chat_history: list, last_bot: str, next_bot: str):
            step_history = []
            for message in bot_chat_history:
                step_role = None
                if message['bot_name'] == last_bot:
                    step_role = 'user'
                elif message['bot_name'] == next_bot:
                    step_role = 'assistant'

                step_history.append({
                    'role': step_role,
                    'content': message['content'],
                })
            return step_history

        accumulated_history = bot_chat_history.copy()

        for round_idx in range(max_turns):
            last_bot = accumulated_history[-1]['bot_name']
            next_bot = list(r for r in bot_profile_dict if r != last_bot)[0]

            step_history = rewrite_history_for_single_call(accumulated_history, last_bot, next_bot)
            step_bot_setting = {
                'bot': {
                    'name': next_bot,
                    'content': bot_profile_dict[next_bot],
                },
                'user': {
                    'name': last_bot,
                    'content': bot_profile_dict[last_bot],
                },
            }

            chunk = {}
            async for chunk in self._chat_with_bot_profile_simple(model_name, step_history, step_bot_setting, model_param, client_param):
                pass

            round_resp_content = chunk.get('accumulated_content', None)
            call_detail = {
                'round_idx': round_idx,
                'finish_reason': chunk.get('finish_reason', None),
                'usage': chunk.get('usage', None),
                'first_token_time': chunk.get('first_token_time', None),
                'completion_time': chunk.get('completion_time', None),
            }

            accumulated_history.append({
                'bot_name': next_bot,
                'content': round_resp_content,
                'call_detail': call_detail,
            })

            yield {
                'round_idx': round_idx,
                'bot_name': next_bot,
                'content': round_resp_content,
                'call_detail': call_detail,
            }

    async def two_bot_chat_multi_turns(
            self,
            model_name,
            bot_chat_history,
            group_chat_param,
            model_param,
            client_param,
    ):
        assert self.support_multi_bot_chat, "multi_bot_chat not supported"

        group_chat_param = group_chat_param.copy()
        bot_profile_dict = group_chat_param['bot_profile_dict']
        max_turns = group_chat_param.pop('max_turns')

        assert len(bot_profile_dict) == 2, "two_bot_chat must have two bot profile"
        assert len(bot_chat_history) > 0, "bot_chat_history must have at least one message"
        assert max_turns > 0, "max_turns must be greater than 0"

        for message in bot_chat_history:
            assert message['bot_name'] in bot_profile_dict, f"bot_name {message['bot_name']} not in bot_profile_dict"

        for round_idx in range(max_turns):
            last_bot = bot_chat_history[-1]['bot_name']
            next_bot = list(r for r in bot_profile_dict if r != last_bot)[0]

            step_group_chat_param = {
                **group_chat_param,
                'next_bot': next_bot,
            }
            async for chunk in self._multi_bot_chat(model_name, bot_chat_history, step_group_chat_param, model_param, client_param):
                pass
            bot_chat_history.append(chunk)

            yield {
                'round_idx': round_idx,
                **chunk
            }


class SensitiveBlockError(Exception):
    pass
