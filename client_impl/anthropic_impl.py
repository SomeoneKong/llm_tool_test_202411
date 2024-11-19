import os
import time

from llm_client_base import *

# pip install anthropic
import anthropic
from anthropic.types import *
from anthropic.lib.streaming._types import *

# config from .env
# ANTHROPIC_API_KEY
# HTTP_PROXY
# HTTPS_PROXY



class Anthropic_Client(LlmClientBase):
    support_system_message: bool = True

    server_location = 'west'

    def __init__(self, api_key=None, headers=None):
        super().__init__()

        if api_key is None:
            api_key = os.getenv('ANTHROPIC_API_KEY')
        assert api_key is not None

        self.client = anthropic.AsyncAnthropic(
            api_key=api_key,
            default_headers=headers,
        )

    async def chat_stream_async(self, model_name, history, model_param, client_param):
        model_param = model_param.copy()
        temperature = model_param['temperature']
        input_tools = model_param.pop('tools', None)
        max_tokens = model_param.pop('max_tokens', None)  # 必选项
        if '3-5' in model_name:
            max_tokens = max_tokens or 1024 * 8
        elif '3-haiku' in model_name:
            max_tokens = max_tokens or 1024 * 4
        else:
            max_tokens = max_tokens or 1024 * 3

        system_message_list = [m for m in history if m['role'] == 'system']
        system_prompt = system_message_list[-1]['content'] if system_message_list else []

        message_list = [m for m in history if m['role'] != 'system']

        tools = []
        if input_tools:
            for tool in input_tools:
                tool = tool['function']
                tool = {**tool}
                parameters = tool.pop('parameters', None)
                assert parameters is not None, 'tools parameters is required'
                tool['input_schema'] = parameters
                tools.append(tool)
        
        if len(tools) == 0:
            tools = None

        start_time = time.time()
        first_token_time = None
        
        content_part_list = []
        message_stop_event = None
        async with self.client.messages.stream(
                model=model_name,
                messages=message_list,
                system=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
        ) as stream:
            async for event in stream:
                delta = None
                if isinstance(event, MessageStartEvent):
                    continue
                elif isinstance(event, ContentBlockStartEvent):
                    content_part_list.append(event.content_block)
                    continue
                elif isinstance(event, ContentBlockStopEvent):
                    content_part_list[-1] = event.content_block
                    if isinstance(event.content_block, TextBlock):
                        continue
                    else:
                        # ToolUseBlock，跟结束时候一起返回
                        pass
                elif isinstance(event, TextEvent):
                    content_part_list[-1].text += event.text
                    delta = event.text
                elif isinstance(event, ContentBlockDeltaEvent):
                    if isinstance(event.delta, TextBlock):
                        content_part_list[-1].text += event.delta.text
                        delta = event.delta.text
                    elif isinstance(event.delta, InputJSONDelta):
                        # 丢弃tool call时候的中间event
                        continue
                elif isinstance(event, InputJsonEvent):
                    continue
                elif isinstance(event, RawMessageDeltaEvent):
                    continue
                elif isinstance(event, MessageStopEvent):
                    message_stop_event = event
                    continue
                else:
                    raise Exception(f'unsupported chunk type: {event}')
                
                if delta and first_token_time is None:
                    first_token_time = time.time()
                
                if delta:
                    yield LlmResponseChunk(
                        role='assistant',
                        delta_content=delta,
                        accumulated_content=content_part_list[-1].text,
                    )

        completion_time = time.time()

        total_message = message_stop_event.message
        
        result_text = ''
        tool_use_list = []
        for content_block in total_message.content:
            if content_block.type == 'text':
                result_text += content_block.text
            elif content_block.type == 'tool_use':
                tool_use_list.append(LlmToolCallInfo(
                    tool_call_id=content_block.id,
                    tool_name=content_block.name,
                    tool_args_json=json.dumps(content_block.input, ensure_ascii=False),
                ))
            else:
                raise Exception(f'unsupported content type: {content_block.type}')

        usage = {
            'prompt_tokens': total_message.usage.input_tokens,
            'completion_tokens': total_message.usage.output_tokens,
        }

        raw_response_message = {
            'role': total_message.role,
            'content': total_message.content,
        }
        
        yield LlmResponseTotal(
            role=total_message.role,
            accumulated_content=result_text,
            tool_calls=tool_use_list,
            finish_reason=total_message.stop_reason,
            raw_response_message=raw_response_message,
            raw_response_message_obj=total_message.model_dump(),
            real_model=total_message.model,
            usage=usage,
            first_token_time=first_token_time - start_time if first_token_time else None,
            completion_time=completion_time - start_time,
        )
    
    async def chat_async(self, model_name, history, model_param, client_param):
        model_param = model_param.copy()
        temperature = model_param['temperature']
        input_tools = model_param.pop('tools', None)
        max_tokens = model_param.pop('max_tokens', None)  # 必选项
        if '3-5' in model_name:
            max_tokens = max_tokens or 1024 * 8
        elif '3-haiku' in model_name:
            max_tokens = max_tokens or 1024 * 4
        else:
            max_tokens = max_tokens or 1024 * 3

        system_message_list = [m for m in history if m['role'] == 'system']
        system_prompt = system_message_list[-1]['content'] if system_message_list else []

        message_list = [m for m in history if m['role'] != 'system']

        tools = []
        if input_tools:
            for tool in input_tools:
                tool = tool['function']
                tool = {**tool}
                parameters = tool.pop('parameters', None)
                assert parameters is not None, 'tools parameters is required'
                tool['input_schema'] = parameters
                tools.append(tool)
        
        if len(tools) == 0:
            tools = None

        start_time = time.time()
        response = await self.client.messages.create(
                model=model_name,
                messages=message_list,
                system=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
        )
        content_block_list = response.content
        result_text = ''
        tool_use_list = []
        for content_block in content_block_list:
            if content_block.type == 'text':
                result_text += content_block.text
            elif content_block.type == 'tool_use':
                tool_use_list.append(LlmToolCallInfo(
                    tool_call_id=content_block.id,
                    tool_name=content_block.name,
                    tool_args_json=json.dumps(content_block.input, ensure_ascii=False),
                ))
            else:
                raise Exception(f'unsupported content type: {content_block.type}')
        
        raw_response_message = {
            'role': response.role,
            'content': response.content,
        }
        
        return LlmResponseTotal(
            role=response.role,
            accumulated_content=result_text,
            tool_calls=tool_use_list,
            finish_reason=response.stop_reason,
            raw_response_message=raw_response_message,
            raw_response_message_obj=response.model_dump(),
            real_model=response.model,
            usage={
                'prompt_tokens': response.usage.input_tokens,
                'completion_tokens': response.usage.output_tokens,
            },
            first_token_time=None,
            completion_time=time.time() - start_time,
        )

    async def close(self):
        await self.client.close()

if __name__ == '__main__':
    import asyncio
    import os

    os.environ['HTTP_PROXY'] = "http://127.0.0.1:7890/"
    os.environ['HTTPS_PROXY'] = "http://127.0.0.1:7890/"

    client = Anthropic_Client()
    model_name = "claude-3-haiku-20240307"
    history = [{"role": "user", "content": "Hello, how are you?"}]

    model_param = {
        'temperature': 0.01,
    }

    async def main():
        async for chunk in client.chat_stream_async(model_name, history, model_param, client_param={}):
            print(chunk)

    asyncio.run(main())
