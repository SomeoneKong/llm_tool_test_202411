

import os
import time
import sys
if __name__ == '__main__':
    base_path = os.path.join(os.path.dirname(__file__), '..')
    sys.path.insert(0, base_path)


from llm_client_base import *
from typing import List

# pip install google-generativeai
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold, Tool, ContentDict
from google.generativeai import protos
from google.generativeai.types import ContentDict
from google.generativeai.types.content_types import to_part, to_content
import PIL.Image

# config from .env
# GOOGLE_API_KEY

def proto_to_dict(proto_obj):
    type(proto_obj).to_dict(proto_obj)

class Gemini_Client(LlmClientBase):
    support_system_message: bool = True
    support_image_message: bool = True

    server_location = 'west'

    def __init__(self, auto_break_on_first_tool_call: bool = True):
        super().__init__()

        api_key = os.getenv('GOOGLE_API_KEY')
        assert api_key is not None
        
        self.auto_break_on_first_tool_call = auto_break_on_first_tool_call

        genai.configure(api_key=api_key)

    def role_convert_to_openai(self, role):
        if role == 'user':
            return 'user'
        elif role == 'model':
            return 'assistant'
        else:
            return 'unknown'

    def role_convert_from_openai(self, role):
        if role == 'user':
            return 'user'
        elif role == 'assistant':
            return 'model'
        elif role == 'tool':
            return 'user'
        elif role == 'model':
            return 'model'
        else:
            raise Exception(f'Unsupported role {role}')

    def get_safety_settings(self):
        return {
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

    def _tool_declare_convert_type(self, type_str):
        if type_str == 'object':
            return genai.protos.Type.OBJECT
        elif type_str == 'array':
            return genai.protos.Type.ARRAY
        elif type_str == 'string':
            return genai.protos.Type.STRING
        elif type_str == 'number':
            return genai.protos.Type.NUMBER
        elif type_str == 'integer':
            return genai.protos.Type.INTEGER
        elif type_str == 'boolean':
            return genai.protos.Type.BOOLEAN
        else:
            assert False, f"Unsupported type {type_str}"
    
    def _tool_declare_convert_obj(self, obj):
        genai_type = self._tool_declare_convert_type(obj['type'])
        if genai_type == genai.protos.Type.OBJECT:
            return genai.protos.Schema(
                type_=genai_type,
                properties={k: self._tool_declare_convert_obj(v) for k, v in obj['properties'].items()},
                required=obj.get('required', []),
                description=obj.get('description', None),
            )
        elif genai_type == genai.protos.Type.ARRAY:
            return genai.protos.Schema(
                type_=genai_type,
                items=self._tool_declare_convert_obj(obj['items']),
                description=obj.get('description', None),
            )
        elif genai_type in [genai.protos.Type.STRING, genai.protos.Type.NUMBER, genai.protos.Type.INTEGER, genai.protos.Type.BOOLEAN]:
            return genai.protos.Schema(
                type_=genai_type,
                description=obj.get('description', None),
            )
        else:
            assert False, f"Unsupported type {genai_type}"

    def _tool_declare_convert(self, tool_declare):
        assert tool_declare['type'] == 'function', f"Unsupported tool type {tool_declare['type']}"
        raw_parameters = tool_declare['function']['parameters']
        assert raw_parameters['type'] == 'object', f"Unsupported parameters type {raw_parameters['type']}"

        parameters_schema = self._tool_declare_convert_obj(raw_parameters)
        function_declaration = genai.protos.FunctionDeclaration(
            name=tool_declare['function']['name'],
            description=tool_declare['function'].get('description', None),
            parameters=parameters_schema,
        )
        return function_declaration

    async def chat_stream_async(self, model_name, history, model_param, client_param):
        model_param = model_param.copy()
        temperature = model_param['temperature']
        use_google_search = client_param.get('use_google_search', False)
        force_calc_token_num = client_param.get('force_calc_token_num', False)
        input_tools = model_param.get('tools', None)

        system_message_list = [m for m in history if m['role'] == 'system']

        system_instruction = system_message_list[0]['content'] if system_message_list else None
        model = genai.GenerativeModel(
            model_name,
            system_instruction=system_instruction
        )
        generation_config = genai.types.GenerationConfig(
            temperature=temperature)
        messages = [{
            'role': self.role_convert_from_openai(m['role']),
            'parts': [m['content']]
        } for m in history
            if m['role'] != 'system'
        ]

        start_time = time.time()

        tools = []
        tool_config = None
        if use_google_search:
            tool = Tool.from_google_search_retrieval()
            tools.append(tool)

        if input_tools:
            for tool_declare in input_tools:
                function_declaration = self._tool_declare_convert(tool_declare)
                tools.append(genai.protos.Tool(
                    function_declarations=[function_declaration],
                ))
        if tools:
            tool_config = {
                'function_calling_config': {
                    'mode': 'ANY',
                },
            }

        response = model.generate_content_async(
            messages,
            generation_config=generation_config,
            safety_settings=self.get_safety_settings(),
            tools=tools,
            tool_config=tool_config,
            stream=True)

        role = None
        result_buffer = ''
        finish_reason = None
        first_token_time = None
        usage = None
        function_call_info_list = []

        async for chunk_message in await response:
            chunk = type(chunk_message).to_dict(chunk_message)
            if chunk['candidates']:
                # finish_reason = chunk['candidates'][0]['finish_reason']  # int
                finish_reason = chunk_message.candidates[0].finish_reason.name
                delta_info = chunk['candidates'][0]

                parts = delta_info['content']['parts']
                for part in parts:
                    if part.get('text'):
                        result_buffer += part['text']
                    if first_token_time is None:
                        first_token_time = time.time()
                    
                    if part.get('function_call'):
                        function_call_info = part['function_call']
                        call_info = LlmToolCallInfo(
                            tool_call_id=None,
                            tool_name=function_call_info['name'],
                            tool_args_json=json.dumps(function_call_info['args'], ensure_ascii=False),
                        )
                        function_call_info_list.append(call_info)


                    role = self.role_convert_to_openai(delta_info['content']['role'])
                    yield LlmResponseChunk(
                        role=role,
                        delta_content=part.get('text', ''),
                        accumulated_content=result_buffer,
                    )
                if delta_info.get('usage_metadata'):
                    usage = {
                        'prompt_tokens': delta_info['usage_metadata']['prompt_token_count'],
                        'completion_tokens': delta_info['usage_metadata']['candidates_token_count'],
                        'cached_tokens': delta_info['usage_metadata']['cached_content_token_count'],
                    }

        completion_time = time.time()

        if usage is None and force_calc_token_num:
            prompt_token_num = model.count_tokens(messages).total_tokens
            completion_token_num = 0
            if result_buffer:
                completion_token_num = model.count_tokens(result_buffer).total_tokens
            usage = {
                'prompt_tokens': prompt_token_num,
                'completion_tokens': completion_token_num,
            }

        yield LlmResponseTotal(
            role=role,
            accumulated_content=result_buffer,
            tool_calls=function_call_info_list,
            finish_reason=finish_reason,
            usage=usage,
            first_token_time=first_token_time - start_time if first_token_time else completion_time - start_time,
            completion_time=completion_time - start_time,
        )


    async def chat_async(self, model_name, history, model_param, client_param):
        model_param = model_param.copy()
        temperature = model_param['temperature']
        use_google_search = client_param.get('use_google_search', False)
        force_calc_token_num = client_param.get('force_calc_token_num', False)
        input_tools = model_param.get('tools', None)

        system_message_list = [m for m in history if isinstance(m, dict) and m['role'] == 'system']

        system_instruction = system_message_list[0]['content'] if system_message_list else None
        model = genai.GenerativeModel(
            model_name,
            system_instruction=system_instruction
        )
        generation_config = genai.types.GenerationConfig(
            temperature=temperature)
        
        message_list = []
        for m in history:
            if isinstance(m, dict) and m['role'] == 'system':
                continue
            
            if type(m).__name__ == 'Content':
                message_list.append(m)
                continue
            
            role = self.role_convert_from_openai(m['role'])
            if isinstance(m['content'], str):
                message = ContentDict(parts=[to_part(m['content'])], role=role)
            elif m['role'] == 'tool':
                parts = [
                    to_part({
                        "function_response": {
                            'name': tool_result['tool_name'],
                            'response': {
                                'content': tool_result['content'],
                            }
                        }
                    })
                    for tool_result in m['content']
                ]
                message = ContentDict(parts=parts, role=role)
            else:
                raise ValueError(f'unknown content type: {type(m["content"]).__name__}')
            
            message_list.append(message)
        
        start_time = time.time()

        tools = []
        tool_config = None
        if use_google_search:
            tool = Tool.from_google_search_retrieval()
            tools.append(tool)

        if input_tools:
            for tool_declare in input_tools:
                function_declaration = self._tool_declare_convert(tool_declare)
                tools.append(genai.protos.Tool(
                    function_declarations=[function_declaration],
                ))

        response_message = await model.generate_content_async(
            message_list,
            generation_config=generation_config,
            safety_settings=self.get_safety_settings(),
            tools=tools,
            tool_config=tool_config,
            stream=False
        )
        
        response = type(response_message).to_dict(response_message)
        # print(f'response: {response}')

        role = None
        result_buffer = ''
        finish_reason = None
        first_token_time = None
        usage = None
        function_call_info_list = []

        assert len(response['candidates']) > 0, f'No candidates in response'
        if 'content' not in response['candidates'][0]:
            raise Exception(f'No content in response, finish reason: {response_message.candidates[0].finish_reason.name}')

        # break on first tool call
        if self.auto_break_on_first_tool_call:
            first_tool_call_part_idx = None
            for part_idx, part in enumerate(response['candidates'][0]['content']['parts']):
                if part.get('function_call'):
                    first_tool_call_part_idx = part_idx
                    break
            if first_tool_call_part_idx is not None and first_tool_call_part_idx < len(response['candidates'][0]['content']['parts']) - 1:
                drop_num = len(response['candidates'][0]['content']['parts']) - first_tool_call_part_idx
                print(f'drop {drop_num} parts after first tool call')
                response['candidates'][0]['content']['parts'] = response['candidates'][0]['content']['parts'][:first_tool_call_part_idx+1]
                response_message.candidates[0].content.parts = response_message.candidates[0].content.parts[:first_tool_call_part_idx+1]

        raw_response_message = response_message.candidates[0].content
        raw_response_message_obj = response['candidates'][0]['content']

        # finish_reason = chunk['candidates'][0]['finish_reason']  # int
        finish_reason = response_message.candidates[0].finish_reason.name
        message_info = response['candidates'][0]

        parts = message_info['content']['parts']
        for part in parts:
            if part.get('text'):
                result_buffer += part['text']
            if first_token_time is None:
                first_token_time = time.time()
            
            if part.get('function_call'):
                function_call_info = part['function_call']
                call_info = LlmToolCallInfo(
                    tool_call_id=None,
                    tool_name=function_call_info['name'],
                    tool_args_json=json.dumps(function_call_info['args'], ensure_ascii=False),
                )
                function_call_info_list.append(call_info)


            role = self.role_convert_to_openai(message_info['content']['role'])

        if response.get('usage_metadata'):
            usage = {
                'prompt_tokens': response['usage_metadata']['prompt_token_count'],
                'completion_tokens': response['usage_metadata']['candidates_token_count'],
                'cached_tokens': response['usage_metadata']['cached_content_token_count'],
            }

        completion_time = time.time()

        if usage is None and force_calc_token_num:
            prompt_token_num = model.count_tokens(message_list).total_tokens
            completion_token_num = 0
            if result_buffer:
                completion_token_num = model.count_tokens(result_buffer).total_tokens
            usage = {
                'prompt_tokens': prompt_token_num,
                'completion_tokens': completion_token_num,
            }

        return LlmResponseTotal(
            role=role,
            accumulated_content=result_buffer,
            tool_calls=function_call_info_list,
            raw_response_message=raw_response_message,
            raw_response_message_obj=raw_response_message_obj,
            finish_reason=finish_reason,
            usage=usage,
            first_token_time=first_token_time - start_time if first_token_time else completion_time - start_time,
            completion_time=completion_time - start_time,
        )


    def convert_multimodal_message(self, content: list, file_handle_list:list):
        assert MultimodalMessageUtils.check_content_valid(content), f"Invalid content {content}"

        content_part_list = []
        for content_part in content:
            if isinstance(content_part, MultimodalMessageContentPart_Text):
                content_part_list.append(content_part.text)
            elif isinstance(content_part, MultimodalMessageContentPart_ImageUrl):
                assert False, "Not support ImageUrl"
            elif isinstance(content_part, MultimodalMessageContentPart_ImagePath):
                image = ImageFile.load_from_path(content_part.image_path)
                assert image.mime_type in ['image/jpeg', 'image/png', 'image/webp', 'image/heic', 'image/heif'], f"Unsupported image format {image.mime_type}"
                pil_image = PIL.Image.open(content_part.image_path)
                content_part_list.append(pil_image)
            elif isinstance(content_part, MultimodalMessageContentPart_AudioPath):
                # https://ai.google.dev/gemini-api/docs/audio?lang=python
                ext = os.path.splitext(content_part.audio_path)[1][1:]
                assert ext in ['mp3', 'wav', 'flac', 'ogg', 'aiff', 'aac'], f"Unsupported audio format {ext}"
                file = genai.upload_file(content_part.audio_path)
                content_part_list.append(file)
                file_handle_list.append(file)
            elif isinstance(content_part, MultimodalMessageContentPart_VideoPath):
                # https://ai.google.dev/gemini-api/docs/vision?lang=python#prompting-video
                ext = os.path.splitext(content_part.video_path)[1][1:]
                assert ext in ['mp4', 'mpeg', 'mov', 'avi', 'mkv', 'webm', 'flv'], f"Unsupported video format {ext}"
                file = genai.upload_file(content_part.video_path)
                content_part_list.append(file)
                file_handle_list.append(file)

        return content_part_list

    async def multimodal_chat_stream_async(self, model_name, history: List, model_param, client_param):
        model_param = model_param.copy()
        temperature = model_param['temperature']
        force_calc_token_num = client_param.get('force_calc_token_num', False)

        system_message_list = [m for m in history if m['role'] == 'system']

        system_instruction = system_message_list[0]['content'] if system_message_list else None
        model = genai.GenerativeModel(model_name, system_instruction=system_instruction)
        generation_config = genai.types.GenerationConfig(
            temperature=temperature)


        message_list = []
        file_handle_list = []
        for message in history:
            if message['role'] == 'system':
                continue

            if isinstance(message['content'], str):
                message_list.append({
                    'role': self.role_convert_from_openai(message['role']),
                    'parts': [message['content']]
                })
            elif isinstance(message['content'], list):
                message_list.append({
                    'role': self.role_convert_from_openai(message['role']),
                    'parts': self.convert_multimodal_message(message['content'], file_handle_list),
                })

        if len(file_handle_list) > 0:
            print(f'waiting for uploaded files to be processed ...')
            for file in file_handle_list:
                while True:
                    file_info = genai.get_file(file.name)

                    if file_info.state == protos.File.State.ACTIVE:
                        break

                    assert file_info.state != protos.File.State.FAILED, f'File {file.name} gemini process failed'

                    await asyncio.sleep(1)
            print(f'uploaded files processed')

        start_time = time.time()

        response = model.generate_content_async(
            message_list,
            generation_config=generation_config,
            safety_settings=self.get_safety_settings(),
            stream=True
        )

        role = None
        result_buffer = ''
        finish_reason = None
        first_token_time = None

        async for chunk in await response:
            # print(chunk)
            if chunk.candidates:
                finish_reason = chunk.candidates[0].finish_reason.name
                delta_info = chunk.candidates[0]
                if delta_info.content.parts:
                    result_buffer += delta_info.content.parts[0].text
                    if first_token_time is None:
                        first_token_time = time.time()

                    role = self.role_convert_to_openai(delta_info.content.role)
                    yield LlmResponseChunk(
                        role=role,
                        delta_content=delta_info.content.parts[0].text,
                        accumulated_content=result_buffer,
                    )

        completion_time = time.time()

        usage = None
        if force_calc_token_num:
            prompt_token_num = model.count_tokens(message_list).total_tokens
            completion_token_num = 0
            if result_buffer:
                completion_token_num = model.count_tokens(result_buffer).total_tokens
            usage = {
                'prompt_tokens': prompt_token_num,
                'completion_tokens': completion_token_num,
            }

        for file in file_handle_list:
            try:
                genai.delete_file(file.name)
            except Exception as e:
                print(f'Error deleting file {file.name}: {e}')

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

    os.environ['HTTP_PROXY'] = "http://127.0.0.1:7890/"
    os.environ['HTTPS_PROXY'] = "http://127.0.0.1:7890/"

    client = Gemini_Client()
    model_name = "gemini-1.5-pro-latest"
    history = [
        # {"role": "system", "content": "You are an assistant for home cooks. "},
        {"role": "user", "content": "爱因斯坦的生日是哪一天？"},
    ]

    model_param = {
        'temperature': 0.01,
        'use_google_search': True,
    }

    async def stream_main():
        async for chunk in client.chat_stream_async(model_name, history, model_param, client_param={}):
            print(chunk)

    async def nostream_main():
        total_chunk = await client.chat_async(model_name, history, model_param, client_param={})
        print(total_chunk)

    asyncio.run(nostream_main())
