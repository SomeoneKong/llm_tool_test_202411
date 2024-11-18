

import os
import time
import json

import sys


if __name__ == '__main__':
    base_path = os.path.join(os.path.dirname(__file__), '..')
    sys.path.insert(0, base_path)


from llm_client_base import *
from typing import List

# pip install google-cloud-aiplatform

# config from .env
# GOOGLE_APPLICATION_CREDENTIALS

import vertexai
from vertexai.generative_models import (
    GenerationConfig,
    GenerativeModel,
    FunctionDeclaration,
    Part,
    Tool,
    Content,
    ChatSession,
    grounding,
    HarmCategory,
    HarmBlockThreshold,
)
from google.oauth2 import service_account

class GoogleSearchGroundingChunk(BaseModel):
    url: str
    title: str

class GoogleSearchGroundingSupport(BaseModel):
    confidence_scores: List[float]
    grounding_chunk_indices: List[int]
    segment_text: str
    segment_start_index: int
    segment_end_index: int

class GoogleSearchGrounding(BaseModel):
    web_search_queries: List[str]
    rendered_content: str
    grounding_chunk_list: List[GoogleSearchGroundingChunk]
    grounding_support_list: List[GoogleSearchGroundingSupport]
    google_search_dynamic_retrieval_score: Optional[float]

class GoogleVertexAI_Client(LlmClientBase):
    support_system_message: bool = True
    # support_image_message: bool = True

    def __init__(self, auth_json_file=None):
        super().__init__()
        
        # 使用json文件凭据
        auth_json_file = auth_json_file or os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        assert auth_json_file, 'Please provide auth_json_file'

        with open(auth_json_file, "r") as json_file:
            json_data = json.load(json_file)
        
        project_id = json_data["project_id"]
        credentials = service_account.Credentials.from_service_account_file(auth_json_file)

        vertexai.init(project=project_id, location="us-central1", credentials=credentials)

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
            raise ValueError(f'unknown role: {role}')

    def get_safety_settings(self):
        return {
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

    async def chat_stream_async(self, model_name, history, model_param, client_param):
        model_param = model_param.copy()
        temperature = model_param.get('temperature', None)
        use_google_search = model_param.get('use_google_search', False)
        force_calc_token_num = client_param.get('force_calc_token_num', False)
        input_tools = model_param.get('tools', None)

        system_message_list = [m for m in history if isinstance(m, dict) and m['role'] == 'system']
        system_instruction = system_message_list[0]['content'] if system_message_list else None

        generation_config = GenerationConfig(
            temperature=temperature
        )
        
        message_list = []
        for m in history:
            if isinstance(m, dict) and m['role'] == 'system':
                continue

            if isinstance(m, Content):
                message_list.append(m)
            else:
                role = self.role_convert_from_openai(m['role'])
                if isinstance(m['content'], str):
                    message = Content(parts=[Part.from_text(m['content'])], role=role)
                elif m['role'] == 'tool':
                    parts = [
                        Part.from_function_response(
                            name=tool_result['tool_name'],
                            response={
                                'content': tool_result['content'],
                            }
                        )
                        for tool_result in m['content']
                    ]
                    message = Content(parts=parts, role=role)
                else:
                    raise ValueError(f'unknown content type: {type(m["content"]).__name__}')
                
                message_list.append(message)
            # print(f'message: {message_list[-1]}')
        
        tools = []
        if use_google_search:
            # https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/ground-gemini?hl=zh-cn
            tool = Tool.from_google_search_retrieval(grounding.GoogleSearchRetrieval())
            tools.append(tool)
        if input_tools:
            function_declarations = []
            for tool in input_tools:
                function_declarations.append(FunctionDeclaration(
                    name=tool['function']['name'],
                    description=tool['function']['description'],
                    parameters=tool['function']['parameters'],
                ))
            tools.append(Tool.from_function_declarations(function_declarations))
        if len(tools) == 0:
            tools = None

        start_time = time.time()

        model = GenerativeModel(
            model_name,
            system_instruction=system_instruction,
        )

        response = model.generate_content_async(
            message_list,
            tools=tools,
            generation_config=generation_config,
            safety_settings=self.get_safety_settings(),
            stream=True
        )

        role = None
        result_buffer = ''
        finish_reason = None
        first_token_time = None
        grounding_results = []
        function_call_info_list = []

        async for chunk in await response:
            # print(chunk)
            if chunk.candidates:
                finish_reason = chunk.candidates[0].finish_reason.name
                delta_info = chunk.candidates[0]
                if delta_info.grounding_metadata:
                    # print(delta_info.grounding_metadata)
                    grounding_chunk_list = []
                    for chunk in delta_info.grounding_metadata.grounding_chunks:
                        grounding_chunk_list.append(GoogleSearchGroundingChunk(
                            url=chunk.web.uri,
                            title=chunk.web.title
                        ))
                    grounding_support_list = []
                    for support in delta_info.grounding_metadata.grounding_supports:
                        grounding_support_list.append(GoogleSearchGroundingSupport(
                            confidence_scores=support.confidence_scores,
                            grounding_chunk_indices=support.grounding_chunk_indices,
                            segment_text=support.segment.text,
                            segment_start_index=support.segment.start_index,
                            segment_end_index=support.segment.end_index
                        ))
                    grounding_result = GoogleSearchGrounding(
                        web_search_queries=list(delta_info.grounding_metadata.web_search_queries),
                        rendered_content=delta_info.grounding_metadata.search_entry_point.rendered_content,
                        grounding_chunk_list=grounding_chunk_list,
                        grounding_support_list=grounding_support_list,
                        google_search_dynamic_retrieval_score=delta_info.grounding_metadata.retrieval_metadata.google_search_dynamic_retrieval_score
                    )

                    grounding_results.append(grounding_result)
                if delta_info.content.parts:
                    delta_content = ''
                    for part in delta_info.content.parts:
                        # print(f'part: {part.to_dict()}')
                        part_obj = part.to_dict()
                        if 'text' in part_obj:
                            result_buffer += part_obj['text']
                            delta_content += part_obj['text']
                        elif 'function_call' in part_obj:
                            function_call_info = part_obj['function_call']
                            call_info = LlmToolCallInfo(
                                tool_call_id=None,
                                tool_name=function_call_info['name'],
                                tool_args_json=json.dumps(function_call_info['args'], ensure_ascii=False),
                            )
                            function_call_info_list.append({
                                'call_info': call_info,
                                'raw_part': part
                            })
                        if first_token_time is None:
                            first_token_time = time.time()

                    role = self.role_convert_to_openai(delta_info.content.role)
                    yield LlmResponseChunk(
                        role=role,
                        delta_content=delta_content,
                        accumulated_content=result_buffer,
                    )

        # print(f'History==============\n{chat_session.history}\n==============')
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
        

        # raw_response_message
        raw_response_message = Content(parts=[
            Part.from_text(result_buffer)
        ], role=role)
        if function_call_info_list:
            function_call_result_list = [
                    call_info['raw_part']
                    for call_info in function_call_info_list
                ]
            raw_response_message = Content(
                role=role,
                parts=[
                    raw_response_message.parts[0]
                ] + function_call_result_list
            )

        # raw_response_message_obj
        raw_response_message_obj = raw_response_message.to_dict()
        # print(f'raw_response_message_obj: {raw_response_message_obj}')


        yield LlmResponseTotal(
            role=role,
            accumulated_content=result_buffer,
            tool_calls=[call_info['call_info'] for call_info in function_call_info_list],
            raw_response_message=raw_response_message,
            raw_response_message_obj=raw_response_message_obj,
            finish_reason=finish_reason,
            usage=usage,
            first_token_time=first_token_time - start_time if first_token_time else completion_time - start_time,
            completion_time=completion_time - start_time,
            extra={
                'google_search_results': grounding_results
            }
        )
    
    async def chat_async(self, model_name, history, model_param, client_param):
        async for chunk in self.chat_stream_async(model_name, history, model_param, client_param):
            total_chunk = chunk
        return total_chunk
        


if __name__ == '__main__':
    import asyncio
    import os
    os.environ['HTTP_PROXY'] = "http://127.0.0.1:7890/"
    os.environ['HTTPS_PROXY'] = "http://127.0.0.1:7890/"

    client = GoogleVertexAI_Client()
    model_name = "gemini-1.5-pro-002"

    prompt = '''
使用 英文 搜索 `TTFT`
并从结果中找出与以下描述最接近的概念：Time To First Token，指生成第一个token所需的时间

给出该概念的全名，并给出它与上述搜索关键词最接近的写法

以如下格式输出：
{
  "found": bool,
  "name": str,
  "name_similar_to_origin": str,
  "description": str
}
'''

    history = [
        # {"role": "system", "content": "You are an assistant for home cooks. "},
        {"role": "user", "content": prompt},
    ]

    model_param = {
        'temperature': 0.0,
        'use_google_search': True,
    }
    client_param = {
        'force_calc_token_num': True,
    }

    async def main():
        async for chunk in client.chat_stream_async(model_name, history, model_param, client_param):
            print(chunk)
        print(chunk.accumulated_content)
        if chunk.extra.get('google_search_results'):
            for result in chunk.extra['google_search_results']:
                print(result.web_search_queries)
                print(result.grounding_chunk_list)

    asyncio.run(main())

