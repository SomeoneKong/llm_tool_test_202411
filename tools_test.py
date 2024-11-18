import os
from dotenv import load_dotenv
import asyncio
import json
import traceback
import math

from llm_client_base import *

from client_impl.anthropic_impl import Anthropic_Client
from client_impl.alibaba_impl import Alibaba_Client
from client_impl.deepseek_impl import DeepSeek_Client
from client_impl.yi_impl import Yi_Client
from client_impl.zhipu_impl import Zhipu_Client
from client_impl.google_impl import Gemini_Client
from client_impl.google_vertexai_impl import GoogleVertexAI_Client
from client_impl.google_openai_impl import GeminiOpenAI_Client
from client_impl.moonshot_impl import Moonshot_Client
from client_impl.stepfun_impl import StepFun_Client
from client_impl.baichuan_impl import Baichuan_Client
from client_impl.bytedance_impl import ByteDance_Client
from client_impl.baidu_impl import Baidu_Client
from client_impl.tencent_openai_impl import TencentOpenAI_Client
from client_impl.sensenova_openai_impl import SenseNovaOpenAI_Client
from client_impl.xunfei_impl import Xunfei_Client

from client_impl.openrouter_impl import OpenRouter_Client


class ErrorWithMessageList(Exception):
    def __init__(self, message, message_list):
        super().__init__(message)
        self.message_list = message_list


class NoToolCallError(ErrorWithMessageList):
    pass


class BadToolCallArgsError(ErrorWithMessageList):
    pass


class RequestError(Exception):
    pass


async def handle_stream(async_iter):
    async for chunk in async_iter:
        if not chunk.is_end:
            print(chunk.delta_content or '', end='')
        else:
            print()
    if chunk.finish_reason in ['sensitive']:
        raise SensitiveBlockError()
    return chunk


prompt_template = '''
以下是一个访谈录音的语音识别结果，需要将其按讨论的具体内容分割为多个小节。

语音识别结果如下：
```text
{asr_text_buffer}
```


Step 1 总结整个输入文本每段的主题。
总结每个主题的内容，并指名其开始和结束位置。
每个主题的第一个block_idx 记为block_start_idx，最后一个block_idx记为block_last_idx。

输出格式为：
topic 1:
- block_start_idx: xxx
- description: <主题描述>
- block_last_idx: xxx
...

Step 1.1 逐个计算每个topic的token_num。
topic的token是topic包含的所有block的token_num之和。
要调用calc_topic_token_batch进行计算，以获取最准确的结果。

Step 2 优化topic的选择和分割位置。

优化主题分割的规则：
* 分析每个topic的语义完整性，尽量避免分割点不会打断完整的讨论。
* 不要把不相邻的topic合并在一起。
* 如果一个topic的token_num超过 {topic_token_num_limit} ，则要将这个topic拆分为多个topic。
* 对于调整后的topic，要重新计算token_num。
* 如果调整后的topic的token_num超过 {topic_token_num_limit}，则要继续思考如何调整topic的分割位置，例如将其切分为更小的topic。


Step 3 整理输出结果。
使用json格式进行输出，格式见下面的完整回答格式。


请以以下的过程进行回答，当遇到函数调用返回后要接续上一次的回答位置继续回答。
------
Step 1 总结整个输入文本每段的主题。
...

Step 1.1 计算每个topic的token_num。
要调用calc_topic_token_batch进行计算。
...

Step 2 优化topic的选择和分割位置。
...

Step 3 以json格式整理输出结果。
{{
    "topic_list": [
        {{
            "topic_description": str,
            "block_start_idx": int,
            "block_last_idx": int
        }},
        ...
    ]
}}
'''.strip()


async def test_one_run(
    client,
    model_name,
    test_data,
    temperature=0.01,
    max_tokens=None,
    debug_mode=False,
):
    asr_text_buffer = []
    block_token_dict = {}
    for block in test_data['block_list']:
        idx = block['block_idx']
        speaker_id = block['speaker_id']
        speaker_name = block['speaker_name']
        text = block['text']
        token_num = block['token_num']
        asr_text_buffer.append(f'Block {idx}, token_num={token_num}, \n{speaker_id} {speaker_name}\n{text}\n')
        block_token_dict[idx] = token_num

    start_block_idx = test_data['start_block_idx']
    last_block_idx = test_data['last_block_idx']
    topic_token_num_limit = 800

    prompt = prompt_template.format(
        asr_text_buffer='\n'.join(asr_text_buffer),
        topic_token_num_limit=topic_token_num_limit,
        start_block_idx=start_block_idx,
        last_block_idx=last_block_idx,
    )

    # print(prompt)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "calc_topic_token_batch",
                "description": "计算topic的token num",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topic_list": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "topic_id": {"type": "integer"},
                                    "start_block_idx": {"type": "integer"},
                                    "last_block_idx": {"type": "integer"},
                                },
                                "required": ["topic_id", "start_block_idx", "last_block_idx"]
                            }
                        }
                    },
                    "required": ["topic_list"]
                }
            }
        }
    ]

    model_param = {
        'temperature': temperature,
        'tools': tools,
        # 'tool_choice': "auto",
        # 'max_tokens': 4096,
    }
    if max_tokens:
        model_param['max_tokens'] = max_tokens

    # print(f'model_name: {model_name}')
    message_list = [
        {"role": "user", "content": prompt},
    ]
    try:
        total_chunk = await client.chat_async(model_name, message_list, model_param, client_param={})
        assert total_chunk.accumulated_content or total_chunk.tool_calls, f'{model_name} no content or tool call, {total_chunk}'
    except SensitiveBlockError as e:
        raise
    except Exception as e:
        traceback.print_exc()
        raise RequestError() from e
    # print(f'stop reason: {total_chunk.finish_reason}')
    # if debug_mode:
    #   print(f'total_chunk: {total_chunk.raw_response_message_obj or total_chunk}')

    dump_message_list = message_list[:]
    if total_chunk.raw_response_message:
        raw_response_message = total_chunk.raw_response_message
        if debug_mode:
            print(f'raw_response_message: {total_chunk.raw_response_message_obj}')
        if isinstance(raw_response_message, BaseModel):
            message_list.append(raw_response_message.model_dump())
        else:
            message_list.append(raw_response_message)

        dump_message_list.append(total_chunk.raw_response_message_obj)

    tool_call_info_list = total_chunk.tool_calls
    if tool_call_info_list is None or len(tool_call_info_list) == 0:
        raise NoToolCallError('no tool call', message_list)

    tool_call_result_list = []
    try:
        for tool_call in tool_call_info_list:

            # print(f'tool_call: {tool_call}')

            if tool_call.tool_name != 'calc_topic_token_batch':
                raise BadToolCallArgsError(f'tool name error, got {tool_call.tool_name}', message_list)

            arguments = json.loads(tool_call.tool_args_json)
            topic_list = arguments['topic_list']
            result_list = []
            for topic in topic_list:
                topic_id = int(topic['topic_id'])
                start_block_idx = int(topic['start_block_idx'])
                last_block_idx = int(topic['last_block_idx'])
                token_num = 0
                for block_idx in range(start_block_idx, last_block_idx + 1):
                    token_num += block_token_dict[block_idx]
                result_list.append({
                    'topic_id': topic_id,
                    'token_num': token_num,
                })
            tool_call_result_list.append({
                'tool_call_id': tool_call.tool_call_id,
                'result': result_list,
            })
    except Exception as e:
        traceback.print_exc()
        raise BadToolCallArgsError(str(e), message_list) from e

    if 'claude' in model_name:
        content_list = []
        for tool_call_result in tool_call_result_list:
            content_list.append({
                "type": "tool_result",
                'tool_use_id': tool_call_result['tool_call_id'],
                'content': json.dumps(tool_call_result['result'], ensure_ascii=False),
            })
        message_list.append({
            'role': 'user',
            'content': content_list
        })
        dump_message_list.append(message_list[-1])
    elif False and 'gemini' in model_name:
        content_list = []
        for tool_call_result in tool_call_result_list:
            content_list.append({
                "type": "tool_result",
                'tool_name': 'calc_topic_token_batch',
                'content': tool_call_result['result'],
            })
        message_list.append({
            'role': 'tool',
            'content': content_list
        })
        dump_message_list.append(message_list[-1])
    else:
        for tool_call_result in tool_call_result_list:
            message_list.append({
                'role': 'tool',
                'tool_call_id': tool_call_result['tool_call_id'],
                'content': json.dumps(tool_call_result['result'], ensure_ascii=False),
            })
            dump_message_list.append(message_list[-1])

    try:
        total_chunk = await client.chat_async(model_name, message_list, model_param, client_param={})
        assert total_chunk.accumulated_content or total_chunk.tool_calls, f'{model_name} no content or tool call, {total_chunk}'
    except SensitiveBlockError as e:
        raise
    except Exception as e:
        traceback.print_exc()
        raise RequestError() from e

    if total_chunk.raw_response_message_obj:
        raw_response_message = total_chunk.raw_response_message_obj
        dump_message_list.append(raw_response_message)

    return dump_message_list


async def test_main():
    client, model_name, output_dir_model_name = OpenRouter_Client(), "gpt-4o", 'openrouter-gpt-4o'
    # client, model_name, output_dir_model_name = OpenRouter_Client(), "gpt-4o-mini", 'openrouter-gpt-4o-mini'
    
    # client, model_name, output_dir_model_name = OpenRouter_Client(), "anthropic/claude-3.5-sonnet", 'openrouter-claude-3-5-sonnet'  # 会卡死在第二步不返回
    # client, model_name, output_dir_model_name = OpenRouter_Client(), "anthropic/claude-3-5-haiku", 'openrouter-claude-3-5-haiku'  # 会卡死在第二步不返回
    
    # client, model_name, output_dir_model_name = Anthropic_Client(), "claude-3-5-sonnet-20241022", 'anthropic-claude-3-5-sonnet-20241022'
    client, model_name, output_dir_model_name = Anthropic_Client(), "claude-3-5-haiku-20241022", 'anthropic-claude-3-5-haiku'
    
    # client, model_name, output_dir_model_name = GeminiOpenAI_Client(), "gemini-1.5-pro", 'gemini-1-5-pro-openai'
    # client, model_name, output_dir_model_name = Gemini_Client(), "gemini-1.5-pro", 'gemini-1-5-pro'
    # client, model_name, output_dir_model_name = GoogleVertexAI_Client(), "gemini-1.5-pro", 'gemini-1-5-pro'
    # client, model_name, output_dir_model_name = GoogleVertexAI_Client(), "gemini-1.5-flash", 'gemini-1-5-flash'

    # client, model_name, output_dir_model_name = Alibaba_Client(), "qwen2.5-72b-instruct", 'qwen2-5-72b'
    # client, model_name, output_dir_model_name = Alibaba_Client(), "qwen2.5-32b-instruct", 'qwen2-5-32b'
    # client, model_name, output_dir_model_name = Alibaba_Client(), "qwen2.5-14b-instruct", 'qwen2-5-14b'
    # client, model_name, output_dir_model_name = Alibaba_Client(), "qwen-max-2024-09-19", 'qwen-max-2024-09-19'
    # client, model_name, output_dir_model_name = Alibaba_Client(), "qwen-plus-2024-09-19", 'qwen-plus-2024-09-19'

    # client, model_name, output_dir_model_name = Zhipu_Client(), "glm-4-plus", 'glm-4-plus'
    # client, model_name, output_dir_model_name = Zhipu_Client(), "glm-4-0520", 'glm-4-0520'

    # client, model_name, output_dir_model_name = Yi_Client(), "yi-large-fc", 'yi-large-fc'

    # client, model_name, output_dir_model_name = DeepSeek_Client(), "deepseek-chat", 'deepseek-chat'

    # client, model_name, output_dir_model_name = Moonshot_Client(), "moonshot-v1-8k", 'moonshot-v1-8k'
    # client, model_name, output_dir_model_name = Moonshot_Client(), "moonshot-v1-32k", 'moonshot-v1-32k'

    # client, model_name, output_dir_model_name = ByteDance_Client(), os.environ['DOUBAO_PRO_FUNCTIONCALL_ENDPOINT'], 'Doubao-pro-32k-functioncall-241028'

    # client, model_name, output_dir_model_name = StepFun_Client(), "step-1-8k", 'step-1-8k'
    # client, model_name, output_dir_model_name = StepFun_Client(), "step-2-16k", 'step-2-16k'

    # client, model_name, output_dir_model_name = Baichuan_Client(), "Baichuan4-Turbo", "Baichuan4-Turbo"

    # client, model_name, output_dir_model_name = Baidu_Client(), "ernie-3.5-128k", 'ernie-3.5-128k'

    # client, model_name, output_dir_model_name = TencentOpenAI_Client(), "hunyuan-functioncall", 'hunyuan-functioncall'

    # client, model_name, output_dir_model_name = SenseNovaOpenAI_Client(), "SenseChat-5", 'SenseChat-5'

    # client, model_name, output_dir_model_name = Xunfei_Client(), "spark-max", 'spark-max'
    # client, model_name, output_dir_model_name = Xunfei_Client(), "spark-4.0", 'spark-4.0'

    parallel_num = 5

    sample_num = 1

    output_dir = f'test_data_topic_split_output_temp001/{output_dir_model_name}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    test_data_list = os.listdir('test_data_topic_split')
    test_data_list.sort()
    # test_data_list = test_data_list[1:]

    async def run_one_test(test_idx, test_data_filename, sample_idx):
        file_basename = test_data_filename.split('.')[0]
        test_data = json.load(open(f'test_data_topic_split/{test_data_filename}', 'r', encoding='utf-8'))
        log_filename = f'{output_dir}/{file_basename}_{sample_idx}.json'
        if os.path.exists(log_filename):
            print(f'{test_idx}/{len(test_data_list)}: {log_filename} exists')
            return

        ex_type = None
        ex_message = None
        message_list = None
        # print(f'{model_name}, {test_idx}/{len(test_data_list)} {sample_idx}: start')
        try:
            message_list = await test_one_run(client, model_name, test_data, debug_mode=parallel_num == 1)
            print(f'{file_basename} {test_idx}/{len(test_data_list)} {sample_idx}: success')
        except RequestError as e:
            print(f'{file_basename} {test_idx}/{len(test_data_list)} {sample_idx}: request error, skip. {str(e)}')
            return
        except Exception as e:
            if not isinstance(e, ErrorWithMessageList) and not isinstance(e, SensitiveBlockError):
                traceback.print_exc()
            print(f'{file_basename} {test_idx}/{len(test_data_list)} {sample_idx}: {type(e).__name__}')
            ex_type = type(e).__name__
            ex_message = str(e)
            if isinstance(e, ErrorWithMessageList):
                message_list = e.message_list

        log_data = {
            'test_idx': test_idx,
            'sample_idx': sample_idx,
            'success': ex_type is None,
            'ex_type': ex_type,
            'ex_message': ex_message,
            'message_list': message_list,
        }
        json.dump(log_data, open(log_filename, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

    job_list = []
    for test_idx, test_data_filename in enumerate(test_data_list):
        file_basename = test_data_filename.split('.')[0]
        for sample_idx in range(sample_num):
            log_filename = f'{output_dir}/{file_basename}_{sample_idx}.json'
            if os.path.exists(log_filename):
                print(f'{test_idx}/{len(test_data_list)}: {log_filename} exists')
                continue
            job = {
                'test_idx': test_idx,
                'test_data_filename': test_data_filename,
                'sample_idx': sample_idx,
            }
            job_list.append(job)

    for i in range(0, len(job_list), parallel_num):
        print(f'{model_name} batch {i//parallel_num}/{math.ceil(len(job_list)/parallel_num)} start')

        job_batch = job_list[i:i+parallel_num]
        batch_task_list = [run_one_test(job['test_idx'], job['test_data_filename'], job['sample_idx']) for job in job_batch]
        await asyncio.gather(*batch_task_list)
        print(f'batch {i//parallel_num}/{math.ceil(len(job_list)/parallel_num)} done')
        print('===============')

        if parallel_num == 1:
            # break
            pass


# Load environment variables from .env file
load_dotenv()

# os.environ['HTTP_PROXY'] = "http://127.0.0.1:7890/"
# os.environ['HTTPS_PROXY'] = "http://127.0.0.1:7890/"

asyncio.run(test_main())
