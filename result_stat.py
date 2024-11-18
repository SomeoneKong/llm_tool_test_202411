
import os
import json
import collections
import traceback
from pydantic import BaseModel
from typing import Optional

sample_data_dir = 'test_data_topic_split'

result_root_dir = 'test_data_topic_split_output_temp001'

def stat_sensitive_data():
    
    sample_data_name_list = []
    for file_name in os.listdir(sample_data_dir):
        base_name = file_name.split('.')[0]
        sample_data_name_list.append(base_name)


    sensitive_data_stat = collections.defaultdict(set)
    for model_name in os.listdir(result_root_dir):
        model_path = os.path.join(result_root_dir, model_name)
        if not os.path.isdir(model_path):
            continue

        # Remove empty directory
        list_dir = os.listdir(model_path)
        if not list_dir:
            print(f'Remove empty directory: {model_path}')
            os.rmdir(model_path)
            continue
        elif len(list_dir) < 100:
            print(f'{model_name} has only {len(list_dir)} files')
        
        for file_name in os.listdir(model_path):
            if file_name.endswith('.json'):
                base_name = file_name.split('.')[0]
                sample_name = base_name.rsplit('_', 1)[0]

                if sample_name not in sample_data_name_list:
                    os.remove(os.path.join(model_path, file_name))
                    print(f'Remove extra file: {os.path.join(model_path, file_name)}')
                    continue

                with open(os.path.join(model_path, file_name), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if data['ex_type'] == 'SensitiveBlockError':
                    sensitive_data_stat[sample_name].add(model_name)

    for sample_name, model_list in sensitive_data_stat.items():
        print(f'{sample_name}: {model_list}')


class ToolCallFunction(BaseModel):
    name: str
    arguments: str

class ToolCall(BaseModel):
    id: Optional[str]
    type: str
    function: Optional[ToolCallFunction]


def check_args(args_str, sample_data):

    block_token_dict = {}
    for block in sample_data['block_list']:
        idx = block['block_idx']
        token_num = block['token_num']
        block_token_dict[idx] = token_num
    
    try:
        args_obj = json.loads(args_str)
        topic_list = args_obj['topic_list']
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
    except Exception as e:
        traceback.print_exc()
        return False
    return True


def stat_one_model(result_root_dir):
    result_file_list = os.listdir(result_root_dir)
    
    is_claude = 'claude' in result_root_dir
    is_gemini = 'gemini' in result_root_dir
    
    result_type_stat = collections.defaultdict(int)
    sensitive_counter = 0
    for file_name in result_file_list:
        base_name = file_name.split('.')[0]
        sample_name = base_name.rsplit('_', 1)[0]
        
        sample_data = json.load(open(os.path.join(sample_data_dir, f'{sample_name}.json'), 'r', encoding='utf-8'))
        
        
        result_obj = json.load(open(os.path.join(result_root_dir, file_name), 'r', encoding='utf-8'))
        if result_obj['ex_type'] == 'SensitiveBlockError':
            sensitive_counter += 1
        if result_obj['ex_type'] == 'BadToolCallArgsError':
            for tool_call in result_obj['message_list'][1]['tool_calls']:
                if tool_call['function']['name'] != 'calc_topic_token_batch':
                    print(f'tool call name error in {file_name}: {tool_call["function"]["name"]}')
                check_args(tool_call["function"]['arguments'], sample_data)
        if result_obj['ex_type'] is not None:
            result_type_stat[result_obj['ex_type']] += 1
            print(f'{file_name} has error: {result_obj["ex_type"]}')
            continue
        
        # check no pre thinking
        if is_claude:
            answer1 = result_obj['message_list'][1]['content'][0]['text'] or ''
        elif is_gemini:
            answer1 = result_obj['message_list'][1]['parts'][0].get('text') or ''
        else:
            answer1 = result_obj['message_list'][1]['content'] or ''
            
        if len(answer1.split('\n')) <= 1:
            print(f'no pre thinking in {file_name}')
            result_type_stat['no_pre_thinking'] += 1
            continue
        
        tool_call_list = []
        if is_claude:
            for content_block in result_obj['message_list'][1]['content'][1:]:
                if content_block['type'] == 'tool_use':
                    tool_call = ToolCall(
                        id=content_block['id'],
                        type='function',
                        function=ToolCallFunction(
                            name=content_block['name'],
                            arguments=json.dumps(content_block['input'], ensure_ascii=False),
                        )
                    )
                    tool_call_list.append(tool_call)
        elif is_gemini:
            for part in result_obj['message_list'][1]['parts']:
                if 'function_call' in part:
                    tool_call = ToolCall(
                        id=None,
                        type='function',
                        function=ToolCallFunction(
                            name=part['function_call']['name'],
                            arguments=json.dumps(part['function_call']['args'], ensure_ascii=False),
                        )
                    )
                    tool_call_list.append(tool_call)
        else:
            # check tool call topic uniqueness
            tool_call_raw_list = result_obj['message_list'][1]['tool_calls']
            for tool_call_raw in tool_call_raw_list:
                try:
                    tool_call = ToolCall.model_validate(tool_call_raw)
                    tool_call_list.append(tool_call)
                except Exception as e:
                    print(f'tool call error in {file_name}: {e}')
                    traceback.print_exc()
                    result_type_stat['__tool_call_structure_error'] += 1
                    continue
            
        topic_counter_stat = collections.defaultdict(int)
        for tool_call in tool_call_list:
            args = json.loads(tool_call.function.arguments)
            topic_obj = args['topic_list']
            topic_key = json.dumps(topic_obj, ensure_ascii=False, sort_keys=True)
            topic_counter_stat[topic_key] += 1
        
        for topic_key, topic_counter in topic_counter_stat.items():
            if topic_counter > 1:
                result_type_stat['topic_repeat'] += 1
                break
        
        
    success_num = len(result_file_list) - sum(result_type_stat.values())
    print(f'{result_root_dir=}')
    print(f'total_num: {len(result_file_list)}')
    print(f'sensitive_num: {sensitive_counter}')
    print(f'success_num: {success_num}')
    print(dict(result_type_stat))
    


if __name__ == '__main__':
    import sys
    # stat_sensitive_data()
    stat_one_model(sys.argv[1])
