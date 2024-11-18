
import os
import json
import collections

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


if __name__ == '__main__':
    stat_sensitive_data()

