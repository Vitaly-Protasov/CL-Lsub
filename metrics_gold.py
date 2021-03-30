from tqdm import tqdm
from experiment import CL_substitution


def get_best_results(dataset, gold_file_path, candidates, eng_embed, esp_embed, folder_results):
    files_paths = [
                   f'{folder_results}ADD.best',
                   f'{folder_results}BalADD.best',
                   f'{folder_results}MUL.best',
                   f'{folder_results}BalMUL.best',
                   ]
    words_threshold = 1
    delimeter = ' :: '
    # for each_metric in files_names:
    f_gold = open(gold_file_path, 'r')
    for i, instance_string in enumerate(tqdm(f_gold)):
        instance_word_info_for_result = instance_string.split(delimeter)[0]
        target_word = dataset.iloc[i, 0]
        sentence = dataset.iloc[i, 1]
        model_instance_result = CL_substitution(sentence, target_word, eng_embed, esp_embed, candidates, words_threshold)
        for metric_result, each_path in zip(model_instance_result, files_paths):
            f_test_result = open(each_path, 'a+')
            top_sub = list(metric_result.keys())[0]
            
            string_to_write = instance_word_info_for_result + delimeter + top_sub + '\n'
            f_test_result.write(string_to_write)
            f_test_result.close()


def get_oot_results(dataset, gold_file_path, candidates, eng_embed, esp_embed, folder_results):
    files_paths = [
                   f'{folder_results}ADD.oot',
                   f'{folder_results}BalADD.oot',
                   f'{folder_results}MUL.oot',
                   f'{folder_results}BalMUL.oot',
                   ]
    words_threshold = 10
    f_gold = open(gold_file_path)
    
    # for each_metric in files_names:
    f_gold = open(gold_file_path, 'r')
    for i, instance_string in enumerate(tqdm(f_gold)):
        instance_word_info_for_result = instance_string.split(' :: ')[0]
        target_word = dataset.iloc[i, 0]
        sentence = dataset.iloc[i, 1]
        model_instance_result = CL_substitution(sentence, target_word, eng_embed, esp_embed, candidates, words_threshold)
        for metric_result, each_path in zip(model_instance_result, files_paths):
            f_test_result = open(each_path, 'a+')
            top_subs = ';'.join(list(metric_result.keys())[:words_threshold]) + ';'
            
            string_to_write = instance_word_info_for_result + ' ::: ' + top_subs + '\n'
            f_test_result.write(string_to_write)
            f_test_result.close()  