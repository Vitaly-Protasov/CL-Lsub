from tqdm import tqdm
from experiment import CL_substitution
from typing import Dict, Tuple
import pathlib


def oot_files_results_add(tw_pos: str, model_results: Dict[str, float], folder_results: str) -> None:
    delimeter_oot = ' ::: '
    oot_paths = [
                    pathlib.PurePath(folder_results, 'ADD.oot'),
                    pathlib.PurePath(folder_results, 'BalADD.oot'),
                    pathlib.PurePath(folder_results, 'MUL.oot'),
                    pathlib.PurePath(folder_results, 'BalMUL.oot')
                ]
    for metric_result, each_path in zip(model_results, oot_paths):
        f_test_result = open(each_path, 'a+')
        list_candidates = list(metric_result.keys())
        top_subs = ';'.join(list_candidates) + ';'
        string_to_write = tw_pos + delimeter_oot + top_subs + '\n'
        f_test_result.write(string_to_write)
        f_test_result.close()


def best_files_results_add(tw_pos: str, model_results: Dict[str, float], folder_results: str) -> None:
    delimeter_best = ' :: '
    best_paths = [
                    pathlib.PurePath(folder_results, 'ADD.best'),
                    pathlib.PurePath(folder_results, 'BalADD.best'),
                    pathlib.PurePath(folder_results, 'MUL.best'),
                    pathlib.PurePath(folder_results, 'BalMUL.best')
                ]

    for metric_result, each_path in zip(model_results, best_paths):
        f_test_result = open(each_path, 'a+')
        top_sub = list(metric_result.keys())[0]
        
        string_to_write = tw_pos + delimeter_best + top_sub + '\n'
        f_test_result.write(string_to_write)
        f_test_result.close()


def get_metrics_results(
  dataset: pd.DataFrame,
  all_candidates_dict: Dict[str, List[str]],
  eng_embed,
  esp_embed,
  context_window_params: Tuple[int, int],
  stopwords_params: Tuple[bool, bool],
  folder_results: str
) -> None:
    threshold_oot = 10
    threshold_best = 1
    context_window_oot = context_window_params[0]
    context_window_best = context_window_params[1]
    stopwords_oot = stopwords_params[0]
    stopwords_best = stopwords_params[1]
    for i, dataset_string in enumerate(tqdm(dataset.iterrows())):
        target_word = dataset.iloc[i, 0]
        sentence = dataset.iloc[i, 1]
        instance_word_info_for_result = dataset.iloc[i, 2]
        instance_word_with_pos = instance_word_info_for_result.split()[0]
        word_candidates = all_candidates_dict[instance_word_with_pos]
        oot_results = CL_substitution(
          sentence, target_word, eng_embed, esp_embed, word_candidates, threshold_oot, context_window_oot, stopwords_oot
          )
        oot_files_results_add(instance_word_info_for_result, oot_results, folder_results)

        best_results = CL_substitution(
          sentence, target_word, eng_embed, esp_embed, word_candidates, threshold_best, context_window_best, stopwords_best
          )
        best_files_results_add(instance_word_info_for_result, best_results, folder_results)