import collections
from nltk.tokenize import RegexpTokenizer
import nltk
import pandas as pd
import subprocess
import os
import re
from scipy.stats import truncnorm
from typing import List, Dict, Set, Union, Tuple
nltk.download('stopwords')
from nltk.corpus import stopwords


supervised_weights = [
  0.19102208041588065,
  0.3120197570437705,
  0.4434057031112993,
  0.5149057201766486,
  0.6245851170302262,
  0.7530687851551111,
  0.858687172640565,
  0.9916933056066202,
  0.994062241996192
]


def get_truncated_normal(mean=0, sd=1, low=0, upp=1):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


def get_stopwords() -> Set[str]:
    """
    """
    return set(stopwords.words('english'))
    

def return_best_candidates(dict_candidates, words_threshold):
    """
    """
    sordet_candidates_dict = sorted(dict_candidates.items(), key=lambda kv: kv[1], reverse=True)[:words_threshold]
    return collections.OrderedDict(sordet_candidates_dict)

def is_suitable_word(current_word: str, target_word: str, set_stopwords: List[str]):
    bad_tokens = ['quot', 'apos']
    if current_word not in bad_tokens and not current_word.isdigit():
        if target_word == current_word or current_word not in set_stopwords:
            return True
    return False


def add_weight_to_context(list_of_context: List[str], is_left: bool = False):
    list_with_weights = []
    num_words = len(list_of_context)
    weights = supervised_weights[-num_words:]
    if not is_left:
        weights = weights[::-1]
    for cont_w, weight in zip(list_of_context, weights):
      list_with_weights.append((cont_w, weight))
    
    return list_with_weights


def get_clear_context(sentence: str, target_word: str, context_size: int, delete_stopwords: bool) -> List[str]:
    """
    """
    tokenizer = RegexpTokenizer(r'\w+')
    sentence_list = tokenizer.tokenize(sentence.lower())
    set_stopwords = []
    if delete_stopwords:
      set_stopwords = get_stopwords()

    filtered_context = [
                        w for w in sentence_list if is_suitable_word(
                            w, target_word, set_stopwords
                        )
                        ]
    num_tokens = len(filtered_context)
    position_tw = filtered_context.index(target_word)
    context_with_weights = []
    if context_size <= position_tw:
        windows_context_left = filtered_context[position_tw - context_size : position_tw]
        windows_context_right = filtered_context[position_tw + 1 : position_tw  + context_size + 1]
        context_with_weights += add_weight_to_context(windows_context_left, True)
        context_with_weights += add_weight_to_context(windows_context_right)
    else:
      windows_context_left = filtered_context[:position_tw]
      windows_context_right = filtered_context[position_tw + 1 : position_tw + context_size + 1]
      context_with_weights += add_weight_to_context(windows_context_left, True)
      context_with_weights += add_weight_to_context(windows_context_right)

    return context_with_weights


def print_as_df(dict_results: Tuple) -> pd.DataFrame:
    """
    """
    df_dict = {}
    assert len(dict_results) == 4
    columnn_names = ['ADD', 'BalADD', 'MUL', 'BalMUL']
    for name, each_dict in zip(columnn_names, dict_results):
        df_dict[name] = list(each_dict.keys())
    return pd.DataFrame(df_dict)


def print_metrics_results(metric_path_to_folder, gold_filepath, path_to_score_pl):
    """
    """
    for i in os.listdir(metric_path_to_folder):
        metric_path = metric_path_to_folder + i
        command_list = ['perl', path_to_score_pl, metric_path, gold_filepath]

        if i.endswith('.oot'):
            command_list.extend(['-t', 'oot'])
        result = subprocess.run(command_list, 
                                stderr=subprocess.PIPE,
                                stdout=subprocess.PIPE, encoding='utf-8')
        output = result.stdout.split('\n')
        for i in output:
          print(i)


def isascii(s):
    """Check if the characters in string s are in ASCII, U+0-U+7F."""
    return len(s) == len(s.encode())

def substitute_spanish_letters(spanish_word):
    substitution_word = spanish_word[:]
    substitution_letters_pairs = [
                                  ('ñ', 'n'),
                                  ('ó', 'o'),
                                  ('í', 'i'),
                                  ('é', 'e'),
                                  ('á', 'a'),
                                  ('ú', 'u'),
                                  ('ï', 'i'),
                                  ('ṅ', 'n'),
                                  ('ā', 'a')
    ]
    for i in substitution_letters_pairs:
        substitution_word = substitution_word.replace(i[0], i[1])

    return substitution_word


def check_that_no_dupli_word(clear_cand_list):
  for w in clear_cand_list:
    if clear_cand_list.count(w) > 1:
      return False
  return True


def count_cand(candidates):
  s = 0
  for i in candidates.values():
    s += len(list(i))
  return s


def filter_candidates_dict(cand_dict):
  new_dict = {}
  for each_tw in cand_dict:
    tw_candidates = []
    for i in cand_dict[each_tw]:
      tokenizer = RegexpTokenizer(r'\w+')
      cleared_i = re.sub(r'\".*?\"', '', i)
      cleared_i = cleared_i.lower().lstrip().rstrip()
      cleared_i = cleared_i.replace('   ', ' ').replace('  ', ' ')
      clear_cand_list = tokenizer.tokenize(cleared_i)
      if not isascii(cleared_i):
        cleared_i = substitute_spanish_letters(cleared_i)
      if isascii(cleared_i) and check_that_no_dupli_word(clear_cand_list):
        if cleared_i not in tw_candidates and len(cleared_i.split()) < 6:
          tw_candidates.append(cleared_i)

    new_dict[each_tw] = tw_candidates
  print(f'\n{count_cand(cand_dict) - count_cand(new_dict)} words were deleted')
  return new_dict


def cut_dict(dict_to_work, length = 20):
    new_dict = {}
    for k, v in dict_to_work.items():
        new_dict[k] = v[:length]
    return new_dict
