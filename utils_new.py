import collections
from nltk.tokenize import RegexpTokenizer
import nltk
import pandas as pd
import subprocess
import os
import re
from scipy.stats import truncnorm
from typing import List, Dict, Set, Union, Tuple, OrderedDict
nltk.download('stopwords')
from nltk.corpus import stopwords


def substitute_nonunicode_letters(nonunicode_word: str) -> str:
    substitution_word = nonunicode_word[:]
    substitution_letters_pairs = {
        'ñ': 'n', 'ó': 'o', 'í': 'i', 'é': 'e', 'á': 'a', 'ú': 'u', 'ï': 'i', 'ṅ': 'n', 'ā': 'a',
        'а́': 'а', 'ы́': 'ы', 'у́': 'y', 'и́': 'и', 'ю́': 'ю', 'е́': 'е', 'о́': 'о', 
    }
    for i in substitution_letters_pairs:
        if i in substitution_letters_pairs:
            substitution_word = substitution_word.replace(i, substitution_letters_pairs[i])

    return substitution_word


def isascii(word: str) -> bool:
  """Check if the characters in string s are in ASCII, U+0-U+7F."""
  return len(word) == len(word.encode())


def clearing_word(word: str) -> str:
  bad_symbols = ['\'', '/', '_', '^', '@', '-']
  if not isascii(word):
      word = substitute_nonunicode_letters(word)
  if all(c not in bad_symbols for c in word) and not any(map(str.isdigit, word)):
      cleared_w = re.sub(r'[^\w]', '', word.lower().strip())
      if len(cleared_w) >= 2:
          return cleared_w
  return ''


def form_submission(dict_of_predictions: List[OrderedDict[str, float]], df_column: pd.DataFrame, type_task: str, filename:str):
    file_path = f'{filename}.{type_task}'
    if os.path.exists(file_path):
        os.remove(file_path)
    f_test_result = open(file_path, 'a+')
    if type_task == 'oot':
        delimeter = ' ::: '
        for cands_index, name in zip(dict_of_predictions, df_column):
            list_candidates = list(dict_of_predictions[cands_index].keys())

            top_subs = ';'.join(list_candidates) + ';'
            string_to_write = name + delimeter + top_subs + '\n'
            f_test_result.write(string_to_write)
        f_test_result.close()
    elif type_task == 'best':
        delimeter = ' :: '
        for cands_index, name in zip(dict_of_predictions, df_column):
            best_cand = list(dict_of_predictions[cands_index].keys())[0]

            string_to_write = name + delimeter + best_cand + '\n'
            f_test_result.write(string_to_write)
        f_test_result.close()


def print_semeval2010_2_file_results(file_path, gold_filepath, path_to_score_pl):
    """
    """
    command_list = ['perl', path_to_score_pl, file_path, gold_filepath]

    if file_path.endswith('.oot'):
        command_list.extend(['-t', 'oot'])
    result = subprocess.run(command_list, 
                            stderr=subprocess.PIPE,
                            stdout=subprocess.PIPE, encoding='utf-8')
    output = result.stdout.split('\n')
    for i in output:
        print(i)