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
