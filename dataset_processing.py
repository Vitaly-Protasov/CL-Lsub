import re
import pandas as pd
from py_babelnet.calls import BabelnetAPI
from typing import List, Set, Dict
from tqdm import tqdm


def get_dataset_dict(data: str, gold: str):
    dataset_dict = {
        'target_word': [],
        'context': [],
        'gold_candidates': []
        }

    f_dataset = open(data)
    f_gold = open(gold)
    for i in f_dataset:
        start_string = '<context>'
        end_string = '</context>'
        start_tw = '<head>'
        end_tw = '</head>'
        string = i.strip().rstrip()
        if start_string in string:
            context = string[len(start_string) : len(string) - len(end_string)]
            target_word = re.findall(r'\<head>(.*?)\</head>', string)[0]
            final_context = context.replace(start_tw, '').replace(end_tw, '')
            dataset_dict['target_word'].append(target_word)
            dataset_dict['context'].append(final_context)
    
    for j, i in enumerate(f_gold):
        gold_candidates_with_numbers = i.split(' :: ')[1]
        dataset_dict['gold_candidates'].append(gold_candidates_with_numbers)

    return pd.DataFrame(dataset_dict)


def get_list_of_candidates_by_gold(df_candidates_column: pd.DataFrame):
    set_candidates = set()
    for i in df_candidates_column:
        string_no_numbers = re.sub("\d+", " ", i)
        set_each_tw_candidates = set(string_no_numbers.split('  ;')[:-1])
        set_candidates = set_candidates.union(set_each_tw_candidates)
    return list(set_candidates)


def get_dataset_word_context_babelnet(data_path: str, gold_data_path: str):
    dataset_dict = {
        'target_word': [],
        'context': [],
        'tw_for_metrics' : []
        }

    f_dataset = open(data_path)
    f_gold = open(gold_data_path)
    for i in f_dataset:
        start_string = '<context>'
        end_string = '</context>'
        start_tw = '<head>'
        end_tw = '</head>'
        string = i.strip().rstrip()
        if start_string in string:
            target_word = re.findall(r'\<head>(.*?)\</head>', string)[0]
            dataset_dict['target_word'].append(target_word)
            context = string[len(start_string) : len(string) - len(end_string)]
            final_context = context.replace(start_tw, '').replace(end_tw, '')
            dataset_dict['context'].append(final_context)

    for j in f_gold:
        tw = j.split(' :: ')[0]
        dataset_dict['tw_for_metrics'].append(tw)

    return pd.DataFrame(dataset_dict)


def get_correct_pos(cut_pos):
  if cut_pos == 'n':
    return 'NOUN'
  elif cut_pos == 'v':
    return 'VERB'
  elif cut_pos == 'a':
    return 'ADJ'
  elif cut_pos == 'r':
    return 'ADV'


def add_synonyms(candidate_word: str, keys: List[str], pos_word: str):
    new_words = []
    for key in keys:
        try:
          api = BabelnetAPI(key)
          list_synonims = api.get_senses(lemma = candidate_word, searchLang = "ES", pos = pos_word)
          for j in list_synonims:
            candidate_synonim = get_word_from_synset(j)
            if candidate_synonim not in new_words:
                new_words.append(candidate_synonim)
          return new_words
        except:
          continue
    return new_words


def get_word_from_synset(synset):
  # bad_symbols = ['_', '(', ')']
  candidate_word = synset.values()
  candidate_word = list(candidate_word)[1]['fullLemma'].lower()
  # for i in bad_symbols:
  #  if i in candidate_word:
  #    return ' '
  # candidate_word = re.sub(r'\(.*?\)', '', candidate_word)
  return candidate_word


def get_babelnet_words(keys: List[str], target_word_with_pos, is_synonyms = False):
  """
  For each english target word we obtain list of spanish
  synset
  """
  new_words = []
  target_word, pos = target_word_with_pos.split('.')
  pos_word = get_correct_pos(pos)
  for key in keys:
    try:
      api = BabelnetAPI(key)
      SYNSET_IDS = api.get_synset_ids(
          lemma = target_word, searchLang = "EN", targetLang = "ES", pos = pos_word
          )
      for syn_id in SYNSET_IDS:
        dict_words = api.get_synset(id = syn_id['id'], targetLang = "ES")
        words = []
        for w in dict_words['senses']:
          temp_word = w['properties']['fullLemma']
          temp_lang = w['properties']['language']
          if temp_lang == 'ES' and temp_word not in words:
            words.append(temp_word)
        for w in words:
          if w not in new_words:
            new_words.append(w)
      # for i in senset1:
      #   candidate_word = get_word_from_synset(i)
      #   if candidate_word not in new_words and candidate_word != ' ':
      #     new_words.append(candidate_word)
      #     # get synonims
      #     if is_synonyms:
      #       cands_synonyms = add_synonyms(candidate_word, keys, pos_word)
      #       for syn in cands_synonyms:
      #         if syn not in new_words:
      #           new_words.append(syn)
      return new_words
    except:
      continue
  return new_words


def get_dict_of_candidates(
  target_word_with_pos: List[str],
  keys: List[str],
  is_synonyms = False
  ) -> Dict[str, List[str]]:
  """
  This function is aimed to parse candidates in Spanish for each target word in
  English.
  Without any word processing.
  """
  word_candidates_dict = {}
  for temp_tw in tqdm(target_word_with_pos):
    if temp_tw not in word_candidates_dict:
      word_candidates_dict[temp_tw] = []
      spanish_candidates = get_babelnet_words(keys, temp_tw, is_synonyms)
      for i in spanish_candidates:
        if i not in word_candidates_dict[temp_tw]:
          word_candidates_dict[temp_tw].append(i)
  return word_candidates_dict


def words_processing(dict1):
    dict_clear = {}
    bad_symbols = ['(', ')']
    for k in dict1:
        dict_clear[k] = []
        old_values = dict1[k]
        for candidate in old_values:
            clear_cand = candidate.replace('_', ' ').lower()
            without_brackets = re.sub(r'\(.*?\)', '', clear_cand).strip()
            if without_brackets not in dict_clear[k]:
                dict_clear[k].append(without_brackets)
            try:
                within_brackets = clear_cand.split('(', 1)[1].split(')')[0]
                if within_brackets not in dict_clear[k]:
                    dict_clear[k].append(within_brackets)
            except:
                pass
            if bad_symbols[0] not in clear_cand and \
                bad_symbols[1] not in clear_cand and \
                clear_cand not in dict_clear[k]:
                dict_clear[k].append(clear_cand)
    return dict_clear
