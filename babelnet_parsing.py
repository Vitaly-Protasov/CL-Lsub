from py_babelnet.calls import BabelnetAPI
import pandas as pd
from tqdm import tqdm
from typing import List, Dict
from utils_new import clearing_word


class BabelnetCands:
    def __init__(self, tws_list: List[str], keys_list: List[str], is_synonyms: bool = False):
        self.tw_words_with_pos = tws_list
        self.keys = keys_list
        self.is_synonyms = is_synonyms
    
    def _get_correct_pos(self, cut_pos: str) -> str:
        pos_dict = {'n': 'NOUN', 'v': 'VERB', 'a': 'ADJ', 'r': 'ADV', 'j': 'ADJ'}
        return pos_dict[cut_pos]
    
    def _add_synonyms(self, target_word: str, pos_word: str, targetLang: str) -> List[str]:
        syn_new_words = []
        for key in self.keys:
            try:
              api = BabelnetAPI(key)
              senset2 = api.get_senses(lemma = target_word, searchLang = targetLang, pos = pos_word)
              for s in senset2:
                temp_word = s['properties']['fullLemma']
                temp_lang = s['properties']['language']
                cleared_word = clearing_word(temp_word)
                if temp_lang == targetLang and cleared_word not in syn_new_words + ['']:
                    syn_new_words.append(cleared_word)
              return syn_new_words
            except:
              continue
        return syn_new_words

    def _get_babelnet_words(self, tw_with_pos: str, searchLang: str, targetLang: str) -> List[str]:
        new_words = []
        target_word, pos = tw_with_pos.split('.')
        pos_word = self._get_correct_pos(pos)
        for key in self.keys:
          try:
            api = BabelnetAPI(key)
            SYNSET_IDS = api.get_synset_ids(
                lemma = target_word, searchLang = searchLang, targetLang = targetLang, pos = pos_word
                )
            for syn_id in SYNSET_IDS:
              dict_words = api.get_synset(id = syn_id['id'], targetLang = targetLang)
              for s in dict_words['senses']:
                temp_word = s['properties']['fullLemma']
                temp_lang = s['properties']['language']
                cleared_word = clearing_word(temp_word)
                if temp_lang == targetLang and cleared_word not in new_words + ['']:
                  new_words.append(cleared_word)
                  if self.is_synonyms:
                    list_synonyms = self._add_synonyms(cleared_word, pos_word, targetLang)
                    for each_synonym in list_synonyms:
                      if each_synonym not in new_words:
                        new_words.append(each_synonym)
            return new_words
          except:
            continue
        return new_words

    def get_dict_of_candidates(self, searchLang: str='EN', targetLang: str='ES') -> Dict[str, List[str]]:
        word_to_candidates = {}
        for target_word in tqdm(self.tw_words_with_pos):
            if target_word not in word_to_candidates:
                word_to_candidates[target_word] = []
                spanish_candidates = self._get_babelnet_words(target_word, searchLang, targetLang)
                word_to_candidates[target_word] = spanish_candidates
        return word_to_candidates
