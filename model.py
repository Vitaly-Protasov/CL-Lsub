from utils import return_best_candidates, get_clear_context
from fastText_multilingual import fasttext
from typing import Union, List, Dict, Tuple, OrderedDict
from tqdm import trange
import collections
try:
    from nltk.corpus import stopwords
except:
    nltk.download('stopwords')
    from nltk.corpus import stopwords


class CL_embeddings:
    def __init__(
        self,
        tw_embed_model: fasttext.FastVector, 
        cand_embedding: fasttext.FastVector,
        candidates_dict: Dict[str, List[str]],
        num_candidates: int,
        context_size: int,
        delete_stopwords: bool
    ):
        self.tw_embed_model = tw_embed_model
        self.cand_embedding = cand_embedding
        self.candidates_dict = candidates_dict
        self.num_candidates = num_candidates
        self.context_size = context_size
        self.delete_stopwords = delete_stopwords
        self.set_stopwords = []
        if delete_stopwords:
            self.set_stopwords = set(stopwords.words('english'))

    def _get_clear_context(
        self,
        target_word: str,
        context: Union[str, List[str]],
        target_word_id: Optional[int] = None
    ) -> List[Tuple[str, float]]:
        
        context_with_weights = []
        if not isinstance(context, list):
            context = context.split()
        
        lower_context = [c.lower().strip() for c in context]
        filtered_context = [w for w in lower_context if w == target_word or w not in self.set_stopwords]
        if target_word_id is not None:
            position_tw = target_word_id
        else:
            position_tw = filtered_context.index(target_word)

        if self.context_size <= position_tw:
            windows_context_left = filtered_context[position_tw - self.context_size : position_tw]
            windows_context_right = filtered_context[position_tw + 1 : position_tw  + self.context_size + 1]
            context_with_weights += self.__add_weight_to_context(windows_context_left, True)
            context_with_weights += self.__add_weight_to_context(windows_context_right)
        else:
            windows_context_left = filtered_context[:position_tw]
            windows_context_right = filtered_context[position_tw + 1 : position_tw + self.context_size + 1]
            context_with_weights += self.__add_weight_to_context(windows_context_left, True)
            context_with_weights += self.__add_weight_to_context(windows_context_right)

        return context_with_weights

    def __add_weight_to_context(
        self,
        list_of_context: List[str],
        is_left: bool = False
    ) -> List[Tuple[str, float]]:
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
        list_with_weights = []
        num_words = len(list_of_context)
        weights = supervised_weights[-num_words:]
        if not is_left:
            weights = weights[::-1]
        for cont_w, weight in zip(list_of_context, weights):
            list_with_weights.append((cont_w, weight))
        return list_with_weights

    def cos_sim_by_embeddings(
        self,
        word_embedding_1: fasttext.FastVector,
        word_embedding_2: fasttext.FastVector
    ) -> float:
        return fasttext.FastVector.cosine_similarity(word_embedding_1, word_embedding_2)
    
    def BalAdd_BalMul(
        self,
        tw_embedding: fasttext.FastVector,
        cand_embedding: fasttext.FastVector,
        target_word: str,
        context: Union[str, List[str]],
        target_word_id: Optional[int]
    ) -> List[Tuple[float, float]]:
        context_weights = self._get_clear_context(target_word, context, target_word_id)
        s1 = self.cos_sim_by_embeddings(tw_embedding, cand_embedding)
        sp1 = (s1 + 1) / 2
        sa=0
        sm=1
        i=0
        for context_word_with_weight in context_weights:
            context_word = context_word_with_weight[0]
            context_weight = context_word_with_weight[1] ** 2
            if context_word in self.tw_embed_model:
                context_w_embed = self.tw_embed_model[context_word]
                sa += self.cos_sim_by_embeddings(context_w_embed, cand_embedding) * context_weight
                sm *= ((sa + 1) / 2) * context_weight
                i += 1
        
        if i != 0:
            BalAdd = (i * s1 + sa)/(i * 2)
            BalMul = ((sp1 ** i) * sm)**(1 / (2 * i))
        else:
            i += 1
            BalAdd = (i * s1 + sa) / (i * 2)
            BalMul = ((sp1 ** i) * sm) ** (1 / (2 * i))
        return (BalAdd, BalMul)

    def substites_for_one_tw(
        self,
        target_word: str,
        lemma_with_pos: str,
        context: Union[str, List[str]],
        target_word_id: Optional[int] = None
    ) -> OrderedDict[str, float]:
        target_word = target_word.lower().strip()
        tw_embedding = self.tw_embed_model[target_word]
        substites_scores_dict = {}
        tw_candidates = self.candidates_dict[lemma_with_pos]
        for each_candidate in tw_candidates:
            splited_candidate = [cand for cand in each_candidate.split() if cand in self.cand_embedding and len(cand) > 1]
            if len(splited_candidate) == 0:
                continue
            best_part_cand = sorted(
                splited_candidate,
                key=lambda c: self.cos_sim_by_embeddings(tw_embedding, self.cand_embedding[c]),
                reverse=True
            )[0]
            final_candidate_score = self.BalAdd_BalMul(
                tw_embedding, self.cand_embedding[best_part_cand], target_word, context, target_word_id
            )[0]
            substites_scores_dict[each_candidate] = final_candidate_score / len(splited_candidate)

        best_candidates_tuple = sorted(substites_scores_dict.items(), key=lambda kv: kv[1], reverse=True)[:self.num_candidates]
        return collections.OrderedDict(best_candidates_tuple)

    def clls_model(
        self,
        list_target_words: List[str],
        list_lemma_with_pos: List[str],
        list_contexts: List[Union[str, List[str]]],
        target_word_ids: Optional[List[int]] = None
    ) -> Dict[str, OrderedDict[str, float]]:
        language_dict = {}
        for i in trange(len(list_lemma_with_pos)):
            assert len(list_lemma_with_pos[i].split('.')) == 2
            lemma_with_pos = list_lemma_with_pos[i]
            target_word = list_target_words[i]
            context = list_contexts[i]
            target_word_id = None
            if target_word_ids is not None:
                target_word_id = target_word_ids[i]

            top_candidates_and_scores = self.substites_for_one_tw(target_word, lemma_with_pos, context, target_word_id)
            language_dict[i] = top_candidates_and_scores
        return language_dict
