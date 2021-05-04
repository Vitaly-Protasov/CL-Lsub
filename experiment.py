from utils import return_best_candidates, get_clear_context
from wordfreq import word_frequency
try:
    from fastText_multilingual import fasttext
except:
    print('Try: git clone https://github.com/babylonhealth/fastText_multilingual.git')
from typing import Union, List, Dict


def cos_sim_by_embeddings(word1_embed, word2_embed):
    """
    """
    sim = fasttext.FastVector.cosine_similarity(
        word1_embed,
        word2_embed
        )
    return sim


def get_metrics_for_each_candidate(
      sentence: str,
      target_word: str,
      eng_lang_embed,
      esp_lang_embed,
      each_candidate: str,
      context_size: int,
      delete_stopwords: bool
):
    """
    """
    Add={}
    BalAdd={}
    Mul={}
    BalMul={}

    target_word = target_word.lower()
    context = get_clear_context(sentence, target_word, context_size, delete_stopwords)

    if target_word not in eng_lang_embed or each_candidate not in esp_lang_embed:
        return Add, BalAdd, Mul, BalMul

    target_word_embed = eng_lang_embed[target_word]
    candidate_embed = esp_lang_embed[each_candidate]
    s1 = cos_sim_by_embeddings(target_word_embed, candidate_embed)
    sp1 = (s1 + 1) / 2

    sa=0
    sm=1
    i=0
    for context_word_with_weight in context:
        context_word = context_word_with_weight[0]
        context_weight = context_word_with_weight[1] ** 2
        if context_word in eng_lang_embed:
            context_w_embed = eng_lang_embed[context_word]
            sa += cos_sim_by_embeddings(context_w_embed, candidate_embed) * context_weight
            sm *= ((sa + 1) / 2) * context_weight
            i += 1

    Add[each_candidate] = (s1 + sa) / (i + 1)
    Mul[each_candidate] = (sp1 * sm) ** (1 / (1 + i))
    if i!=0:
        BalAdd[each_candidate] = (i * s1 + sa)/(i*2)
        BalMul[each_candidate] = ((sp1**i)*sm)**(1/(2*i))
    else:
        i+=1
        BalAdd[each_candidate] = (i * s1 + sa) / (i * 2)
        BalMul[each_candidate] = ((sp1 ** i) * sm) ** (1 / (2 * i))
    
    return Add, BalAdd, Mul, BalMul


def CL_substitution(
      sentence: str,
      target_word: str,
      eng_lang_embed = None,
      esp_lang_embed = None,
      candidates: Union[List, Dict, None] = None,
      words_threshold: int = 10,
      context_size: int = 5,
      delete_stopwords: bool = False
) -> Dict[str, int]:
    """
    """
    Add_final = {}
    BalAdd_final = {}
    Mul_final = {}
    BalMul_final = {}

    if isinstance(candidates, dict):
        candidates = candidates[target_word]

    for each_candidate in candidates:
        Add_final[each_candidate] = 0
        BalAdd_final[each_candidate] = 0
        Mul_final[each_candidate] = 0
        BalMul_final[each_candidate] = 0
        
        splited_candidate = each_candidate.split()
        num_part_of_candidate = len(splited_candidate)

        for part_of_candidate in splited_candidate:
            if part_of_candidate in esp_lang_embed and len(part_of_candidate) > 1:
                Add, BalAdd, Mul, BalMul = get_metrics_for_each_candidate(
                    sentence,
                    target_word,
                    eng_lang_embed,
                    esp_lang_embed,
                    part_of_candidate,
                    context_size,
                    delete_stopwords
                )
                
                Add_output = Add.get(part_of_candidate, 0)
                BalAdd_output = BalAdd.get(part_of_candidate, 0)
                Mul_output = Mul.get(part_of_candidate, 0)
                BalMul_output = BalMul.get(part_of_candidate, 0)
                if Add_output >= Add_final[each_candidate]:
                  Add_final[each_candidate] = Add_output
                if BalAdd_output >= BalAdd_final[each_candidate]:
                  BalAdd_final[each_candidate] = BalAdd_output
                if Mul_output >= Mul_final[each_candidate]:
                  Mul_final[each_candidate] = Mul_output
                if BalMul_output >= BalMul_final[each_candidate]:
                  BalMul_final[each_candidate] = BalMul_output

        if num_part_of_candidate == 0:
            continue

        Add_final[each_candidate] /= num_part_of_candidate
        BalAdd_final[each_candidate] /= num_part_of_candidate
        Mul_final[each_candidate] /= num_part_of_candidate
        BalMul_final[each_candidate] /= num_part_of_candidate

    ADD_best = return_best_candidates(Add_final, words_threshold)
    BalADD_best = return_best_candidates(BalAdd_final, words_threshold)
    MUL_best = return_best_candidates(Mul_final, words_threshold)
    BalMUL_best = return_best_candidates(BalMul_final, words_threshold)

    return ADD_best, BalADD_best, MUL_best, BalMUL_best
