from typing import List

import transformers

from transformers import AutoTokenizer

from pyarabic.araby import TASHKEEL

from diacritizer.constants import TASHKEEL_TO_ID


def load_tokenizer(model_type: str) -> transformers.CanineTokenizer:
    """تقوم هذه الدالة بتجهيز معالج بيانات النموذج وتضيف عليه بعض الرموز الجديدة"""

    tokenizer = AutoTokenizer.from_pretrained(f'google/canine-{model_type}')

    tokenizer.add_tokens('\ue005')

    return tokenizer


def get_diacritization_labels(text: str) -> List[int]:
    """تقوم هذه الدالة بإستخراج التشكيلات من النص على هيئة قائمة من الارقام"""

    while text[0] in TASHKEEL:
        text = text[1:]
    text += ' '

    labels = list()
    diacritics_expected = False
    i = 0
    while i < len(text):
        if diacritics_expected:
            if text[i] in TASHKEEL:
                diacritics = ''
                while i < len(text) and text[i] in TASHKEEL:
                    diacritics += text[i]
                    i += 1

                try:
                    labels.append(TASHKEEL_TO_ID[diacritics])
                except:
                    labels.append(TASHKEEL_TO_ID[''])
            else:
                labels.append(TASHKEEL_TO_ID[''])
        else:
            i += 1

        diacritics_expected = not diacritics_expected

    return labels
