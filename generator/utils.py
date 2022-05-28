import transformers

from transformers import AutoTokenizer

from generator.constants import METRES_WITH_RHYMES


def load_tokenizer(model_size: str) -> transformers.GPT2TokenizerFast:
    """تقوم هذه الدالة بتجهيز معالج بيانات النموذج وتضيف عليه بعض الرموز الجديدة"""

    tokenizer = AutoTokenizer.from_pretrained(f'aubmindlab/aragpt2-{model_size}')

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens('[شطر]')
    tokenizer.add_tokens(METRES_WITH_RHYMES)

    return tokenizer
