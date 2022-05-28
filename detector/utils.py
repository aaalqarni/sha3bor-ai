import transformers

from transformers import AutoTokenizer


def load_tokenizer(model_name: str) -> transformers.BertTokenizer:
    """تقوم هذه الدالة بتجهيز معالج بيانات النموذج وتضيف عليه بعض الرموز الجديدة"""

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokenizer.add_tokens('[شطر]')

    return tokenizer
