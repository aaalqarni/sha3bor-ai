import argparse

import torch
import transformers

from transformers import AutoModelForCausalLM, pipeline

from arabert.preprocess import ArabertPreprocessor

from generator.constants import METRES, RHYMES, METRES_WITH_RHYMES
from generator.utils import load_tokenizer


def get_prompt(arabert_preprocessor: ArabertPreprocessor) -> str:
    prompt = input('أدخل بعض الكلمات: ')
    cleaned_prompt = arabert_preprocessor.preprocess(prompt)

    return cleaned_prompt


def get_metre() -> str:
    metre = int(
        input(f"أدخل رقم البحر [{'، '.join([str(index + 1) + ' - ' + metre for index, metre in enumerate(METRES)])}]: ")
        or '-1'
    )

    if metre != -1:
        metre = METRES[metre - 1]
        print(f'البحر الذي تم إختياره: {metre}')
    else:
        metre = ''
        print('لم يتم إختيار بحر.')

    return metre


def get_rhyme() -> str:
    rhyme = int(
        input(
            f"أدخل رقم القافية [{'، '.join([str(index + 1) + ' - ' + rhyme for index, rhyme in enumerate(RHYMES)])}]: "
        )
        or '-1'
    )

    if rhyme != -1:
        rhyme = RHYMES[rhyme - 1]
        print(f'القافية التي تم اختيارها: {rhyme}')
    else:
        rhyme = ''
        print('لم يتم إختيار قافية.')

    return rhyme


def get_bad_words_ids(metre: str, rhyme: str, tokenizer: transformers.GPT2TokenizerFast) -> str:
    """
    تقوم هذه الدالة بتجهيز الكلمات التي لا يجب على النموذج إخراجها
    بناءً على البحر والقافية التي تم تحديدها من قبل المستخدم.
    من خلال هذه الدالة يمكن للنموذج التأليف بناءً على بحر وقافية محددين.
    """

    bad_words_ids = None

    if metre and rhyme:
        metres_with_rhymes_copy = METRES_WITH_RHYMES[:]
        metres_with_rhymes_copy.remove(f'[{metre}{rhyme}]')
        metres_with_rhymes_copy = list(map(lambda x: f' {x}', metres_with_rhymes_copy))
        bad_words_ids = tokenizer(metres_with_rhymes_copy, add_special_tokens=False).input_ids

    return bad_words_ids


def split_metre_from_rhyme(metre_with_rhyme: str) -> str:
    """تقوم هذه الدالة بفصل البحر عن القافية"""

    metre_with_rhyme = metre_with_rhyme[1:-1]
    for metre in METRES:
        if metre in metre_with_rhyme and metre_with_rhyme.replace(metre, '') in RHYMES:
            return (metre, metre_with_rhyme.replace(metre, '').replace('هه', 'هـ'))

    raise ValueError('لا يمكن التعرف على البحر أو القافية!')


def format_poem(poem: str) -> str:
    """
    تقوم هذه الدالة بترتيب مخرجات النموذج وتنسيقها على هيئة قصيدة.
    كذلك تتأكد من أن القصيدة تتكون من بحر واحد وقافية واحدة.
    """

    metres_with_rhymes = list()

    for element in METRES_WITH_RHYMES:
        if element in poem:
            metres_with_rhymes.append(element)
            poem = poem.replace(f' {element} ', '\n')
            poem = poem.replace(f' {element}', '\n')
    poem = poem.replace('[شطر]', '***')

    if len(metres_with_rhymes) == 1:
        metre, rhyme = split_metre_from_rhyme(metres_with_rhymes[0])
        poem = f'البحر: {metre} - القافية: {rhyme}\n{poem}\n'
        return poem
    else:
        return 'يحتوي النص على أكثر من بحر وقافية!\n'


def handle_user_input(
    generator: transformers.TextGenerationPipeline,
    arabert_preprocessor: ArabertPreprocessor,
    tokenizer: transformers.GPT2TokenizerFast,
) -> None:
    """
    تقوم هذه الدالة بإستقبال البيانات من المستخدم وتجهيزها وتأليف النصوص بناءً على المدخلات.
    بعد ذلك تقوم بمعالجة المخرجات وترتيبها وعرضها للمستخدم.
    """

    while True:
        prompt = get_prompt(arabert_preprocessor)
        metre = get_metre()
        rhyme = get_rhyme()
        bad_words_ids = get_bad_words_ids(metre, rhyme, tokenizer)

        generated_text = generator(
            prompt,
            bad_words_ids=bad_words_ids,
            do_sample=not args.no_sample,
            max_length=args.max_length,
            num_beams=args.num_beams,
            num_return_sequences=args.num_return_sequences,
            top_k=args.top_k,
            top_p=args.top_p,
            pad_token_id=tokenizer.eos_token_id,
        )

        for index, element in enumerate(generated_text):
            print(f"النص #{index + 1}:\n{format_poem(element['generated_text'])}")


def main(args: argparse.Namespace) -> None:
    # تجهيز معالجة البيانات الأوَّلي من AraBERT.
    arabert_preprocessor = ArabertPreprocessor(f'aragpt2-{args.model_size}')

    # تجهيز معالج بيانات النموذج.
    tokenizer = load_tokenizer(args.model_size)

    # تجهيز النموذج.
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)

    # التعرّف على الجهاز الذي يجب استخدامه.
    device = -1
    if not args.no_gpu and torch.cuda.is_available():
        device = 0

    # تجهيز مسار تأليف النصوص.
    generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device=device)

    handle_user_input(generator, arabert_preprocessor, tokenizer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', required=True, help='اسم أو مسار النموذج الذي يجب استخدامه')
    parser.add_argument(
        '--model_size',
        choices=['base', 'medium', 'large', 'mega'],
        required=True,
        help='حجم النموذج الذي يجب استخدامه',
    )
    parser.add_argument('--no_gpu', action='store_true', help='عدم استخدام المعالج الرسومي')
    parser.add_argument('--no_sample', action='store_true', help='عدم استخدام Sampling')
    parser.add_argument('--max_length', type=int, default=200, help='أقصى طول للنص الذي سيتم تأليفه')
    parser.add_argument('--num_beams', type=int, default=75, help='عدد الأشعة التي سيتم إستخدامها خلال عملية التأليف')
    parser.add_argument('--num_return_sequences', type=int, default=1, help='عدد النصوص المُؤلَّفة التي يتم إرجاعها')
    parser.add_argument('--top_k', type=float, default=50)
    parser.add_argument('--top_p', type=float, default=0.92)
    args = parser.parse_args()

    main(args)
