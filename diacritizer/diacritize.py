import argparse

from typing import List

import torch
import transformers

from pyarabic.araby import strip_tashkeel
from transformers import AutoModelForTokenClassification, pipeline

from diacritizer.constants import ID_TO_TASHKEEL
from diacritizer.utils import load_tokenizer


def preprocess_example(example: str) -> str:
    """تقوم هذه الدالة بتجهيز المدخلات قبل إرسالها للنموذج"""

    example = example.replace('***', '\ue005')
    example = ' '.join(example.split())

    return example


def postprocess_example(example: str, processed_example: str, predictions: List[int], min_prediction_score: int):
    """تقوم هذه الدالة بدمج التشكيلات التي تم استخراجها من النموذج مع النص الأصلي"""

    assert len(predictions) == len(processed_example)

    predictions = list(filter(lambda x: x['word'] != '\ue005', predictions))

    diacritized_example = ''
    i = 0
    for char in example:
        diacritized_example += char

        if i < len(predictions) and char == predictions[i]['word']:
            if predictions[i]['score'] > min_prediction_score:
                diacritized_example += ID_TO_TASHKEEL[int(predictions[i]['entity'].replace('LABEL_', ''))]
            i += 1

    assert i == len(predictions)

    return diacritized_example


def handle_user_input(min_prediction_score: int, diacritizer: transformers.TokenClassificationPipeline) -> None:
    """تقوم هذه الدالة باستقبال المدخلات من المستخدم وتشكيلها وطباعتها على الشاشة"""

    while True:
        text = input('أدخل بيت الشعر مفصول بإستخدام ***: ')
        processed_text = preprocess_example(text)
        predictions = diacritizer(processed_text)

        print(postprocess_example(text, processed_text, predictions, min_prediction_score))


def handle_file_input(
    input_file: str,
    output_file: str,
    min_prediction_score: int,
    diacritizer: transformers.TokenClassificationPipeline,
) -> None:
    """تقوم هذه الدالة بقراءة البيانات من ملف المدخلات وتشكيلها وكتابتها على ملف المخرجات"""

    with open(input_file, 'r') as fp:
        data = list(map(str.strip, fp.readlines()))
    data = list(map(strip_tashkeel, data))

    processed_data = list(map(preprocess_example, data))

    data_predictions = diacritizer(processed_data)

    diacritized_data = [
        postprocess_example(example, processed_example, predictions, min_prediction_score)
        for example, processed_example, predictions in zip(data, processed_data, data_predictions)
    ]

    if output_file == '':
        print(*diacritized_data, sep='\n')
    else:
        with open(output_file, 'w') as fp:
            fp.write('\n'.join(diacritized_data))


def main(args: argparse.Namespace) -> None:
    # تجهيز معالج بيانات النموذج.
    tokenizer = load_tokenizer(args.model_type)

    # تجهيز النموذج.
    model = AutoModelForTokenClassification.from_pretrained(args.model_name_or_path)

    # التعرّف على الجهاز الذي يجب استخدامه.
    device = -1
    if not args.no_gpu and torch.cuda.is_available():
        device = 0

    # تجهيز مسار تشكيل النصوص.
    diacritizer = pipeline('ner', model=model, tokenizer=tokenizer, device=device)

    if args.input_file == '':
        handle_user_input(args.min_prediction_score, diacritizer)
    else:
        handle_file_input(args.input_file, args.output_file, args.min_prediction_score, diacritizer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', required=True, help='اسم أو مسار النموذج الذي يجب استخدامه')
    parser.add_argument('--model_type', choices=['s', 'c'], required=True, help='نوع النموذج الذي يجب تدريبه')
    parser.add_argument('--no_gpu', action='store_true', help='عدم استخدام المعالج الرسومي')
    parser.add_argument('--input_file', default='', help='مسار الملف الذي يحتوي على المدخلات إن وُجِد')
    parser.add_argument('--output_file', default='', help='مسار كتابة البيانات بعد التشكيل في حال استخدام ملف')
    parser.add_argument('--min_prediction_score', type=float, default=0, help='أقل عدد نقاط يمكن اعتباره صحيحًا')
    args = parser.parse_args()

    main(args)
