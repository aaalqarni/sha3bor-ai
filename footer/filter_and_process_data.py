import random
import argparse

from collections import defaultdict

import pandas as pd

from tqdm import tqdm
from aruudy.poetry import prosody
from pyarabic.araby import FATHA, DAMMA, KASRA, SUKUN

from arabert.preprocess import ArabertPreprocessor

from footer.constants import METRE_TO_FEET


random.seed(961)


def process_data(data: pd.DataFrame, arabert_preprocessor: ArabertPreprocessor) -> pd.DataFrame:
    """
    تقوم هذه الدالة بإيجاد تفعيلة كل شطر من الأشطر الموجودة في البيانات من خلال:
    - استخراج الأشطر من البيانات.
    - استخراج الكتابة العروضية لكل شطر باستخدام مكتبة aruudy.
    - مطابقة البحر الخاص بالكتابة العروضية مع البحر الموجود في البيانات الأصلية:
        - إذا تطابق بحر الكتابة العروضية مع بحر البيانات الأصلية يتم إعتبار الكتابة العروضية صحيحة.
    """

    first_part_with_metre = list(data[['الشطر الايمن', 'البحر']].dropna().values)
    second_part_with_metre = list(data[['الشطر الايسر', 'البحر']].dropna().values)

    prosody_foot_to_orig = dict()
    for metre, feet in METRE_TO_FEET.items():
        for foot in feet:
            new_foot = ''

            for char in foot:
                if char in [FATHA, DAMMA, KASRA]:
                    new_foot += 'w'
                elif char == SUKUN:
                    new_foot += 's'

            prosody_foot_to_orig[new_foot] = (foot, metre)
 
    processed_data = defaultdict(list)
    unknown = list()
    for part_with_metre in tqdm(first_part_with_metre + second_part_with_metre):
        part, orig_metre = part_with_metre

        processed_part = prosody.process_shatr(part)

        if processed_part.ameter in prosody_foot_to_orig and orig_metre == prosody_foot_to_orig[processed_part.ameter][1]:
            processed_data[prosody_foot_to_orig[processed_part.ameter][0]].append(part)
        else:
            unknown.append(part)

    processed_data = sorted(processed_data.items(), key=lambda x: len(x[1]))[-args.required_feet:]
    processed_data = [[element, k] for k, v in processed_data for element in v]

    random.shuffle(unknown)
    processed_data.extend([[element, 'لم يتم التحديد'] for element in unknown[:5000]])

    processed_data = list(map(lambda x: [arabert_preprocessor.preprocess(x[0]), x[1]], processed_data))

    return processed_data


def main(args: argparse.Namespace) -> None:
    arabert_preprocessor = ArabertPreprocessor(args.model_name)

    data = pd.read_csv(args.input_file)
    processed_data = process_data(data, arabert_preprocessor)

    processed_data = pd.DataFrame(processed_data, columns=['part', 'foot'])
    processed_data.to_csv(args.output_file, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', required=True)
    parser.add_argument(
        '--input_file',
        default='data/Arabic Poem Comprehensive Dataset (APCD).csv',
        help='مسار البيانات',
    )
    parser.add_argument(
        '--output_file',
        default='footer/data/Preprocessed Arabic Poem Comprehensive Dataset (APCD).csv',
        help='مسار كتابة البيانات بعد المعالجة',
    )
    parser.add_argument('--required_feet', default=50, type=int, help='العدد المطلوب من التفعيلات')
    args = parser.parse_args()

    main(args)
