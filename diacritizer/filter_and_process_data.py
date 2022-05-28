import random
import argparse

import pandas as pd

from pyarabic.araby import LETTERS

from diacritizer.utils import get_diacritization_labels


random.seed(961)


def filter_and_process_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    تقوم هذه الدالة بتصفية البيانات من خلال:
    - إزالة البيت إذا كانت أحد خصائصه غير موجوده.
    - إزالة البيت إذا كانت حروفه مشكّلة بنسبة تقل عن 25%.

    ثم تطبع معلومات البيانات المحدثه وترجعها.
    """

    filtered_data = data.dropna()

    good_verses = list()
    for verse, first_part, second_part in zip(
        filtered_data['البيت'],
        filtered_data['الشطر الايمن'],
        filtered_data['الشطر الايسر'],
    ):
        labels = get_diacritization_labels(verse)

        chars_count = 0
        diacritized_chars_count = 0

        for char, label in zip(verse, labels):
            if char in LETTERS:
                chars_count += 1

                if label != 0:
                    diacritized_chars_count += 1

        if diacritized_chars_count / (chars_count) >= 0.25:
            good_verses.append(f"{' '.join(first_part.split())} [شطر] {' '.join(second_part.split())}")

    print(f'عدد الأبيات التي بها تشكيل >= 25%: {len(good_verses)}')

    """
    عدد الأبيات التي بها تشكيل >= 25%: 658912
    """

    return good_verses


def main(args: argparse.Namespace) -> None:
    data = pd.read_csv(args.input_file)
    good_verses = filter_and_process_data(data)

    random.shuffle(good_verses)

    with open(args.output_file, 'w') as fp:
        fp.write('\n'.join(good_verses))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_file',
        default='data/Arabic Poem Comprehensive Dataset (APCD).csv',
        help='مسار البيانات',
    )
    parser.add_argument(
        '--output_file',
        default='diacritizer/data/Preprocessed Arabic Poem Comprehensive Dataset (APCD).txt',
        help='مسار كتابة البيانات بعد المعالجة',
    )
    args = parser.parse_args()

    main(args)
