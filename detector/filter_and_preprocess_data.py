import argparse

import pandas as pd

from arabert.preprocess import ArabertPreprocessor

from detector.constants import METRES


def filter_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    تقوم هذه الدالة بتصفية البيانات من خلال:
    - إزالة البيت إذا كانت أحد خصائصه غير موجودة.
    - إبقاء الأبيات التي تنتمي لأحد البحور الرئيسية الموجودة في METRES.

    ثم ترجع البيانات المحدثة.
    """

    filtered_data = data.dropna()
    filtered_data = filtered_data[filtered_data['البحر'].isin(METRES)]

    return filtered_data


def process_data(data: pd.DataFrame, arabert_preprocessor: ArabertPreprocessor) -> pd.DataFrame:
    """
    تقوم هذه الدالة بمعالجة البيانات من خلال:
    - تحويل القافية "هـ" إلى "هه" لضمان اختلافها عن القافية "ه" بعد المعالجة باستخدام معالج نصوص AraBERT.
    - جمع شطريْ البيت بفاصلة وهي "[شطر]" وإضافة البحر والقافية إلى نهايته.
    - تغيير أسماء بعض الأعمدة في البيانات.

    ثم ترجع البيانات المحدثة.
    """

    data['القافية'] = data['القافية'].str.replace('ـ', 'ه')

    data['text'] = data['الشطر الايمن'] + ' [شطر] ' + data['الشطر الايسر']
    data['metre_label'] = data['البحر']
    data['rhyme_label'] = data['القافية']

    processed_data = data[['text','metre_label', 'rhyme_label']]
    processed_data['text'] = processed_data['text'].apply(arabert_preprocessor.preprocess)

    return processed_data


def main(args: argparse.Namespace) -> None:
    arabert_preprocessor = ArabertPreprocessor(args.model_name)

    data = pd.read_csv(args.input_file)
    filtered_data = filter_data(data)
    processed_data = process_data(filtered_data, arabert_preprocessor)

    processed_data.to_csv(args.output_file, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', required=True, help='اسم النموذج الذي يجب استخدامه لمعالجة البيانات بطريقته')
    parser.add_argument(
        '--input_file',
        default='data/Arabic Poem Comprehensive Dataset (APCD).csv',
        help='مسار البيانات',
    )
    parser.add_argument(
        '--output_file',
        default='detector/data/Preprocessed Arabic Poem Comprehensive Dataset (APCD).csv',
        help='مسار كتابة البيانات بعد المعالجة',
    )
    args = parser.parse_args()

    main(args)
