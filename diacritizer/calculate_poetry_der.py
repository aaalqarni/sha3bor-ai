import argparse

from pyarabic.araby import strip_tashkeel, LETTERS

from diacritizer.utils import get_diacritization_labels


def main(args: argparse.Namespace) -> None:
    with open(args.reference_file, 'r') as fp:
        reference = list(map(str.strip, fp.readlines()))

    with open(args.predicted_file, 'r') as fp:
        predicted = list(map(str.strip, fp.readlines()))

    correct, total, unknown = 0, 0, 0
    for r, p in zip(reference, predicted):
        r_diacritics = get_diacritization_labels(r)
        p_diacritics = get_diacritization_labels(p)

        assert len(strip_tashkeel(r)) == len(strip_tashkeel(p))
        assert len(r_diacritics) == len(p_diacritics)
        assert len(strip_tashkeel(r)) == len(r_diacritics)

        for r_char, r_diacritic, p_diacritic in zip(r, r_diacritics, p_diacritics):
            if r_char in LETTERS and p_diacritic != 0:
                if r_diacritic == p_diacritic:
                    correct += 1
                elif r_diacritic == 0 and p_diacritic != 0:
                    unknown += 1
                total += 1
    total -= unknown

    print(f'نسبة الخطأ هي: {round((total - correct) / total * 100, 2)}%')
    print(f'العدد الكلي للحروف التي تم أخذها بعين الإعتبار: {total}')
    print(f'عدد الحروف التي تم تشكيلها بشكل صحيح: {correct}')
    print(f'عدد التشكيلات التي لم نتأكد من صحتها: {unknown}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference_file', required=True, help='مسار ملف التشكيل الأصلي')
    parser.add_argument('--predicted_file', required=True, help='مسار ملف تشكيل النموذج')
    args = parser.parse_args()

    main(args)
