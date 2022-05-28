import argparse
import warnings

from typing import Dict
from functools import partial

import datasets
import numpy as np
import transformers

from datasets import load_dataset, load_metric
from pyarabic.araby import strip_tashkeel
from transformers import AutoModelForTokenClassification, DataCollatorForTokenClassification, Trainer, TrainingArguments

from diacritizer.constants import BLOCK_SIZE, TASHKEEL_TO_ID
from diacritizer.utils import load_tokenizer, get_diacritization_labels


def load_model(model_name_or_path: str) -> transformers.CanineForTokenClassification:
    """تقوم هذه الدالة بتحميل النموذج المطلوب بناءً على اسمه أو مساره."""

    model = AutoModelForTokenClassification.from_pretrained(model_name_or_path, num_labels=len(TASHKEEL_TO_ID))

    return model


def preprocess_examples(
    examples: datasets.arrow_dataset.Batch,
    tokenizer: transformers.CanineTokenizer,
) -> transformers.BatchEncoding:
    labels = list()
    processed_examples = list()

    for element in examples['text']:
        element = element.replace('[شطر]', '\ue005')  # معالجة خاصة للتعامل مع بيانات الشعر.
        element = ' '.join(element.split())

        element_labels = get_diacritization_labels(element)
        element = strip_tashkeel(element)

        assert len(element_labels) == len(element)

        for i in range(0, len(element_labels), BLOCK_SIZE):
            labels.append([-100] + element_labels[i : i + BLOCK_SIZE - 2] + [-100])
            processed_examples.append(element[i : i + BLOCK_SIZE - 2])

    tokenized_inputs = tokenizer(processed_examples, truncation=True)
    tokenized_inputs['labels'] = labels

    return tokenized_inputs


def load_and_prepare_dataset(
    train_file: str,
    valid_file: str,
    tokenizer: transformers.CanineTokenizer,
) -> datasets.DatasetDict:
    """
    تقوم هذه الدالة بتجهيز البيانات من خلال:
    - تقسيم البيانات إلى بيانات تدريب (97.5%) وبيانات اختبار (2.5%) إذا لم يوجد ملف لبيانات الاختبار.
    - تمرير الأمثلة من خلال دالة preprocess_examples ليتم معالجتها وتجهيزها لعملية التدريب.

    ثم ترجع البيانات التي تم تجهيزها.
    """

    if valid_file == '':
        dataset = load_dataset('text', split='train', data_files=train_file)
        dataset = dataset.train_test_split(test_size=2.5 / 100, seed=961)
    else:
        dataset = load_dataset(
            'text',
            data_files={
                'train': train_file,
                'test': valid_file,
            },
        )

    tokenized_dataset = dataset.map(
        preprocess_examples,
        batched=True,
        num_proc=8,
        remove_columns=dataset['train'].column_names,
        fn_kwargs={
            'tokenizer': tokenizer,
        },
    )

    return tokenized_dataset


def compute_metrics(seqeval_metric, predictions_and_labels) -> Dict[str, str]:
    predictions, labels = predictions_and_labels
    predictions = np.argmax(predictions, axis=2)

    # يجب إزالة العلامات المميزة (Special Tokens) التي تم إضافتها خلال عملية معالجة النصوص مسبقًا.
    true_predictions = [
        [p for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [l for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)
    ]

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        results = seqeval_metric.compute(predictions=true_predictions, references=true_labels)

    return {
        'precision': results['overall_precision'],
        'recall': results['overall_recall'],
        'f1': results['overall_f1'],
        'accuracy': results['overall_accuracy'],
    }


def train(
    args: argparse.Namespace,
    model: transformers.CanineForTokenClassification,
    tokenized_dataset: datasets.DatasetDict,
    data_collator: DataCollatorForTokenClassification,
) -> None:
    """تقوم هذه الدالة ببدء عملية تدريب النموذج بناءً على المدخلات التي تم استقبالها."""

    seqeval_metric = load_metric('seqeval')

    training_args = TrainingArguments(
        output_dir=args.output_path,
        learning_rate=2e-5,
        weight_decay=0.01,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.train_epochs,
        optim='adamw_torch',
        evaluation_strategy='epoch',
        save_strategy='epoch',
    )

    trainer = Trainer(
        args=training_args,
        model=model,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['test'],
        data_collator=data_collator,
        compute_metrics=partial(compute_metrics, seqeval_metric),
    )

    trainer.train(args.continue_training)
    trainer.save_model()


def main(args: argparse.Namespace) -> None:
    # تجهيز معالج بيانات النموذج.
    tokenizer = load_tokenizer(args.model_type)

    # تجهيز النموذج.
    model = load_model(args.model_name_or_path)

    # تجهيز البيانات.
    tokenized_dataset = load_and_prepare_dataset(args.train_file, args.valid_file, tokenizer)

    # تجهيز معالج حزم التدريب.
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    train(args, model, tokenized_dataset, data_collator)


if __name__ == '__main__':
    """
    لتدريب النموذج لتشكيل النصوص العربية العامة قبل تدريبه على الشعر بشكل مخصص، استعمل الأمر التالي عند تشغيل الملف:
    python -m diacritizer.finetune \
        --train_file=diacritizer/data/pretraining-train.txt \
        --valid_file=diacritizer/data/pretraining-valid.txt \
        --model_name_or_path=google/canine-s \
        --output_path=diacritizer/models/results-canine-s-pretrained \
        --batch_size=64 \
        --train_epochs=10

    ولتدريب النموذج لتشكيل الشعر بشكل مخصص، استعمل الأمر التالي لتشغيل الملف:
    python -m diacritizer.finetune \
        --model_name_or_path=<ضع في هذا المكان مسار النموذج الذي تم تدريبه مسبقًا لتشكيل النصوص العربية العامة> \
        --model_type=s \
        --train_file="diacritizer/data/Preprocessed Arabic Poem Comprehensive Dataset (APCD).txt" \
        --output_path=diacritizer/models/results-canine-s \
        --batch_size=256 \
        --train_epochs=15
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', required=True, help='اسم أو مسار النموذج الذي يجب تدريبه')
    parser.add_argument('--model_type', choices=['s', 'c'], required=True, help='نوع النموذج الذي يجب تدريبه')
    parser.add_argument(
        '--train_file',
        default='diacritizer/data/Preprocessed Arabic Poem Comprehensive Dataset (APCD).txt',
        help='مسار بيانات التدريب',
    )
    parser.add_argument('--valid_file', default='', help='مسار بيانات الاختبار')
    parser.add_argument('--output_path', default='diacritizer/models/results', help='مسار حفظ النموذج بعد التدريب')
    parser.add_argument('--batch_size', type=int, default=256, help='عدد الأمثلة في حزمة التدريب الواحدة')
    parser.add_argument('--train_epochs', type=int, default=15, help='عدد مرات تدريب النموذج على كامل البيانات')
    parser.add_argument('--continue_training', action='store_true', help='بدأ التدريب من حيث توقف في آخر مرة')
    args = parser.parse_args()

    main(args)
