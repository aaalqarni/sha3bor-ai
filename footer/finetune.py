import argparse

from typing import Dict
from functools import partial

import datasets
import numpy as np
import transformers

from datasets import load_dataset, load_metric
from transformers import AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer

from footer.constants import FEET_51
from footer.utils import load_tokenizer


def load_model(model_name: str) -> transformers.BertForSequenceClassification:
    """
    تقوم هذه الدالة بتحميل النموذج المطلوب بناءً على اسمه.
    كذلك يتم تغيير عدد المخرجات الخاصة بالنموذج بناءً على عدد التفعيلات التي سيتم توقعها.
    """

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(FEET_51))

    return model


def preprocess_examples(
    examples: datasets.arrow_dataset.Batch,
    tokenizer: transformers.BertTokenizer,
) -> transformers.BatchEncoding:
    tokenized_inputs = tokenizer(examples['part'], truncation=True)

    tokenized_inputs['label'] = list(map(FEET_51.index, examples[f'foot']))

    return tokenized_inputs


def load_and_prepare_dataset(
    input_file: str,
    tokenizer: transformers.BertTokenizer,
) -> datasets.DatasetDict:
    """
    تقوم هذه الدالة بتجهيز البيانات من خلال:
    - تقسيم البيانات إلى بيانات تدريب (97.5%) وبيانات اختبار (2.5%) إذا لم يوجد ملف لبيانات الاختبار.
    - تمرير الأمثلة من خلال دالة preprocess_examples ليتم معالجتها وتجهيزها لعملية التدريب.

    ثم ترجع البيانات التي تم تجهيزها.
    """
    dataset = load_dataset('csv', split='train', data_files=input_file)
    dataset = dataset.train_test_split(test_size=2.5 / 100, seed=961)

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


def compute_metrics(accuracy_metric, f1_metric, predictions_and_labels) -> Dict[str, str]:
    predictions, labels = predictions_and_labels
    predictions = np.argmax(predictions, axis=1)

    return {
        'accuracy': accuracy_metric.compute(predictions=predictions, references=labels)['accuracy'],
        'f1_micro': f1_metric.compute(predictions=predictions, references=labels, average='micro')['f1'],
        'f1_macro': f1_metric.compute(predictions=predictions, references=labels, average='macro')['f1'],
        'f1_weighted': f1_metric.compute(predictions=predictions, references=labels, average='weighted')['f1'],
    }


def train(
    args: argparse.Namespace,
    model: transformers.BertForSequenceClassification,
    tokenized_dataset: datasets.DatasetDict,
    data_collator: DataCollatorWithPadding,
) -> None:
    """تقوم هذه الدالة ببدء عملية تدريب النموذج بناءً على المدخلات التي تم استقبالها."""

    accuracy_metric = load_metric('accuracy')
    f1_metric = load_metric('f1')

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
        compute_metrics=partial(compute_metrics, accuracy_metric, f1_metric),
    )

    trainer.train(args.continue_training)
    trainer.save_model()


def main(args: argparse.Namespace) -> None:
    # تجهيز معالج بيانات النموذج.
    tokenizer = load_tokenizer(args.model_name)

    # تجهيز النموذج.
    model = load_model(args.model_name)

    # تجهيز البيانات.
    tokenized_dataset = load_and_prepare_dataset(args.input_file, tokenizer)

    # تجهيز معالج حزم التدريب.
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train(args, model, tokenized_dataset, data_collator)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', required=True, help='اسم النموذج الذي يجب تدريبه')
    parser.add_argument(
        '--input_file',
        default='footer/data/Preprocessed Arabic Poem Comprehensive Dataset (APCD).csv',
        help='مسار البيانات',
    )
    parser.add_argument('--output_path', default='footer/models/results', help='مسار حفظ النموذج بعد التدريب')
    parser.add_argument('--batch_size', type=int, default=512, help='عدد الأمثلة في حزمة التدريب الواحدة')
    parser.add_argument('--train_epochs', type=int, default=10, help='عدد مرات تدريب النموذج على كامل البيانات')
    parser.add_argument('--continue_training', action='store_true', help='بدأ التدريب من حيث توقف في آخر مرة')
    args = parser.parse_args()

    main(args)
