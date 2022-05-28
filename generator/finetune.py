import argparse

from typing import Dict, List

import datasets
import transformers

from datasets import load_dataset
from transformers import AutoModelForCausalLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments

from arabert.preprocess import ArabertPreprocessor

from generator.constants import BLOCK_SIZE
from generator.utils import load_tokenizer


def load_model(model_size: str, tokenizer: transformers.GPT2TokenizerFast) -> transformers.GPT2LMHeadModel:
    """تقوم هذه الدالة بتحميل النموذج المطلوب بناءً على حجمه."""

    model = AutoModelForCausalLM.from_pretrained(f'aubmindlab/aragpt2-{model_size}')

    model.resize_token_embeddings(len(tokenizer))

    return model


def preprocess_examples(
    examples: datasets.arrow_dataset.Batch,
    arabert_preprocessor: ArabertPreprocessor,
    tokenizer: transformers.GPT2TokenizerFast,
) -> transformers.BatchEncoding:
    preprocessed_examples = [arabert_preprocessor.preprocess(x) for x in examples['text']]
    return tokenizer(preprocessed_examples, truncation=True)


def group_examples(examples: datasets.arrow_dataset.Batch) -> Dict[str, List[int]]:
    result = dict()

    for k in examples.keys():
        result[k] = list()

        for example in examples[k]:
            for i in range(0, len(example), BLOCK_SIZE):
                result[k].append(example[i : i + BLOCK_SIZE])

    return result


def load_and_prepare_dataset(
    input_file: str,
    arabert_preprocessor: ArabertPreprocessor,
    tokenizer: transformers.GPT2TokenizerFast,
) -> datasets.DatasetDict:
    """
    تقوم هذه الدالة بتجهيز البيانات من خلال:
    - تقسيم البيانات إلى بيانات تدريب (97.5%) وبيانات اختبار (2.5%).
    - تمرير الأمثلة من خلال دالة preprocess_examples ليتم معالجتها وتجهيزها لعملية التدريب.
    - تمرير الأمثلة من خلال دالة group_examples ليتم تقسيم البيانات إلى أجزاء أصغر ليكون التدريب ممكنًا.

    ثم ترجع البيانات التي تم تجهيزها.
    """

    dataset = load_dataset('text', split='train', data_files=input_file)
    dataset = dataset.train_test_split(test_size=2.5 / 100, seed=961)

    tokenized_dataset = dataset.map(
        preprocess_examples,
        batched=True,
        num_proc=8,
        remove_columns=dataset['train'].column_names,
        fn_kwargs={
            'arabert_preprocessor': arabert_preprocessor,
            'tokenizer': tokenizer,
        },
    )

    lm_dataset = tokenized_dataset.map(group_examples, batched=True, num_proc=8)

    return lm_dataset


def train(
    args: argparse.Namespace,
    model: transformers.GPT2LMHeadModel,
    lm_dataset: datasets.DatasetDict,
    data_collator: DataCollatorForLanguageModeling,
) -> None:
    """تقوم هذه الدالة ببدء عملية تدريب النموذج بناءً على المدخلات التي تم استقبالها."""

    training_args = TrainingArguments(
        output_dir=f'generator/models/results-{args.model_size}',
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
        train_dataset=lm_dataset['train'],
        eval_dataset=lm_dataset['test'],
        data_collator=data_collator,
    )

    trainer.train(args.continue_training)
    trainer.save_model()


def main(args: argparse.Namespace) -> None:
    # تجهيز معالج البيانات الأوَّلي من AraBERT.
    arabert_preprocessor = ArabertPreprocessor(f'aragpt2-{args.model_size}')

    # تجهيز معالج بيانات النموذج.
    tokenizer = load_tokenizer(args.model_size)

    # تجهيز النموذج.
    model = load_model(args.model_size, tokenizer)

    # تجهيز البيانات.
    lm_dataset = load_and_prepare_dataset(args.input_file, arabert_preprocessor, tokenizer)

    # تجهيز معالج حزم التدريب.
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    train(args, model, lm_dataset, data_collator)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_size',
        choices=['base', 'medium', 'large', 'mega'],
        required=True,
        help='حجم النموذج الذي يجب تدريبه',
    )
    parser.add_argument(
        '--input_file',
        default='generator/data/Preprocessed Arabic Poem Comprehensive Dataset (APCD).txt',
        help='مسار البيانات',
    )
    parser.add_argument('--batch_size', type=int, default=2, help='عدد الأمثلة في حزمة التدريب الواحدة')
    parser.add_argument('--train_epochs', type=int, default=15, help='عدد مرات تدريب النموذج على كامل البيانات')
    parser.add_argument('--continue_training', action='store_true', help='بدأ التدريب من حيث توقف في آخر مرة')
    args = parser.parse_args()

    main(args)
