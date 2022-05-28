# شعبور

ملاحظة: كل الأوامر المذكورة في هذا الملف يجب تنفيذها من داخل المجلد الرئيسي لشعبور.

## المتطلبات

1- قم بتثبيت المكتبات المطلوبة من خلال تنفيذ الأمر التالي:

```bash
pip install -r requirements.txt
```

2- قم بتحميل مجموعة البيانات `Arabic Poem Comprehensive Dataset (APCD)` من خلال تنفيذ الأوامر التالية:

```bash
gdown 1xcb2p_TsQbexX9TIxfKMlLLcJvb-Dmly -O data/
unzip "data/Arabic Poem Comprehensive Dataset (APCD).zip" -d data/
rm "data/Arabic Poem Comprehensive Dataset (APCD).zip"
```

إذا حدثت بعض المشاكل خلال تنفيذ الأوامر السابقة، يمكنك تحميل الملف من [هذا](https://drive.google.com/open?id=1xcb2p_TsQbexX9TIxfKMlLLcJvb-Dmly) الرابط وفك الضغط عنه داخل مجلد `data`.
  
3- قم بتحميل مستودع مشروع `AraBERT` من خلال تنفيذ الأمر التالي:

```bash
git clone https://github.com/aub-mind/arabert
```

## مكونات شعبور

شعبور كبيئة عمل يتكون من عدد من المكونات المنفصلة وهي:
1. [مُؤلِّف الشعر](/generator) بناءً على كلمة، صدر بيت، بحر وقافية معينين، أو تأليف شعر عشوائي.
2. [مِشكال الشعر](/diacritizer) والذي يمكنه تشكيل الأبيات والقصائد.
3. [مُكتشف البحور والقوافي](/detector) لتحديد بحر وقافية بيت معيّن.
4. [كاتِب التفعيلات](/footer) لكتابة تفعيلات الأبيات.

لمعرفة التفاصيل الخاصة بكل مكوّن من مكونات شعبور، انتقل إلى المجلد الخاص به.
