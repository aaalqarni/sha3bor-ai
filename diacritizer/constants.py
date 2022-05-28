from pyarabic.araby import FATHA, DAMMA, KASRA, FATHATAN, DAMMATAN, KASRATAN, SUKUN, SHADDA


BLOCK_SIZE = 256


# ثابت لتحويل فئات التشكيل المختلفة إلى أرقام.
TASHKEEL_TO_ID = {
	'': 0,
	FATHA: 1,
	DAMMA: 2,
	KASRA: 3,
	FATHATAN: 4,
	DAMMATAN: 5,
	KASRATAN: 6,
	SUKUN: 7,
	SHADDA: 8,
	SHADDA + FATHA: 9,
	SHADDA + DAMMA: 10,
	SHADDA + KASRA: 11,
	SHADDA + FATHATAN: 12,
	SHADDA + DAMMATAN: 13,
	SHADDA + KASRATAN: 14,
}


# ثابت يعكس تحويل فئات التشكيل المختلفة إلى أرقام.
ID_TO_TASHKEEL = { v: k for k, v in TASHKEEL_TO_ID.items() }
