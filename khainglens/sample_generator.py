"""Deterministic generator for synthetic English and Burmese sample datasets."""

import json
from pathlib import Path


def generate_sample_files(target_dir: Path | None = None) -> list[Path]:
    """Write sample_burmese.csv, sample_burmese.jsonl, and sample_english.txt to sample_data/."""
    out_dir = target_dir or (Path(__file__).resolve().parent.parent / "sample_data")
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # 1. sample_burmese.csv
    # Demonstrates: valid Kinzi/medials/stacks, exact duplicates, empty string,
    # leading/trailing spaces, repeated internal spaces, zero-width space (U+200B),
    # duplicate diacritic (U+1037 U+1037), suspected Zawgyi (U+1031 before base),
    # short/long outliers, and 1 malformed row with extra unquoted commas.
    csv_lines = [
        "id,text,category",
        '1,"မင်္ဂလာပါခင်ဗျာ။ မြန်မာဘာသာစကား အချက်အလက်များကို စနစ်တကျ စစ်ဆေးပါသည်။",greeting',
        '2,"ရန်ကုန်တက္ကသိုလ်နှင့် မန္တလေးတက္ကသိုလ်တို့တွင် သဘာဝဘာသာစကား သုတေသန ပြုလုပ်ကြသည်။",education',
        '3,"မင်္ဂလာပါခင်ဗျာ။ မြန်မာဘာသာစကား အချက်အလက်များကို စနစ်တကျ စစ်ဆေးပါသည်။",greeting',
        '4,"ပြည်ထောင်စုလွှတ်တော်တွင် ကြွေးမြီဆိုင်ရာ ဥပဒေကြမ်းကို ဆွေးနွေးခဲ့သည်။",news',
        '5,"   ဦးဆောင်မှုဖြင့် အောင်မြင်ခဲ့ကြောင်း ဖော်ပြထားသည်။  ",whitespace_lead_trail',
        '6,"မြန်မာစာ   အချက်အလက်များကို    သန့်စင်ခြင်းသည် အရေးကြီးသည်။",repeated_spaces',
        '7,"မြန်\u200bမာ\u200bဘာသာစကားတွင် သုညအကျယ် အကွာအဝေးများ ပါဝင်နေသည်။",zero_width_space',
        '8,"ဤစာကြောင်းတွင် အောက်မြစ်နှစ်ထပ် က\u102d\u102dန\u1037\u1037 မှားယွင်းနေပါသည်။",duplicate_diacritic',
        '9,"\u1031\u1019\u1010\u1039\u1010\u102c\u1016\u1033\u1004\u1039\u1037 ကူညီဆောင်ရွက်ပေးပါသည်။",suspected_zawgyi',
        "10,,empty_cell",
        '11,"   ",whitespace_only',
        '12,"တို",short_outlier',
        (
            '13,"သဘာဝဘာသာစကားပြုပြင်ထုတ်လုပ်မှု (Natural Language Processing) တွင် '
            "အချက်အလက်အရည်အသွေး စစ်ဆေးခြင်းသည် မော်ဒယ်လ်များ၏ တိကျမှန်ကန်မှုအတွက် အဓိကကျသော "
            "အဆင့်တစ်ခုဖြစ်ပြီး မြန်မာဘာသာစကားကဲ့သို့ ရှုပ်ထွေးသော အက္ခရာစဉ်များတွင် ပိုမိုအရေးပါပါသည်။ "
            'ထို့ကြောင့် စာလုံးပေါင်းနှင့် ယူနီကုဒ်စံနှုန်းများကို စနစ်တကျ စစ်ဆေးရန် လိုအပ်ပါသည်။",long_outlier'
        ),
        "14,Malformed unquoted row,with too,many columns,extra",
    ]
    csv_path = out_dir / "sample_burmese.csv"
    csv_path.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    written.append(csv_path)

    # 2. sample_burmese.jsonl
    # Demonstrates: valid Burmese, exact duplicates, NFC normalization duplicate
    # (\u1000\u1037\u103a vs \u1000\u103a\u1037), null value, empty string,
    # embedded BOM (U+FEFF), and 1 malformed JSON syntax line.
    jsonl_records = [
        json.dumps(
            {
                "id": "my_01",
                "text": "မြန်မာဘာသာစကား သုတေသနအတွက် အရည်အသွေးမြင့် ဒေတာစုစည်းမှု။",
                "source": "synthetic",
            },
            ensure_ascii=False,
        ),
        json.dumps(
            {
                "id": "my_02",
                "text": "မြန်မာဘာသာစကား သုတေသနအတွက် အရည်အသွေးမြင့် ဒေတာစုစည်းမှု။",
                "source": "synthetic_dup",
            },
            ensure_ascii=False,
        ),
        json.dumps(
            {
                "id": "my_03_nfc_canonical",
                "text": "ကျွန်ုပ်တို့၏ လုပ်ငန်းစဉ်\u1000\u1037\u103a အောင်မြင်ပါစေ။",
                "source": "nfc_order",
            },
            ensure_ascii=False,
        ),
        json.dumps(
            {
                "id": "my_04_non_nfc_order",
                "text": "ကျွန်ုပ်တို့၏ လုပ်ငန်းစဉ်\u1000\u103a\u1037 အောင်မြင်ပါစေ။",
                "source": "non_nfc_order",
            },
            ensure_ascii=False,
        ),
        json.dumps(
            {
                "id": "my_05_bom_zwsp",
                "text": "\ufeffဖိုင်ပေါင်းစပ်ရာမှ ပါဝင်လာသော \u200b အမှတ်အသား။",
                "source": "concat_artifact",
            },
            ensure_ascii=False,
        ),
        json.dumps({"id": "my_06_empty", "text": "", "source": "blank"}, ensure_ascii=False),
        json.dumps({"id": "my_07_null", "text": None, "source": "missing"}, ensure_ascii=False),
        '{"id": "my_08_broken_json", "text": unquoted_value_error}',
        json.dumps(
            {
                "id": "my_09_clean",
                "text": "ဤစာကြောင်းသည် မည်သည့်ချို့ယွင်းချက်မှ မရှိသော စံနမူနာ စာကြောင်းဖြစ်ပါသည်။",
                "source": "clean",
            },
            ensure_ascii=False,
        ),
    ]
    jsonl_path = out_dir / "sample_burmese.jsonl"
    jsonl_path.write_text("\n".join(jsonl_records) + "\n", encoding="utf-8")
    written.append(jsonl_path)

    # 3. sample_english.txt
    # Demonstrates: clean sentences, exact duplicates, empty line, whitespace-only line,
    # leading/trailing whitespace, repeated spaces, NBSP (U+00A0), ZWSP (U+200B),
    # embedded tab, short outlier, and long outlier.
    txt_lines = [
        "High-quality text corpora require systematic inspection before model training.",
        "Dataset curation prevents silent tokenization failures and train-test leakage.",
        "High-quality text corpora require systematic inspection before model training.",
        "  This sentence has leading and trailing ASCII whitespace characters.   ",
        "Multiple   consecutive    spaces often appear after OCR or HTML tag stripping.",
        "Invisible\u200bzero-width\u200bspaces can split single English words into fragments.",
        "Non-breaking\u00a0spaces\u00a0prevent normal whitespace splitting in naive pipelines.",
        "Embedded\ttab\tcharacters inside a plain text line may indicate unparsed TSV data.",
        "",
        "     ",
        "OK",
        "Clean validation benchmark sentence number one with standard spacing.",
        "Clean validation benchmark sentence number two with standard spacing.",
        "Clean validation benchmark sentence number three with standard spacing.",
        (
            "An unusually long paragraph concatenated onto a single physical line can skew "
            "sequence length distributions, exhaust transformer context windows during batching, "
            "and indicate that newline delimiters were accidentally stripped during upstream "
            "document ingestion or PDF text extraction workflows."
        ),
    ]
    txt_path = out_dir / "sample_english.txt"
    txt_path.write_text("\n".join(txt_lines) + "\n", encoding="utf-8")
    written.append(txt_path)

    return written


if __name__ == "__main__":
    for path in generate_sample_files():
        print(f"Generated: {path.relative_to(Path.cwd()) if path.is_relative_to(Path.cwd()) else path}")