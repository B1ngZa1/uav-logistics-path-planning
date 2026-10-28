from pathlib import Path
import csv, re, json
from pypdf import PdfReader

ROOT = Path(r"C:\University\Files\Training program\无人机项目设计")
RAW = ROOT / "02_数据保存" / "上市公司财报" / "原始报告"
OUT = ROOT / "02_数据保存" / "上市公司财报" / "分析输出"
OUT.mkdir(parents=True, exist_ok=True)

files = [
    ("中无人机", "688297", "中无人机_688297_2024_年度报告_镜像.pdf"),
    ("纵横股份", "688070", "纵横股份_688070_2024_年度报告_镜像.pdf"),
    ("EHang", "EH", "EHang_EH_2024_Form20F.pdf"),
    ("AeroVironment", "AVAV", "AeroVironment_AVAV_2025_10K.pdf"),
    ("Parrot", "PARRO", "Parrot_PARRO_2024_URD.pdf"),
    ("Parrot", "PARRO23", "Parrot_PARRO_2023_URD.pdf"),
]

records = []
for company, ticker, fn in files:
    p = RAW / fn
    reader = PdfReader(str(p))
    text_parts = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").replace("\x00", " ")
        text_parts.append(text)
        for pattern in [
            r"(?i)consolidated statements of operations",
            r"(?i)consolidated balance sheets",
            r"(?i)consolidated statements of cash flows",
            r"(?i)consolidated income statement",
            r"(?i)consolidated balance sheet",
            r"(?i)consolidated cash flow statement",
            r"合并利润表",
            r"合并资产负债表",
            r"合并现金流量表",
            r"研发投入",
            r"research and development",
            r"capital expenditures",
            r"cash and cash equivalents",
            r"cash flow from operating activities",
        ]:
            if re.search(pattern, text):
                records.append({
                    "company": company, "ticker": ticker, "file": fn,
                    "page": i, "match": pattern,
                    "snippet": re.sub(r"\s+", " ", text[:500]),
                })
    (OUT / f"{ticker}_report_text.txt").write_text("\n\n".join(text_parts), encoding="utf-8")
    records.append({"company": company, "ticker": ticker, "file": fn, "page": len(reader.pages), "match": "__PAGE_COUNT__", "snippet": "page_count"})

with (OUT / "report_text_hits.csv").open("w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=["company", "ticker", "file", "page", "match", "snippet"])
    w.writeheader(); w.writerows(records)
print(json.dumps({"reports": len(files), "hits": len(records), "output": str(OUT)}, ensure_ascii=False))
