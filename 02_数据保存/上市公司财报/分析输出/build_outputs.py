from __future__ import annotations

import csv
from pathlib import Path

OUT = Path(__file__).resolve().parent
RAW = OUT / "financials_raw.csv"
NORM = OUT / "financials_normalized.csv"
INDEX = OUT / "annual_reports_index.csv"


def row(company, ticker, period, metric, value, currency, units, source_id, source_url,
        source_file, source_page, evidence, confidence="high", note=""):
    return {
        "company": company,
        "ticker": ticker,
        "period": period,
        "metric": metric,
        "reported_value": "" if value is None else value,
        "currency": currency,
        "reported_units": units,
        "source_id": source_id,
        "source_url": source_url,
        "source_file": source_file,
        "source_page": source_page,
        "evidence_label": evidence,
        "confidence": confidence,
        "notes": note,
    }


rows = []

def add(company, ticker, years, values, currency, units, source_id, url, file, page, evidence="source_reported", confidence="high", note=""):
    for metric, vals in values.items():
        for period, value in zip(years, vals):
            rows.append(row(company, ticker, period, metric, value, currency, units, source_id, url, file, page, evidence, confidence, note))


# Currency/unit conventions are preserved exactly as reported by each issuer.
add("EHang Holdings", "EH", [2022, 2023, 2024], {
    "revenue": [44317, 117426, 456152],
    "gross_profit": [29219, 75311, 279946],
    "operating_profit": [-303950, -296252, -254075],
    "net_income": [-329331, -302341, -230032],
    "operating_cash_flow": [-173458, -88410, 157959],
    "capex_ppe": [-12166, -7832, -38966],
    "intangible_capex": [-1653, -1255, -1112],
    "rd_expense": [135082, 167315, 199465],
    "gross_margin_pct": [0.6594, 0.6413, 0.6138],
    "rd_pct_opex": [None, None, 0.354],
    "cash_and_equivalents": [None, 228250, 610877],
    "interest_bearing_debt": [None, 82644, 95250],
    "contract_liabilities": [None, 37169, 62561],
}, "RMB", "RMB thousand", "EH20F2024", "https://ir.ehang.com/static-files/4c2db5da-74de-40b2-bf45-370a30b9f901", "../原始报告/EHang_EH_2024_Form20F.pdf", "F-5/F-8/F-13", note="Debt is short-term bank loans plus current and long-term portions of bank loans; 2022 balance-sheet debt not available in the extracted scope.")

add("AeroVironment", "AVAV", [2023, 2024, 2025], {
    "revenue": [540536, 716720, 820627],
    "gross_profit": [173514, 283931, 318636],
    "operating_profit": [-178663, 71824, 40795],
    "net_income": [-176167, 59666, 43619],
    "operating_cash_flow": [11400, 15292, -1318],
    "capex_ppe_and_capitalized_software": [-14868, -22983, -22816],
    "rd_expense": [64255, 97687, 100729],
    "cash_and_equivalents": [132859, 73301, 40862],
    "interest_bearing_debt": [None, 27092, 30000],
    "funded_backlog": [None, 400200, 726600],
    "unfunded_backlog": [None, 135400, 774600],
}, "USD", "USD thousand", "AVAV10K2025", "https://investor.avinc.com/static-files/ed74debc-867f-474f-9a4f-2614dc2bd7c6", "../原始报告/AeroVironment_AVAV_2025_10K.pdf", "64/79/80/83", note="Fiscal years end April 30; funded and unfunded backlog are company disclosures in USD millions converted to USD thousand for raw-data consistency.")

add("Parrot", "PARRO", [2022, 2023, 2024], {
    "revenue": [71949, 64981, 78054],
    "gross_profit": [55996, 46977, 57869],
    "operating_profit": [-19874, -30033, -7294],
    "net_income": [-19722, -32980, -9397],
    "operating_cash_flow": [-20120, -40744, 6729],
    "capex_ppe_and_intangibles": [-1872, -774, -2249],
    "rd_expense": [-43614, -45494, -39674],
    "cash_and_equivalents": [68465, 28063, 33604],
    "debt_and_lease_liabilities": [10180, 8343, 8733],
    "gross_margin_pct": [0.778, 0.723, 0.741],
}, "EUR", "EUR thousand", "PARRO_URD", "https://www.parrot.com/assets/s3fs-public/media-public/PARROT_URD-2024_PDF.pdf", "../原始报告/Parrot_PARRO_2024_URD.pdf", "97-100", note="Debt includes current/non-current lease liabilities plus financial debt where disclosed; 2022 cross-year data is from the 2023 URD.")

add("中无人机", "688297", [2022, 2023, 2024], {
    "revenue": [2773101113.73, 2663960724.50, 685242333.64],
    "net_income_attributable_parent": [370195859.71, 302573609.66, -53916050.03],
    "operating_cash_flow": [377861790.69, 117629579.62, 139953229.81],
    "total_assets": [7807292817.74, 7362397641.85, 8728901145.10],
    "equity_attributable_parent": [5760406621.81, 5846252734.37, 5685726173.79],
    "roe_pct": [0.1069, 0.0521, -0.0094],
    "rd_pct_revenue": [0.0682, 0.0648, 0.3912],
}, "RMB", "RMB yuan", "688297_AR2024_SUM", "https://file.finance.sina.com.cn/211.154.219.97%3A9494/MRGG/CNSESH_STOCK/2025/2025-4/2025-04-01/10837522.PDF", "../原始报告/中无人机_688297_2024_年度报告_镜像.pdf", "摘要/主要会计数据", "web_verified", "medium", note="Local file is a 10-page annual-report summary mirror; gross profit, operating profit, capex, cash and debt are intentionally blank until the full official report is available.")

add("纵横股份", "688070", [2022, 2023, 2024], {
    "revenue": [287307583.88, 301763426.69, 474202644.94],
    "net_income_attributable_parent": [-26055652.94, -64499860.51, -35791711.87],
    "operating_cash_flow": [-81257911.08, -21782729.82, 60112034.30],
    "total_assets": [951008158.39, 940706916.99, 1057433297.32],
    "equity_attributable_parent": [670037000.59, 605537140.08, 569745428.21],
    "gross_margin_pct": [None, 0.4211, 0.4220],
    "rd_expense": [None, 38364464.19, 42877280.26],
    "rd_investment_pct_revenue": [0.2408, None, 0.1949],
    "contract_liabilities": [None, None, 106041102.70],
    "investing_cash_flow": [None, -69087267.08, -44552312.05],
}, "RMB", "RMB yuan", "688070_AR2024_WEB", "https://money.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?id=10891735&stockid=688070", "../原始报告/纵横股份_688070_2024_年度报告_镜像.pdf", "主要会计数据/年报摘要", "web_verified", "medium", note="Values are cross-checked against public annual-report text/mirror; the locally saved 1-page file is only a summary. Gross profit, operating profit, capex, cash and debt remain blank where the local source does not disclose them.")


raw_fields = ["company", "ticker", "period", "metric", "reported_value", "currency", "reported_units", "source_id", "source_url", "source_file", "source_page", "evidence_label", "confidence", "notes"]
with RAW.open("w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=raw_fields)
    w.writeheader()
    w.writerows(rows)


scale = {"RMB thousand": 1000, "USD thousand": 1000, "EUR thousand": 1000, "RMB yuan": 1}
norm_fields = ["company", "ticker", "period", "metric", "value_local_millions", "normalized_unit", "currency", "source_id", "source_file", "source_page", "evidence_label", "confidence", "notes"]
with NORM.open("w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=norm_fields)
    w.writeheader()
    for r in rows:
        out = dict(r)
        value = r["reported_value"]
        is_ratio = r["metric"].endswith("_pct") or r["metric"] == "gross_margin_pct"
        if value == "":
            norm = ""
        elif is_ratio:
            norm = float(value)
        elif r["reported_units"] == "RMB yuan":
            norm = float(value) / 1_000_000
        else:
            # Reported-thousand units are divided by 1,000 to become local-currency millions.
            norm = float(value) / 1000
        w.writerow({
            "company": r["company"], "ticker": r["ticker"], "period": r["period"], "metric": r["metric"],
            "value_local_millions": norm, "normalized_unit": ("ratio" if is_ratio else f"{r['currency']} million"), "currency": r["currency"],
            "source_id": r["source_id"], "source_file": r["source_file"], "source_page": r["source_page"],
            "evidence_label": r["evidence_label"], "confidence": r["confidence"], "notes": r["notes"],
        })


index_rows = [
    ["中无人机", "688297", 2024, "年度报告摘要（镜像）", "RMB", "RMB yuan", "https://static.sse.com.cn/disclosure/listedinfo/announcement/c/new/2025-04-01/688297_20250401_DPFB.pdf", "https://file.finance.sina.com.cn/211.154.219.97%3A9494/MRGG/CNSESH_STOCK/2025/2025-4/2025-04-01/10837522.PDF", "../原始报告/中无人机_688297_2024_年度报告_镜像.pdf", 10, "summary_mirror", "上交所直链被反爬挑战拦截；本地为新浪镜像摘要，不是完整年报"],
    ["纵横股份", "688070", 2024, "年度报告摘要（镜像）", "RMB", "RMB yuan", "https://static.sse.com.cn/disclosure/listedinfo/announcement/c/new/2025-04-18/688070_20250418_M0CV.pdf", "https://epaper.cs.com.cn/zgzqb/images/2025-04/18/B144/zqB14418.pdf", "../原始报告/纵横股份_688070_2024_年度报告_镜像.pdf", 1, "summary_mirror", "上交所直链被反爬挑战拦截；本地为中国证券报镜像摘要，不是完整年报"],
    ["EHang Holdings", "EH", 2024, "Form 20-F", "RMB", "RMB thousand", "https://ir.ehang.com/static-files/4c2db5da-74de-40b2-bf45-370a30b9f901", "", "../原始报告/EHang_EH_2024_Form20F.pdf", 261, "public_filing", "FY2024 ended 2024-12-31; RMB is reporting currency; official SEC/IR filing PDF"],
    ["AeroVironment", "AVAV", 2025, "Form 10-K", "USD", "USD thousand", "https://investor.avinc.com/static-files/ed74debc-867f-474f-9a4f-2614dc2bd7c6", "", "../原始报告/AeroVironment_AVAV_2025_10K.pdf", 142, "public_filing", "Fiscal year ended 2025-04-30; official IR filing PDF"],
    ["Parrot", "PARRO", 2024, "Universal Registration Document", "EUR", "EUR thousand", "https://www.parrot.com/assets/s3fs-public/media-public/PARROT_URD-2024_PDF.pdf", "", "../原始报告/Parrot_PARRO_2024_URD.pdf", 184, "public_filing", "IFRS; FY2024 ended 2024-12-31"],
    ["Parrot", "PARRO", 2023, "Universal Registration Document", "EUR", "EUR thousand", "https://www.parrot.com/assets/s3fs-public/media-public/Parrot_URD-2023.pdf", "", "../原始报告/Parrot_PARRO_2023_URD.pdf", 197, "public_filing", "Used for FY2022 comparative data"],
]
idx_fields = ["company", "ticker", "report_year", "report_type", "currency", "reported_units", "official_report_url", "mirror_or_alt_url", "local_file", "pages", "retrieval_status", "notes"]
with INDEX.open("w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(idx_fields)
    w.writerows(index_rows)

print(f"wrote {len(rows)} raw metric rows, {RAW}, {NORM}, {INDEX}")
