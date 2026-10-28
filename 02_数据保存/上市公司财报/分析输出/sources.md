# 财报与市场分析来源

## 财报来源

1. 中无人机（688297）官方完整年报地址：`https://static.cninfo.com.cn/finalpage/2025-04-01/1222974842.PDF`；实测直链可响应但下载极慢，约 3.8 MB 文件在约 67 秒仅完成约 340 KB，未保留不完整文件。官方检索结果显示完整年报约 227 页。本地可打开文件 `../原始报告/中无人机_688297_2024_年度报告_镜像.pdf` 是 10 页摘要，仍未作为完整年报底稿。新浪镜像摘要是当前可复现替代入口；另一个已检索的 100est 链接 `https://www.100est.com/res/financial-report/r2024/SH688297_202503311649676289.pdf` 直接下载返回 404，未作为底稿。
   - 上交所公告入口 `https://static.sse.com.cn/disclosure/listedinfo/announcement/c/new/2025-04-01/688297_20250401_DPFB.pdf` 及中商产业研究院页面 `https://s.askci.com/stock/financialreport/688297/financefile/` 均可定位到完整年报；上交所直链实测 HTTP 200 但 `Content-Type: text/html` 且 `X-Tengine-Error: denied by bot`，临时响应 3,872 bytes 非 PDF，pypdf 失败。中商页面跳转的同花顺 PDF `https://notice.10jqka.com.cn/api/pdf/ab3bcf6984c12b0.pdf` 实测约 60 秒仅下载约 80 KB，未通过 PDF 魔数/页数校验，已删除不完整文件。
2. 纵横股份（688070）官方完整年报：`https://static.cninfo.com.cn/finalpage/2025-04-18/1223128738.PDF`；备用镜像：`https://dataclouds.cninfo.com.cn/shgonggao/2025/2025-04-18/ed86c1cd1b8311f084cdfa163e957f7a.pdf`。本地文件 `../原始报告/纵横股份_688070_2024_年度报告_完整.pdf` 共259页，财务报表使用第7、11、120–129页。
3. EHang Holdings 2024 Form 20-F：`https://ir.ehang.com/static-files/4c2db5da-74de-40b2-bf45-370a30b9f901`；本地文件 `../原始报告/EHang_EH_2024_Form20F.pdf`，261页。财务报表使用 F-5、F-8、F-13 等报告页码；另保存官方 20-F HTML 归档。
4. AeroVironment 2025 Form 10-K：`https://investor.avinc.com/static-files/ed74debc-867f-474f-9a4f-2614dc2bd7c6`；本地文件 `../原始报告/AeroVironment_AVAV_2025_10K.pdf`，142 页。财务报表页码约为 79–83，MD&A/订单信息见 64 页及公司年报披露。
5. Parrot 2024 URD：`https://www.parrot.com/assets/s3fs-public/media-public/PARROT_URD-2024_PDF.pdf`；本地文件 `../原始报告/Parrot_PARRO_2024_URD.pdf`，184 页。合并利润表/资产负债表/现金流量表集中在报告财务报表章节。
6. Parrot 2023 URD：`https://www.parrot.com/assets/s3fs-public/media-public/Parrot_URD-2023.pdf`；本地文件 `../原始报告/Parrot_PARRO_2023_URD.pdf`，197 页。用于补取 2022 比较期。

## 市场与行业来源

- 中国民航局无人机/低空运行公开统计：`https://www.caac.gov.cn/PHONE/XWZX/MHYW/202405/t20240531_224334.html`。
- 低空经济市场数据和来源分类见 `../低空经济市场/source_register.csv`、`../低空经济市场/market_data_normalized.csv`。各条记录区分官方统计、政策、公司披露、行业估算和推断。

## 证据与单位规则

- `financials_raw.csv` 保留报告单位：RMB yuan、RMB thousand、USD thousand 或 EUR thousand。
- `financials_normalized.csv` 只做本币百万缩放；不做人民币、美元、欧元之间的汇率换算。
- 空值是有意保留的质量标记：来源没有披露，或本地 A 股文件只有摘要，不以估算替代原始数据。
