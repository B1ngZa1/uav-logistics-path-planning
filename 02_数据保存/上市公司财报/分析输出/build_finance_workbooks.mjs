import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const outputDir = "C:/University/Files/Training program/无人机项目设计/02_数据保存/上市公司财报/分析输出";
const threeReference = "C:/Users/14012/.codex/plugins/cache/openai-curated-remote/openai-templates/0.1.1/skills/artifact-template-three-statement-forecast/assets/reference.xlsx";
const pipelineReference = "C:/Users/14012/.codex/plugins/cache/openai-curated-remote/openai-templates/0.1.1/skills/artifact-template-sales-pipeline/assets/reference.xlsx";

const dark = "#1F4E78";
const mid = "#D9EAF7";
const inputFill = "#FFF2CC";
const light = "#F3F6F9";

function formatHeader(range) {
  range.format = {
    fill: dark,
    font: { bold: true, color: "#FFFFFF" },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
  };
  range.format.borders = { preset: "all", style: "thin", color: "#B7C9D6" };
}

function formatSection(range) {
  range.format = { fill: mid, font: { bold: true, color: "#1F2937" } };
  range.format.borders = { preset: "all", style: "thin", color: "#D9E2F3" };
}

async function buildThreeStatement() {
  const wb = await SpreadsheetFile.importXlsx(await FileBlob.load(threeReference));
  const readme = wb.worksheets.getItem("Read Me");
  readme.getRange("B2").values = [["Low-Altitude Economy Three-Statement Forecast"]];
  readme.getRange("B4").values = [["Parrot 2024A base | 2025E–2027E project scenario | EUR millions"]];
  readme.getRange("B6").values = [["Purpose"]];
  readme.getRange("C6").values = [["Source-backed Parrot 2024 historical base with transparent project-level forecast assumptions; retained template schedules are reference structure only."]];
  readme.getRange("B7").values = [["Historical source"]];
  readme.getRange("C7").values = [["Parrot 2024 Universal Registration Document, consolidated statements and cash flow statement."]];
  readme.getRange("B8").values = [["Forecast status"]];
  readme.getRange("C8").values = [["Scenario analysis, not company guidance or an investment recommendation."]];

  const forecast = wb.worksheets.add("Project Forecast");
  forecast.showGridLines = false;
  forecast.getRange("A1:F1").merge();
  forecast.getRange("A1").values = [["Parrot / Low-Altitude Economy Project Forecast"]];
  forecast.getRange("A1:F1").format = { fill: dark, font: { bold: true, color: "#FFFFFF", size: 14 }, horizontalAlignment: "left" };
  forecast.getRange("A2:F2").merge();
  forecast.getRange("A2").values = [["Consolidated Parrot 2024 actuals with project scenario forecast; EUR millions unless stated"]];
  forecast.getRange("A4:F4").values = [["Metric", "2024A", "2025E", "2026E", "2027E", "Units / source note"]];
  formatHeader(forecast.getRange("A4:F4"));

  const labels = [
    ["INCOME STATEMENT"],
    ["Revenue"], ["Revenue growth"], ["Cost of sales"], ["Gross profit"], ["Gross margin"],
    ["Research & development"], ["Sales & marketing"], ["Other operating expenses"], ["Operating income"], ["Operating margin"],
    ["Net interest / other finance"], ["Taxes"], ["Net income"],
    ["CASH FLOW"], ["Depreciation & amortization"], ["Change in working capital (cash contribution)"], ["Other non-cash / CFO items"], ["Operating cash flow"], ["Capital expenditure"], ["Financing cash flow"], ["FX / other cash movement"], ["Free cash flow"], ["Net change in cash"], ["Ending cash"],
    ["BALANCE SHEET BRIDGE"], ["Non-cash assets"], ["Financial and lease debt"], ["Other liabilities"], ["Equity"], ["Total assets"], ["Total liabilities + equity"], ["Balance sheet check"],
    ["FORECAST ASSUMPTIONS"], ["Revenue growth"], ["Gross margin"], ["R&D % revenue"], ["Sales & marketing % revenue"], ["Other operating expense % revenue"], ["Net interest % revenue"], ["Cash tax rate"], ["D&A % revenue"], ["Change in NWC % revenue"], ["Other CFO % revenue"], ["Capex % revenue"], ["Financing cash flow"], ["FX / other cash movement"],
  ];
  forecast.getRange(`A5:A${4 + labels.length}`).values = labels;
  for (const row of [5, 19, 30, 38]) formatSection(forecast.getRange(`A${row}:F${row}`));

  const actual = {
    6: 78.054, 7: null, 8: -20.184, 9: 57.869, 10: 0.741,
    11: -39.674, 12: -10.390, 13: -15.099, 14: -7.294, 15: -0.0934,
    16: -1.191, 17: -0.912, 18: -9.397,
    20: 1.877, 21: 12.107, 22: 2.142, 23: 6.729, 24: -2.249, 25: -1.084, 26: 0.298, 27: 4.480, 28: 5.244, 29: 33.604,
    31: 51.806, 32: 8.733, 33: 26.932, 34: 49.745, 35: 85.410, 36: 85.410, 37: 0,
  };
  for (const [row, value] of Object.entries(actual)) forecast.getRange(`B${row}`).values = [[value]];
  forecast.getRange("F6:F18").values = [
    ["EUR m; Parrot consolidated 2024"], ["Derived"], ["EUR m; cost of sales"], ["EUR m; reported gross profit"], ["Reported gross margin"],
    ["EUR m; reported R&D"], ["EUR m; reported S&M"], ["EUR m; includes net other operating items to reconcile reported operating income"], ["EUR m; reported operating result"], ["Derived"], ["EUR m; finance/equity-investment/tax bridge"], ["EUR m; reported tax"], ["EUR m; reported net income"],
  ];
  forecast.getRange("F20:F29").values = [
    ["EUR m; depreciation and impairment"], ["EUR m; positive cash contribution from BFR"], ["EUR m; residual reported CFO bridge"], ["EUR m; reported CFO"], ["EUR m; reported acquisitions of PPE/intangibles"], ["EUR m; reported financing cash flow"], ["EUR m; reported FX effect"], ["EUR m; CFO + capex"], ["EUR m; CFO + capex + financing + FX"], ["EUR m; reported closing cash"],
  ];
  forecast.getRange("F31:F37").values = [["EUR m; assets excluding cash"], ["EUR m; debt + leases"], ["EUR m; residual operating liabilities"], ["EUR m; reported equity"], ["EUR m; reported total assets"], ["EUR m; formula"], ["EUR m; must equal zero"]];

  const assumptions = [
    ["", "", 0.20, 0.15, 0.12],
    ["", "", 0.740, 0.745, 0.750],
    ["", "", 0.440, 0.420, 0.400],
    ["", "", 0.130, 0.125, 0.120],
    ["", "", 0.200, 0.190, 0.180],
    ["", "", 0.010, 0.009, 0.009],
    ["", "", 0.210, 0.210, 0.210],
    ["", "", 0.024, 0.024, 0.024],
    ["", "", 0.020, 0.020, 0.020],
    ["", "", 0.010, 0.010, 0.010],
    ["", "", 0.030, 0.030, 0.030],
    ["", "", -1.000, -1.000, -1.000],
    ["", "", 0.000, 0.000, 0.000],
  ];
  forecast.getRange("C39:E51").values = assumptions.map(row => [row[2], row[3], row[4]]);
  forecast.getRange("F39:F51").values = [
    ["Project scenario input; editable"], ["Project scenario input; editable"], ["Project scenario input; editable"], ["Project scenario input; editable"], ["Project scenario input; editable"], ["Project scenario input; editable"], ["Project scenario input; editable"], ["Project scenario input; editable"], ["Project scenario input; editable"], ["Project scenario input; editable"], ["Project scenario input; editable"], ["Project scenario input; editable"], ["Project scenario input; editable"],
  ];
  forecast.getRange("C39:E51").format.fill = inputFill;

  // Forecast formulas. The historical column remains source data; the forecast columns are formula-driven.
  forecast.getRange("C6").formulas = [["=B6*(1+C39)"]]; forecast.getRange("C6:E6").fillRight();
  forecast.getRange("C7").formulas = [["=C6/B6-1"]]; forecast.getRange("C7:E7").fillRight();
  forecast.getRange("C8").formulas = [["=-C6*(1-C40)"]]; forecast.getRange("C8:E8").fillRight();
  forecast.getRange("C9").formulas = [["=C6+C8"]]; forecast.getRange("C9:E9").fillRight();
  forecast.getRange("C10").formulas = [["=C9/C6"]]; forecast.getRange("C10:E10").fillRight();
  forecast.getRange("C11").formulas = [["=-C6*C41"]]; forecast.getRange("C11:E11").fillRight();
  forecast.getRange("C12").formulas = [["=-C6*C42"]]; forecast.getRange("C12:E12").fillRight();
  forecast.getRange("C13").formulas = [["=-C6*C43"]]; forecast.getRange("C13:E13").fillRight();
  forecast.getRange("C14").formulas = [["=SUM(C9,C11:C13)"]]; forecast.getRange("C14:E14").fillRight();
  forecast.getRange("C15").formulas = [["=C14/C6"]]; forecast.getRange("C15:E15").fillRight();
  forecast.getRange("C16").formulas = [["=-C6*C44"]]; forecast.getRange("C16:E16").fillRight();
  forecast.getRange("C17").formulas = [["=-MAX(0,C14+C16)*C45"]]; forecast.getRange("C17:E17").fillRight();
  forecast.getRange("C18").formulas = [["=SUM(C14,C16:C17)"]]; forecast.getRange("C18:E18").fillRight();
  forecast.getRange("C21").formulas = [["=C6*C47"]]; forecast.getRange("C21:E21").fillRight();
  forecast.getRange("C22").formulas = [["=C6*C48"]]; forecast.getRange("C22:E22").fillRight();
  forecast.getRange("C23").formulas = [["=SUM(C18,C20:C22)"]]; forecast.getRange("C23:E23").fillRight();
  forecast.getRange("C24").formulas = [["=-C6*C49"]]; forecast.getRange("C24:E24").fillRight();
  forecast.getRange("C25").formulas = [["=C50"]]; forecast.getRange("C25:E25").fillRight();
  forecast.getRange("C26").formulas = [["=C51"]]; forecast.getRange("C26:E26").fillRight();
  forecast.getRange("C27").formulas = [["=C23+C24"]]; forecast.getRange("C27:E27").fillRight();
  forecast.getRange("C28").formulas = [["=SUM(C23:C26)"]]; forecast.getRange("C28:E28").fillRight();
  forecast.getRange("C29").formulas = [["=B29+C28"]]; forecast.getRange("C29:E29").fillRight();
  forecast.getRange("C31").formulas = [["=B31-C24-C20-C21"]]; forecast.getRange("C31:E31").fillRight();
  forecast.getRange("C32").formulas = [["=B32+C25"]]; forecast.getRange("C32:E32").fillRight();
  forecast.getRange("C33").formulas = [["=B33+C22+C26"]]; forecast.getRange("C33:E33").fillRight();
  forecast.getRange("C34").formulas = [["=B34+C18"]]; forecast.getRange("C34:E34").fillRight();
  forecast.getRange("C35").formulas = [["=C29+C31"]]; forecast.getRange("C35:E35").fillRight();
  forecast.getRange("C36").formulas = [["=SUM(C32:C34)"]]; forecast.getRange("C36:E36").fillRight();
  forecast.getRange("C37").formulas = [["=C35-C36"]]; forecast.getRange("C37:E37").fillRight();

  forecast.getRange("B6:E38").format.numberFormat = [["#,##0.000"]];
  forecast.getRange("B7:E7").format.numberFormat = [["0.0%"]];
  forecast.getRange("B10:E10").format.numberFormat = [["0.0%"]];
  forecast.getRange("B15:E15").format.numberFormat = [["0.0%"]];
  forecast.getRange("C39:E49").format.numberFormat = [["0.0%"]];
  forecast.getRange("C50:E51").format.numberFormat = [["#,##0.000"]];
  forecast.getRange("A4:F51").format.borders = { preset: "all", style: "thin", color: "#D9E2F3" };
  forecast.getRange("A1:F53").format.font = { name: "Aptos", size: 10 };
  forecast.getRange("A1:F1").format.font = { bold: true, color: "#FFFFFF", size: 14 };
  forecast.getRange("A1:F51").format.wrapText = true;
  forecast.getRange("A:A").format.columnWidth = 34;
  forecast.getRange("B:E").format.columnWidth = 14;
  forecast.getRange("F:F").format.columnWidth = 48;
  forecast.freezePanes.freezeRows(4);

  const sources = wb.worksheets.add("Sources & Audit");
  sources.showGridLines = false;
  sources.getRange("A1:E1").merge();
  sources.getRange("A1").values = [["Sources and Audit Notes"]];
  sources.getRange("A1:E1").format = { fill: dark, font: { bold: true, color: "#FFFFFF", size: 14 } };
  sources.getRange("A3:E3").values = [["Source ID", "Company / dataset", "Period", "Metric use", "URL / local evidence"]];
  formatHeader(sources.getRange("A3:E3"));
  sources.getRange("A4:E8").values = [
    ["P01", "Parrot", "FY2024", "Consolidated income statement, balance sheet and cash flow", "https://www.parrot.com/assets/s3fs-public/media-public/PARROT_URD-2024_PDF.pdf"],
    ["P02", "Parrot", "FY2024", "Official financial reports landing page", "https://www.parrot.com/fr/corporate/rapports-financiers-annuels-et-semestriels"],
    ["M01", "China CAAC", "2024–2025", "Drone registrations, flight hours and low-altitude service stations", "https://www.caac.gov.cn/PHONE/XWZX/MHYW/202405/t20240531_224334.html"],
    ["M02", "EHang", "FY2024", "Company disclosure used in market notes; local PDF retrieval timed out", "https://ir.ehang.com/static-files/821db5e5-327d-4b5f-8b48-540557afcb2b"],
    ["L01", "Local evidence", "2026-09-20", "Local downloaded report files and pypdf page checks", "../原始报告/Parrot_PARRO_2024_URD.pdf; ../原始报告/中无人机_688297_2024_年度报告_镜像.pdf; ../原始报告/纵横股份_688070_2024_年度报告_镜像.pdf"],
  ];
  sources.getRange("A10:E15").values = [
    ["Audit item", "Result", "Interpretation", "Action", "Owner"],
    ["Historical currency", "EUR m", "Parrot consolidated statements use EUR thousands; workbook converts to EUR millions", "Keep units visible", "Project analyst"],
    ["2024 cash-flow bridge", "6.729 OCF / -2.249 capex / -1.084 financing / 0.298 FX", "Matches the consolidated cash-flow statement", "Reconcile before forecast changes", "Project analyst"],
    ["Forecast basis", "2025E–2027E scenario", "Not company guidance; editable assumptions are yellow", "Stress test growth, margins, NWC and capex", "Project analyst"],
    ["Balance check", "Formula row 38", "Cash + non-cash assets equals debt + other liabilities + equity", "Must equal zero", "Project analyst"],
    ["Coverage limitation", "Parrot complete; Chinese mirror files incomplete", "Chinese local PDFs are retained as mirror/summary evidence, not treated as full reports", "Replace with full official filings before investment use", "Project analyst"],
  ];
  formatHeader(sources.getRange("A10:E10"));
  sources.getRange("A3:E15").format.borders = { preset: "all", style: "thin", color: "#D9E2F3" };
  sources.getRange("A1:E15").format.wrapText = true;
  sources.getRange("A:A").format.columnWidth = 18; sources.getRange("B:B").format.columnWidth = 25; sources.getRange("C:C").format.columnWidth = 22; sources.getRange("D:D").format.columnWidth = 52; sources.getRange("E:E").format.columnWidth = 80;
  sources.freezePanes.freezeRows(3);

  const errors = await wb.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!", options: { useRegex: true, maxResults: 300 }, summary: "three-statement formula error scan" });
  console.log("THREE_ERRORS", errors.ndjson);
  const check = await wb.inspect({ kind: "table", sheetId: "Project Forecast", range: "A31:F37", include: "values,formulas", tableMaxRows: 10, tableMaxCols: 6, maxChars: 5000 });
  console.log("THREE_CHECK", check.ndjson);
  const preview = await wb.render({ sheetName: "Project Forecast", range: "A1:F51", scale: 1, format: "png" });
  await fs.writeFile(`${outputDir}/three_statement_forecast_preview.png`, new Uint8Array(await preview.arrayBuffer()));
  const xlsx = await SpreadsheetFile.exportXlsx(wb);
  await xlsx.save(`${outputDir}/three_statement_forecast.xlsx`);
}

async function buildPipeline() {
  const wb = await SpreadsheetFile.importXlsx(await FileBlob.load(pipelineReference));
  const sheet = wb.worksheets.getItem("Sales Pipeline");
  sheet.getRange("B2").values = [["Low-Altitude Economy Opportunity Pipeline"]];
  sheet.getRange("B3").values = [["Illustrative research cases; not contracted sales or company guidance"]];
  sheet.getRange("B5").values = [["TOTAL OPEN OPPORTUNITIES"]];
  sheet.getRange("E5").values = [["WEIGHTED OPEN OPPORTUNITIES"]];
  sheet.getRange("H5").values = [["HORIZON OPPORTUNITY"]];
  sheet.getRange("K5").values = [["PIPELINE COVERAGE"]];
  sheet.getRange("S4").values = [[new Date("2026-09-20")]];
  sheet.getRange("R6").values = [["Scenario target"]];
  sheet.getRange("S6").values = [[15000000]];
  const rows = [
    ["EHang / EH216-S tourism and UAM pilot", "Research", "Technical Validation", 24000000, "Best Case", new Date("2027-03-31"), 22, "Verify route, operator, certification and unit-economics evidence", "EHang 2024 20-F", "UAM", "High"],
    ["中无人机 / emergency communications and weather missions", "Research", "Qualification", 18000000, "Pipeline", new Date("2027-06-30"), 18, "Validate public-sector procurement, payload and delivery schedule", "中无人机 2024 summary", "Public Sector", "High"],
    ["纵横股份 / energy and pipeline inspection fleet", "Research", "Solution Fit", 12000000, "Best Case", new Date("2027-04-30"), 20, "Confirm inspection workflow, fleet uptime and service attach", "纵横股份 2024 summary", "Industrial", "Medium"],
    ["AeroVironment / allied UAS modernization", "Research", "Proposal", 50000000, "Commit", new Date("2027-09-30"), 35, "Reconcile funded backlog, delivery capacity and working-capital load", "AVAV FY2025 10-K", "Defense", "High"],
    ["Parrot / public-safety microdrone deployment", "Research", "Qualification", 8000000, "Pipeline", new Date("2027-05-31"), 16, "Test product/mission fit, procurement path and support cost", "Parrot 2024 URD", "Public Safety", "Medium"],
    ["Parrot / Pix4D mapping and digital-twin services", "Research", "Negotiation", 6000000, "Commit", new Date("2027-04-30"), 28, "Validate software attach, retention and gross-margin durability", "Parrot 2024 URD", "Software", "Medium"],
    ["纵横股份 / low-altitude service and industrial data", "Research", "Discovery", 10000000, "Pipeline", new Date("2027-10-31"), 9, "Map customer approvals, data rights and recurring-service scope", "纵横股份 2024 summary", "Services", "High"],
    ["EHang / UAM route operations ecosystem", "Research", "Discovery", 30000000, "Best Case", new Date("2027-12-31"), 11, "Separate aircraft, operations, infrastructure and insurance economics", "EHang 2024 20-F", "UAM", "High"],
  ];
  sheet.getRange("B24:O114").clear({ applyTo: "contents" });
  sheet.getRange("B16:E24").values = rows.map(r => r.slice(0, 4));
  sheet.getRange("G16:I24").values = rows.map(r => [r[4], r[5], r[6]]);
  sheet.getRange("K16:N24").values = rows.map(r => [r[7], r[8], r[9], r[10]]);
  sheet.getRange("F16").formulas = [["=IF(B16=\"\",\"\",IFERROR(INDEX($R$11:$R$18,MATCH(D16,$Q$11:$Q$18,0)),0))"]]; sheet.getRange("F16:F114").fillDown();
  sheet.getRange("J16").formulas = [["=IF(B16=\"\",\"\",IF(OR(D16=\"Closed Won\",D16=\"Closed Lost\"),\"Closed\",IF(OR(I16>$S$7,H16<$S$4),\"STALE\",\"Current\")))"]]; sheet.getRange("J16:J114").fillDown();
  sheet.getRange("O16").formulas = [["=IF(B16=\"\",\"\",E16*F16)"]]; sheet.getRange("O16:O114").fillDown();
  sheet.getRange("B14").values = [["Eight company-linked research opportunities. Deal sizes are illustrative RMB notionals, not disclosed contracts, backlog or company sales guidance."]];
  sheet.getRange("B16:O23").format.rowHeight = 42;
  const sources = wb.worksheets.add("Sources");
  sources.showGridLines = false;
  sources.getRange("A1:D1").merge(); sources.getRange("A1").values = [["Pipeline Source Register"]];
  sources.getRange("A1:D1").format = { fill: dark, font: { bold: true, color: "#FFFFFF", size: 14 } };
  sources.getRange("A3:D3").values = [["Source", "Use", "URL", "Caveat"]]; formatHeader(sources.getRange("A3:D3"));
  sources.getRange("A4:D9").values = [
    ["EHang", "FY2024 revenue, delivery and financial disclosures", "https://ir.ehang.com/static-files/4c2db5da-74de-40b2-bf45-370a30b9f901", "Company filing; opportunity values are not guidance"],
    ["AeroVironment", "FY2025 backlog, financials and cash-flow evidence", "https://investor.avinc.com/static-files/ed74debc-867f-474f-9a4f-2614dc2bd7c6", "Official 10-K; opportunity values are not backlog"],
    ["Parrot", "FY2024 consolidated financials and product context", "https://www.parrot.com/assets/s3fs-public/media-public/PARROT_URD-2024_PDF.pdf", "Official URD; opportunity values are illustrative"],
    ["中无人机", "FY2024 revenue, CFO and R&D ratio", "https://file.finance.sina.com.cn/211.154.219.97%3A9494/MRGG/CNSESH_STOCK/2025/2025-4/2025-04-01/10837522.PDF", "Summary mirror; full report required before decision use"],
    ["纵横股份", "FY2024 revenue, CFO, contract liabilities and R&D", "https://money.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?id=10891735&stockid=688070", "Public annual-report text; local PDF is only a summary mirror"],
    ["CAAC", "Drone registrations, flight hours and low-altitude service stations", "https://www.caac.gov.cn/PHONE/XWZX/MHYW/202405/t20240531_224334.html", "Official statistics; not revenue or opportunity size"],
  ];
  sources.getRange("A3:D9").format.borders = { preset: "all", style: "thin", color: "#D9E2F3" }; sources.getRange("A1:D9").format.wrapText = true;
  sources.getRange("A:A").format.columnWidth = 22; sources.getRange("B:B").format.columnWidth = 42; sources.getRange("C:C").format.columnWidth = 76; sources.getRange("D:D").format.columnWidth = 58;
  sources.freezePanes.freezeRows(3);
  const errors = await wb.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!", options: { useRegex: true, maxResults: 300 }, summary: "pipeline formula error scan" });
  console.log("PIPE_ERRORS", errors.ndjson);
  const check = await wb.inspect({ kind: "table", sheetId: "Sales Pipeline", range: "B5:O24", include: "values,formulas", tableMaxRows: 25, tableMaxCols: 14, maxChars: 9000 });
  console.log("PIPE_CHECK", check.ndjson);
  const preview = await wb.render({ sheetName: "Sales Pipeline", range: "A1:S24", scale: 1, format: "png" });
  await fs.writeFile(`${outputDir}/market_opportunity_pipeline_preview.png`, new Uint8Array(await preview.arrayBuffer()));
  const xlsx = await SpreadsheetFile.exportXlsx(wb);
  await xlsx.save(`${outputDir}/market_opportunity_pipeline.xlsx`);
}

await fs.mkdir(outputDir, { recursive: true });
await buildThreeStatement();
await buildPipeline();
