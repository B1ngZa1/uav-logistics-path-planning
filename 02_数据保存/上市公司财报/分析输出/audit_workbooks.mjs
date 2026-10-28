import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

for (const [name, path] of [
  ["three", "C:/University/Files/Training program/无人机项目设计/02_数据保存/上市公司财报/分析输出/three_statement_forecast.xlsx"],
  ["pipeline", "C:/University/Files/Training program/无人机项目设计/02_数据保存/上市公司财报/分析输出/market_opportunity_pipeline.xlsx"]
]) {
  const wb = await SpreadsheetFile.importXlsx(await FileBlob.load(path));
  const errors = await wb.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!", options: { useRegex: true, maxResults: 300 }, summary: `${name} formula errors` });
  console.log(`${name}_ERROR_SCAN=${errors.ndjson}`);
  const range = name === "three" ? "A35:F37" : "B14:O23";
  const sheet = name === "three" ? "Project Forecast" : "Sales Pipeline";
  const check = await wb.inspect({ kind: "table", sheetId: sheet, range, include: "values,formulas", tableMaxRows: 12, tableMaxCols: 14, maxChars: 9000 });
  console.log(`${name}_CHECK=${check.ndjson}`);
}
