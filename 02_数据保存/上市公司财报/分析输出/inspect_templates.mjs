import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const files = [
  ["three", "C:/Users/14012/.codex/plugins/cache/openai-curated-remote/openai-templates/0.1.1/skills/artifact-template-three-statement-forecast/assets/reference.xlsx"],
  ["pipeline", "C:/Users/14012/.codex/plugins/cache/openai-curated-remote/openai-templates/0.1.1/skills/artifact-template-sales-pipeline/assets/reference.xlsx"],
];

for (const [name, path] of files) {
  const wb = await SpreadsheetFile.importXlsx(await FileBlob.load(path));
  const summary = await wb.inspect({ kind: "workbook,sheet,table", maxChars: 12000, tableMaxRows: 5, tableMaxCols: 12 });
  console.log(`---${name}---`);
  console.log(summary.ndjson);
}
