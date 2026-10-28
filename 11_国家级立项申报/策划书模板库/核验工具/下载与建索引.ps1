# ASCII-only repeatable checker for the downloaded library.
$ErrorActionPreference = 'Stop'
$libraryRoot = Split-Path $PSScriptRoot -Parent
$indexPath = (Get-ChildItem -LiteralPath $libraryRoot -File | Where-Object { $_.Extension -eq '.csv' } | Select-Object -First 1).FullName
if ([string]::IsNullOrWhiteSpace($indexPath)) { throw 'CSV index is missing.' }

$rows = @(Get-Content -LiteralPath $indexPath -Raw -Encoding UTF8 | ConvertFrom-Csv)
$names = @($rows[0].PSObject.Properties.Name)
$relativeColumn = $names[2]
$hashColumn = $names[19]
$missing = @()
$mismatch = @()
foreach ($row in $rows) {
    $filePath = Join-Path $libraryRoot $row.$relativeColumn
    if (!(Test-Path -LiteralPath $filePath)) {
        $missing += $row.$relativeColumn
        continue
    }
    $actual = (Get-FileHash -LiteralPath $filePath -Algorithm SHA256).Hash
    if ($actual -ne $row.$hashColumn) {
        $mismatch += ($row.$relativeColumn + ' expected=' + $row.$hashColumn + ' actual=' + $actual)
    }
}

Write-Output ('Rows: ' + $rows.Count)
Write-Output ('Missing: ' + $missing.Count)
Write-Output ('SHA256 mismatches: ' + $mismatch.Count)
if ($missing.Count -gt 0) { $missing | ForEach-Object { Write-Output ('MISSING ' + $_) } }
if ($mismatch.Count -gt 0) { $mismatch | ForEach-Object { Write-Output ('MISMATCH ' + $_) } }
if ($missing.Count -gt 0 -or $mismatch.Count -gt 0) { exit 2 }

$verifier = (Get-ChildItem -LiteralPath $PSScriptRoot -Filter '*.py' -File | Select-Object -First 1).FullName
python $verifier
