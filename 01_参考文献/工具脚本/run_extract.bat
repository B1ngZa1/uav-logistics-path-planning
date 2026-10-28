@echo off
setlocal enabledelayedexpansion
echo ========================================
echo PDF Batch Extract Tool
echo ========================================
echo.

cd /d "C:\University\Files\Training program\无人机项目设计\01_参考文献"

if not exist "knowledge_base" mkdir "knowledge_base"

echo Starting PDF extraction...
echo.

set count=0
for %%f in (*.pdf) do (
    set /a count+=1
    echo [!count!] Processing: %%f
    mineru-open-api flash-extract "%%f" --language ch -o "knowledge_base\%%~nf.md"
    echo.
)

echo ========================================
echo Done! Processed !count! PDF files
echo Output: knowledge_base\
echo ========================================
pause
