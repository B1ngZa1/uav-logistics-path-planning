@echo off
chcp 65001 >nul
echo ========================================
echo PDF批量提取工具
echo ========================================
echo.

cd /d "C:\University\Files\Training program\无人机项目设计\01_参考文献"

set OUTPUT_DIR=知识库\中文文献提取
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

echo 开始提取PDF文件...
echo.

set count=0
for %%f in (*.pdf) do (
    set /a count+=1
    echo [!count!] 正在处理: %%f
    mineru-open-api flash-extract "%%f" --language ch -o "%OUTPUT_DIR%\%%~nf.md"
    echo.
)

echo ========================================
echo 完成！共处理 %count% 个PDF文件
echo 输出目录: %OUTPUT_DIR%
echo ========================================
pause
