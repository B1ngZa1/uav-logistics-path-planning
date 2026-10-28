@echo off
echo ========================================
echo Organize Reference Literature Files
echo ========================================
echo.

cd /d "C:\University\Files\Training program\无人机项目设计\01_参考文献"

echo Creating directories...
mkdir "PDF文件\第一批" 2>nul
mkdir "PDF文件\第二批" 2>nul
mkdir "提取内容\第一批" 2>nul
mkdir "提取内容\第二批" 2>nul
mkdir "工具脚本" 2>nul

echo Moving PDF files (first batch)...
for %%f in (
    "基于HCA-MAPPO深度强化学习的多无人机协同路径规划方法_林振希.pdf"
    "基于多智能体强化学习的多无人机协同交通监控路径优化_刘晓波.pdf"
    "先验引导的时序融合多无人机协同避障航路规划方法_王傲.pdf"
    "基于双层强化学习的异构多无人机任务规划_付星桐.pdf"
    "基于多智能体强化学习的无人机对抗技术研究_孙笠晋.pdf"
    "基于多智能体强化学习的无人机集群协同综述_方伟.pdf"
    "基于多策略改进霜冰算法的多无人机协同任务分配_陈海洋.pdf"
    "基于强化学习的无人机海上搜寻路径规划_张楠.pdf"
    "基于改进MATD3强化学习算法的多无人机通感一体系统能效优化_田峰.pdf"
    "基于深度学习的多无人机任务分配和路径优化方法研究_张警伟.pdf"
    "基于深度强化学习的无人机目标跟踪和障碍规避融合控制方法_李治威.pdf"
    "基于深度强化学习的无人机箔条干扰自卫防御决策方法设计_安通.pdf"
    "基于深度强化学习的架空输电线吊运无人机上下线导航法_梁加凯.pdf"
    "多无人机分区协同区域覆盖航迹规划_黄姣茹.pdf"
    "无人机物流配送路径规划的双目标约束深度Q_网络方法研究_赵学健.pdf"
    "融合深度强化学习的自适应势场路径规划算法_胥斌.pdf"
    "面向低空经济的无人机配送路径规划综述_廖勇.pdf"
    "基于分层强化学习的无人机导航方案_郑钧洋.pdf"
    "基于多智能体的无人机轨迹与资源优化算法_熊俊.pdf"
    "基于AI辅助的无人机物流动态路径规划策略_谭斯予.pdf"
    "基于元平均场Q学习的大规模无人机组网抗干扰与隐蔽通信机制_钱玉文.pdf"
    "面向无人机集群的通信与组网技术_赵海涛.pdf"
) do (
    if exist %%f move /y %%f "PDF文件\第一批\" >nul 2>&1
)
echo Done!

echo Moving scripts...
move /y "extract_all_v2.py" "工具脚本\" >nul 2>&1
move /y "extract_theses.ps1" "工具脚本\" >nul 2>&1
move /y "extract_pdfs.bat" "工具脚本\" >nul 2>&1
move /y "extract_theses.bat" "工具脚本\" >nul 2>&1
move /y "extract_all_pdfs.bat" "工具脚本\" >nul 2>&1
move /y "run_extract.bat" "工具脚本\" >nul 2>&1
move /y "organize_files.ps1" "工具脚本\" >nul 2>&1
echo Done!

echo.
echo ========================================
echo File structure organized!
echo ========================================
pause
