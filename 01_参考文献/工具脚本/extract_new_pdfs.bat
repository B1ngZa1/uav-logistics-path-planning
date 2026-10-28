@echo off
setlocal enabledelayedexpansion
echo ========================================
echo Extract New 30 PDF Files
echo ========================================
echo.

cd /d "C:\University\Files\Training program\无人机项目设计\01_参考文献"

if not exist "knowledge_base_new" mkdir "knowledge_base_new"

echo Starting PDF extraction...
echo.

REM New PDFs (30 files)
set files[1]=多无人机类脑智能决策与协同控制方法研究_赵欣怡.pdf
set files[2]=无人机集群编队自主规划与控制技术研究_张博渊.pdf
set files[3]=基于强化学习的无人机智能空战决策研究_杨啟明.pdf
set files[4]=四旋翼无人机安全控制问题研究_金涛.pdf
set files[5]=生物启发的多无人机协同四维航迹规划方法研究_杨祖强.pdf
set files[6]=机器人深度强化学习控制方法研究_王远大.pdf
set files[7]=基于深度强化学习的旋翼无人机敏捷飞行控制与运动规划_陈时雨.pdf
set files[8]=基于深度强化学习的目标跟踪算法研究_杨宇翔.pdf
set files[9]=基于深度强化学习的认知物联网资源分配研究_郭少艾.pdf
set files[10]=无人机通信系统的无线资源智能管理技术研究_周世阳.pdf
set files[11]=无人机集群编队及其避障控制关键技术研究_赵伟伟.pdf
set files[12]=基于深度强化学习的聚变堆巡检四旋翼无人机关键技术研究_余超.pdf
set files[13]=专家知识辅助的强化学习研究及其在无人机路径规划中的应用_王国芳.pdf
set files[14]=多无人机协同路径规划方法研究_刘希阳.pdf
set files[15]=深度强化学习SAC算法在运动控制中的应用研究_翁士博.pdf
set files[16]=无人机自组织网络中的高效智能接入机制研究_张庭辉.pdf
set files[17]=基于深度强化学习的无人机对战战术决策的研究_胡真财.pdf
set files[18]=基于深度强化学习的无人机自主降落方法研究_王帅康.pdf
set files[19]=基于深度强化学习和迁移学习的反应式避障方法研究_吴宇豪.pdf
set files[20]=基于强化学习的多无人机协同任务规划算法研究_樊龙涛.pdf
set files[21]=基于RRT改进的机器人路径规划算法研究_蒋环宇.pdf
set files[22]=面向无人机自主避障导航的深度强化学习算法研究_张运涛.pdf
set files[23]=无人机路径规划算法研究_段跃飞.pdf
set files[24]=基于遗传算法和深度强化学习的多无人机协同区域监视的航路规划_李艳庆.pdf
set files[25]=基于强化学习的多无人机路径规划_张仕充.pdf
set files[26]=无人机路径规划算法研究_段跃飞 (1).pdf
set files[27]=动态环境下多无人机协同任务规划方法_竺殊荣.pdf
set files[28]=基于强化学习多无人机路径规划算法研究及实现_于盛.pdf
set files[29]=基于强化学习的无人机空中格斗算法研究_魏航.pdf
set files[30]=基于强化学习的四旋翼无人机路径规划方法研究_张天泽.pdf

REM Try flash-extract first (20 pages limit)
for /L %%i in (1,1,30) do (
    echo [%%i/30] Processing: !files[%%i]!
    mineru-open-api flash-extract "!files[%%i]!" --language ch -o "knowledge_base_new\!files[%%i]:.pdf=.md!" 2>nul
    if errorlevel 1 (
        echo   ^> Exceeds 20 pages, splitting...
        mineru-open-api flash-extract "!files[%%i]!" --language ch --pages 1-20 -o "knowledge_base_new\!files[%%i]:.pdf=_p1-20.md!"
        mineru-open-api flash-extract "!files[%%i]!" --language ch --pages 21-40 -o "knowledge_base_new\!files[%%i]:.pdf=_p21-40.md!"
        mineru-open-api flash-extract "!files[%%i]!" --language ch --pages 41-60 -o "knowledge_base_new\!files[%%i]:.pdf=_p41-60.md!"
        mineru-open-api flash-extract "!files[%%i]!" --language ch --pages 61-80 -o "knowledge_base_new\!files[%%i]:.pdf=_p61-80.md!"
        mineru-open-api flash-extract "!files[%%i]!" --language ch --pages 81-100 -o "knowledge_base_new\!files[%%i]:.pdf=_p81-100.md!"
    )
    echo.
)

echo ========================================
echo Done! Check knowledge_base_new\ folder
echo ========================================
pause
