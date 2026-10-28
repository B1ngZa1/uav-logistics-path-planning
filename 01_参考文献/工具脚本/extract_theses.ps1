# PowerShell script to extract new PDFs
Write-Host "========================================"
Write-Host "Extract New 30 PDF Files (PowerShell)"
Write-Host "========================================"
Write-Host ""

Set-Location "C:\University\Files\Training program\无人机项目设计\01_参考文献"

if (!(Test-Path "knowledge_base_new")) {
    New-Item -ItemType Directory -Path "knowledge_base_new"
}

Write-Host "Processing doctoral and master theses (split by 20 pages)..."
Write-Host ""

# Theses that exceed 20 pages
$theses = @(
    @{name="多无人机类脑智能决策与协同控制方法研究_赵欣怡"; pages=@("1-20","21-40","41-60")},
    @{name="无人机集群编队自主规划与控制技术研究_张博渊"; pages=@("1-20","21-40","41-60")},
    @{name="基于强化学习的无人机智能空战决策研究_杨啟明"; pages=@("1-20","21-40","41-60")},
    @{name="四旋翼无人机安全控制问题研究_金涛"; pages=@("1-20","21-40")},
    @{name="生物启发的多无人机协同四维航迹规划方法研究_杨祖强"; pages=@("1-20","21-40")},
    @{name="基于深度强化学习的旋翼无人机敏捷飞行控制与运动规划_陈时雨"; pages=@("1-20","21-40")},
    @{name="无人机集群编队及其避障控制关键技术研究_赵伟伟"; pages=@("1-20","21-40")},
    @{name="基于深度强化学习的聚变堆巡检四旋翼无人机关键技术研究_余超"; pages=@("1-20","21-40")},
    @{name="专家知识辅助的强化学习研究及其在无人机路径规划中的应用_王国芳"; pages=@("1-20","21-40")},
    @{name="多无人机协同路径规划方法研究_刘希阳"; pages=@("1-20","21-40")},
    @{name="基于强化学习的多无人机协同任务规划算法研究_樊龙涛"; pages=@("1-20","21-40")},
    @{name="面向无人机自主避障导航的深度强化学习算法研究_张运涛"; pages=@("1-20","21-40")},
    @{name="无人机路径规划算法研究_段跃飞"; pages=@("1-20","21-40")},
    @{name="基于遗传算法和深度强化学习的多无人机协同区域监视的航路规划_李艳庆"; pages=@("1-20","21-40")},
    @{name="动态环境下多无人机协同任务规划方法_竺殊荣"; pages=@("1-20","21-40")},
    @{name="基于强化学习多无人机路径规划算法研究及实现_于盛"; pages=@("1-20","21-40")},
    @{name="基于强化学习的无人机空中格斗算法研究_魏航"; pages=@("1-20","21-40")},
    @{name="基于强化学习的四旋翼无人机路径规划方法研究_张天泽"; pages=@("1-20","21-40")}
)

$i = 0
foreach ($thesis in $theses) {
    $i++
    Write-Host "[$i/$($theses.Count)] $($thesis.name)"
    foreach ($page in $thesis.pages) {
        $pdfFile = "$($thesis.name).pdf"
        $mdFile = "knowledge_base_new\$($thesis.name)_p$page.md"
        Write-Host "  Extracting pages $page..."
        mineru-open-api flash-extract $pdfFile --language ch --pages $page -o $mdFile 2>$null
    }
    Write-Host ""
}

Write-Host "========================================"
Write-Host "Processing shorter papers (under 20 pages)..."
Write-Host "========================================"
Write-Host ""

# Shorter papers
$shortPapers = @(
    "机器人深度强化学习控制方法研究_王远大",
    "基于深度强化学习的目标跟踪算法研究_杨宇翔",
    "基于深度强化学习的认知物联网资源分配研究_郭少艾",
    "无人机通信系统的无线资源智能管理技术研究_周世阳",
    "深度强化学习SAC算法在运动控制中的应用研究_翁士博",
    "无人机自组织网络中的高效智能接入机制研究_张庭辉",
    "基于深度强化学习的无人机对战战术决策的研究_胡真财",
    "基于深度强化学习的无人机自主降落方法研究_王帅康",
    "基于深度强化学习和迁移学习的反应式避障方法研究_吴宇豪",
    "基于RRT改进的机器人路径规划算法研究_蒋环宇",
    "基于强化学习的多无人机路径规划_张仕充",
    "基于深度强化学习的无人机群智感知：多模态融合增益与资源消耗的显式耦合调度策略_杨桂松"
)

$j = 0
foreach ($paper in $shortPapers) {
    $j++
    Write-Host "[$j/$($shortPapers.Count)] $paper"
    $pdfFile = "$paper.pdf"
    $mdFile = "knowledge_base_new\$paper.md"
    mineru-open-api flash-extract $pdfFile --language ch -o $mdFile 2>$null
    Write-Host ""
}

Write-Host "========================================"
Write-Host "Done! All 30 PDFs processed"
Write-Host "Output: knowledge_base_new\"
Write-Host "========================================"
