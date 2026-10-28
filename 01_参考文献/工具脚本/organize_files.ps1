# 文件整理脚本
Write-Host "========================================"
Write-Host "Organize Reference Literature Files"
Write-Host "========================================"
Write-Host ""

Set-Location "C:\University\Files\Training program\无人机项目设计\01_参考文献"

# Create directory structure
Write-Host "Creating directory structure..."
New-Item -ItemType Directory -Path "PDF文件\第一批" -Force | Out-Null
New-Item -ItemType Directory -Path "PDF文件\第二批" -Force | Out-Null
New-Item -ItemType Directory -Path "提取内容\第一批" -Force | Out-Null
New-Item -ItemType Directory -Path "提取内容\第二批" -Force | Out-Null
New-Item -ItemType Directory -Path "工具脚本" -Force | Out-Null

Write-Host "Moving PDF files..."

# First batch PDFs (24 files)
$firstBatch = @(
    "基于HCA-MAPPO深度强化学习的多无人机协同路径规划方法_林振希.pdf",
    "基于多智能体强化学习的多无人机协同交通监控路径优化_刘晓波.pdf",
    "先验引导的时序融合多无人机协同避障航路规划方法_王傲.pdf",
    "基于双层强化学习的异构多无人机任务规划_付星桐.pdf",
    "基于多智能体强化学习的无人机对抗技术研究_孙笠晋.pdf",
    "基于多智能体强化学习的无人机集群协同综述_方伟.pdf",
    "基于多策略改进霜冰算法的多无人机协同任务分配_陈海洋.pdf",
    "基于多策略改进霜冰算法的多无人机协同任务分配_陈海洋 (1).pdf",
    "基于强化学习的无人机海上搜寻路径规划_张楠.pdf",
    "基于改进MATD3强化学习算法的多无人机通感一体系统能效优化_田峰.pdf",
    "基于深度学习的多无人机任务分配和路径优化方法研究_张警伟.pdf",
    "基于深度强化学习的无人机目标跟踪和障碍规避融合控制方法_李治威.pdf",
    "基于深度强化学习的无人机箔条干扰自卫防御决策方法设计_安通.pdf",
    "基于深度强化学习的无人机群智感知：多模态融合增益与资源消耗的显式耦合调度策略_杨桂松.pdf",
    "基于深度强化学习的架空输电线吊运无人机上下线导航法_梁加凯.pdf",
    "多无人机分区协同区域覆盖航迹规划_黄姣茹.pdf",
    "无人机物流配送路径规划的双目标约束深度Q_网络方法研究_赵学健.pdf",
    "融合深度强化学习的自适应势场路径规划算法_胥斌.pdf",
    "面向低空经济的无人机配送路径规划综述_廖勇.pdf",
    "基于分层强化学习的无人机导航方案_郑钧洋.pdf",
    "基于多智能体的无人机轨迹与资源优化算法_熊俊.pdf",
    "基于AI辅助的无人机物流动态路径规划策略_谭斯予.pdf",
    "基于元平均场Q学习的大规模无人机组网抗干扰与隐蔽通信机制_钱玉文.pdf",
    "面向无人机集群的通信与组网技术_赵海涛.pdf"
)

foreach ($file in $firstBatch) {
    if (Test-Path $file) {
        Move-Item $file "PDF文件\第一批\" -Force
        Write-Host "  Moved: $file"
    }
}

# Second batch PDFs (30 files)
$secondBatch = @(
    "多无人机类脑智能决策与协同控制方法研究_赵欣怡.pdf",
    "无人机集群编队自主规划与控制技术研究_张博渊.pdf",
    "基于强化学习的无人机智能空战决策研究_杨啟明.pdf",
    "四旋翼无人机安全控制问题研究_金涛.pdf",
    "生物启发的多无人机协同四维航迹规划方法研究_杨祖强.pdf",
    "机器人深度强化学习控制方法研究_王远大.pdf",
    "基于深度强化学习的旋翼无人机敏捷飞行控制与运动规划_陈时雨.pdf",
    "基于深度强化学习的目标跟踪算法研究_杨宇翔.pdf",
    "基于深度强化学习的认知物联网资源分配研究_郭少艾.pdf",
    "无人机通信系统的无线资源智能管理技术研究_周世阳.pdf",
    "无人机集群编队及其避障控制关键技术研究_赵伟伟.pdf",
    "基于深度强化学习的聚变堆巡检四旋翼无人机关键技术研究_余超.pdf",
    "专家知识辅助的强化学习研究及其在无人机路径规划中的应用_王国芳.pdf",
    "多无人机协同路径规划方法研究_刘希阳.pdf",
    "深度强化学习SAC算法在运动控制中的应用研究_翁士博.pdf",
    "无人机自组织网络中的高效智能接入机制研究_张庭辉.pdf",
    "基于深度强化学习的无人机对战战术决策的研究_胡真财.pdf",
    "基于深度强化学习的无人机自主降落方法研究_王帅康.pdf",
    "基于深度强化学习和迁移学习的反应式避障方法研究_吴宇豪.pdf",
    "基于强化学习的多无人机协同任务规划算法研究_樊龙涛.pdf",
    "基于RRT改进的机器人路径规划算法研究_蒋环宇.pdf",
    "面向无人机自主避障导航的深度强化学习算法研究_张运涛.pdf",
    "无人机路径规划算法研究_段跃飞.pdf",
    "基于遗传算法和深度强化学习的多无人机协同区域监视的航路规划_李艳庆.pdf",
    "基于强化学习的多无人机路径规划_张仕充.pdf",
    "无人机路径规划算法研究_段跃飞 (1).pdf",
    "动态环境下多无人机协同任务规划方法_竺殊荣.pdf",
    "基于强化学习多无人机路径规划算法研究及实现_于盛.pdf",
    "基于强化学习的无人机空中格斗算法研究_魏航.pdf",
    "基于强化学习的四旋翼无人机路径规划方法研究_张天泽.pdf"
)

foreach ($file in $secondBatch) {
    if (Test-Path $file) {
        Move-Item $file "PDF文件\第二批\" -Force
        Write-Host "  Moved: $file"
    }
}

Write-Host ""
Write-Host "Moving extracted content..."

# Move extracted files
if (Test-Path "knowledge_base") {
    Move-Item "knowledge_base\*" "提取内容\第一批\" -Force
    Remove-Item "knowledge_base" -Force -Recurse
    Write-Host "  Moved knowledge_base to 提取内容\第一批"
}

if (Test-Path "knowledge_base_new") {
    Move-Item "knowledge_base_new\*" "提取内容\第二批\" -Force
    Remove-Item "knowledge_base_new" -Force -Recurse
    Write-Host "  Moved knowledge_base_new to 提取内容\第二批"
}

Write-Host ""
Write-Host "Moving scripts..."

# Move scripts
$scripts = @(
    "extract_all_v2.py",
    "extract_theses.ps1",
    "extract_pdfs.bat",
    "extract_theses.bat",
    "extract_all_pdfs.bat",
    "run_extract.bat"
)

foreach ($script in $scripts) {
    if (Test-Path $script) {
        Move-Item $script "工具脚本\" -Force
        Write-Host "  Moved: $script"
    }
}

Write-Host ""
Write-Host "========================================"
Write-Host "Done! File structure organized."
Write-Host "========================================"
Write-Host ""
Write-Host "New structure:"
Write-Host "  01_参考文献/"
Write-Host "    ├── README.md"
Write-Host "    ├── 参考文献清单.md"
Write-Host "    ├── PDF文件/"
Write-Host "    │   ├── 第一批/ (24 files)"
Write-Host "    │   └── 第二批/ (30 files)"
Write-Host "    ├── 提取内容/"
Write-Host "    │   ├── 第一批/ (20 files)"
Write-Host "    │   └── 第二批/ (24 files)"
Write-Host "    ├── 知识库/"
Write-Host "    │   └── ... (organized by category)"
Write-Host "    └── 工具脚本/"
Write-Host "        └── ... (extraction scripts)"
