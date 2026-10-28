# -*- coding: utf-8 -*-
"""
Extract all new PDF files using MinerU
"""
import os
import subprocess

# Set working directory
os.chdir(r"C:\University\Files\Training program\无人机项目设计\01_参考文献")

# Create output directory
os.makedirs("knowledge_base_new", exist_ok=True)

# Theses that exceed 20 pages (split into chunks)
theses = [
    ("多无人机类脑智能决策与协同控制方法研究_赵欣怡", ["1-20", "21-40", "41-60"]),
    ("无人机集群编队自主规划与控制技术研究_张博渊", ["1-20", "21-40", "41-60"]),
    ("基于强化学习的无人机智能空战决策研究_杨啟明", ["1-20", "21-40", "41-60"]),
    ("四旋翼无人机安全控制问题研究_金涛", ["1-20", "21-40"]),
    ("生物启发的多无人机协同四维航迹规划方法研究_杨祖强", ["1-20", "21-40"]),
    ("基于深度强化学习的旋翼无人机敏捷飞行控制与运动规划_陈时雨", ["1-20", "21-40"]),
    ("无人机集群编队及其避障控制关键技术研究_赵伟伟", ["1-20", "21-40"]),
    ("基于深度强化学习的聚变堆巡检四旋翼无人机关键技术研究_余超", ["1-20", "21-40"]),
    ("专家知识辅助的强化学习研究及其在无人机路径规划中的应用_王国芳", ["1-20", "21-40"]),
    ("多无人机协同路径规划方法研究_刘希阳", ["1-20", "21-40"]),
    ("基于强化学习的多无人机协同任务规划算法研究_樊龙涛", ["1-20", "21-40"]),
    ("面向无人机自主避障导航的深度强化学习算法研究_张运涛", ["1-20", "21-40"]),
    ("无人机路径规划算法研究_段跃飞", ["1-20", "21-40"]),
    ("基于遗传算法和深度强化学习的多无人机协同区域监视的航路规划_李艳庆", ["1-20", "21-40"]),
    ("动态环境下多无人机协同任务规划方法_竺殊荣", ["1-20", "21-40"]),
    ("基于强化学习多无人机路径规划算法研究及实现_于盛", ["1-20", "21-40"]),
    ("基于强化学习的无人机空中格斗算法研究_魏航", ["1-20", "21-40"]),
    ("基于强化学习的四旋翼无人机路径规划方法研究_张天泽", ["1-20", "21-40"]),
]

# Shorter papers (under 20 pages)
short_papers = [
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
    "基于深度强化学习的无人机群智感知：多模态融合增益与资源消耗的显式耦合调度策略_杨桂松",
]

print("=" * 60)
print("Extracting PDF files using MinerU")
print("=" * 60)
print()

# Process theses (split by pages)
print("Processing doctoral/master theses (split by 20 pages)...")
print()

for i, (name, pages) in enumerate(theses, 1):
    pdf_file = f"{name}.pdf"
    print(f"[{i}/{len(theses)}] {name}")

    for page in pages:
        md_file = f"knowledge_base_new/{name}_p{page}.md"
        print(f"  Extracting pages {page}...")
        try:
            result = subprocess.run(
                ["mineru-open-api", "flash-extract", pdf_file, "--language", "ch", "--pages", page, "-o", md_file],
                capture_output=True, text=True, encoding='utf-8'
            )
            if result.returncode == 0:
                print(f"    Done: {md_file}")
            else:
                print(f"    Error: {result.stderr[:100]}")
        except Exception as e:
            print(f"    Error: {str(e)[:100]}")
    print()

# Process short papers
print("=" * 60)
print("Processing shorter papers (under 20 pages)...")
print("=" * 60)
print()

for j, name in enumerate(short_papers, 1):
    pdf_file = f"{name}.pdf"
    md_file = f"knowledge_base_new/{name}.md"
    print(f"[{j}/{len(short_papers)}] {name}")
    try:
        result = subprocess.run(
            ["mineru-open-api", "flash-extract", pdf_file, "--language", "ch", "-o", md_file],
            capture_output=True, text=True, encoding='utf-8'
        )
        if result.returncode == 0:
            print(f"  Done: {md_file}")
        else:
            print(f"  Error: {result.stderr[:100]}")
    except Exception as e:
        print(f"  Error: {str(e)[:100]}")
    print()

print("=" * 60)
print("Done! All 30 PDFs processed")
print("Output: knowledge_base_new/")
print("=" * 60)
