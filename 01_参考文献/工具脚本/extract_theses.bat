@echo off
setlocal enabledelayedexpansion
echo ========================================
echo Extract New PDF Files (Theses)
echo ========================================
echo.

cd /d "C:\University\Files\Training program\无人机项目设计\01_参考文献"

if not exist "knowledge_base_new" mkdir "knowledge_base_new"

echo Processing doctoral and master theses...
echo These exceed 20 pages, so we split them.
echo.

REM Theses that likely exceed 20 pages - split into chunks
echo [1] 多无人机类脑智能决策与协同控制方法研究_赵欣怡
mineru-open-api flash-extract "多无人机类脑智能决策与协同控制方法研究_赵欣怡.pdf" --language ch --pages 1-20 -o "knowledge_base_new\赵欣怡_p1-20.md"
mineru-open-api flash-extract "多无人机类脑智能决策与协同控制方法研究_赵欣怡.pdf" --language ch --pages 21-40 -o "knowledge_base_new\赵欣怡_p21-40.md"
mineru-open-api flash-extract "多无人机类脑智能决策与协同控制方法研究_赵欣怡.pdf" --language ch --pages 41-60 -o "knowledge_base_new\赵欣怡_p41-60.md"

echo [2] 无人机集群编队自主规划与控制技术研究_张博渊
mineru-open-api flash-extract "无人机集群编队自主规划与控制技术研究_张博渊.pdf" --language ch --pages 1-20 -o "knowledge_base_new\张博渊_p1-20.md"
mineru-open-api flash-extract "无人机集群编队自主规划与控制技术研究_张博渊.pdf" --language ch --pages 21-40 -o "knowledge_base_new\张博渊_p21-40.md"
mineru-open-api flash-extract "无人机集群编队自主规划与控制技术研究_张博渊.pdf" --language ch --pages 41-60 -o "knowledge_base_new\张博渊_p41-60.md"

echo [3] 基于强化学习的无人机智能空战决策研究_杨啟明
mineru-open-api flash-extract "基于强化学习的无人机智能空战决策研究_杨啟明.pdf" --language ch --pages 1-20 -o "knowledge_base_new\杨啟明_p1-20.md"
mineru-open-api flash-extract "基于强化学习的无人机智能空战决策研究_杨啟明.pdf" --language ch --pages 21-40 -o "knowledge_base_new\杨啟明_p21-40.md"
mineru-open-api flash-extract "基于强化学习的无人机智能空战决策研究_杨啟明.pdf" --language ch --pages 41-60 -o "knowledge_base_new\杨啟明_p41-60.md"

echo [4] 四旋翼无人机安全控制问题研究_金涛
mineru-open-api flash-extract "四旋翼无人机安全控制问题研究_金涛.pdf" --language ch --pages 1-20 -o "knowledge_base_new\金涛_p1-20.md"
mineru-open-api flash-extract "四旋翼无人机安全控制问题研究_金涛.pdf" --language ch --pages 21-40 -o "knowledge_base_new\金涛_p21-40.md"

echo [5] 生物启发的多无人机协同四维航迹规划方法研究_杨祖强
mineru-open-api flash-extract "生物启发的多无人机协同四维航迹规划方法研究_杨祖强.pdf" --language ch --pages 1-20 -o "knowledge_base_new\杨祖强_p1-20.md"
mineru-open-api flash-extract "生物启发的多无人机协同四维航迹规划方法研究_杨祖强.pdf" --language ch --pages 21-40 -o "knowledge_base_new\杨祖强_p21-40.md"

echo [6] 基于深度强化学习的旋翼无人机敏捷飞行控制与运动规划_陈时雨
mineru-open-api flash-extract "基于深度强化学习的旋翼无人机敏捷飞行控制与运动规划_陈时雨.pdf" --language ch --pages 1-20 -o "knowledge_base_new\陈时雨_p1-20.md"
mineru-open-api flash-extract "基于深度强化学习的旋翼无人机敏捷飞行控制与运动规划_陈时雨.pdf" --language ch --pages 21-40 -o "knowledge_base_new\陈时雨_p21-40.md"

echo [7] 无人机集群编队及其避障控制关键技术研究_赵伟伟
mineru-open-api flash-extract "无人机集群编队及其避障控制关键技术研究_赵伟伟.pdf" --language ch --pages 1-20 -o "knowledge_base_new\赵伟伟_p1-20.md"
mineru-open-api flash-extract "无人机集群编队及其避障控制关键技术研究_赵伟伟.pdf" --language ch --pages 21-40 -o "knowledge_base_new\赵伟伟_p21-40.md"

echo [8] 基于深度强化学习的聚变堆巡检四旋翼无人机关键技术研究_余超
mineru-open-api flash-extract "基于深度强化学习的聚变堆巡检四旋翼无人机关键技术研究_余超.pdf" --language ch --pages 1-20 -o "knowledge_base_new\余超_p1-20.md"
mineru-open-api flash-extract "基于深度强化学习的聚变堆巡检四旋翼无人机关键技术研究_余超.pdf" --language ch --pages 21-40 -o "knowledge_base_new\余超_p21-40.md"

echo [9] 专家知识辅助的强化学习研究及其在无人机路径规划中的应用_王国芳
mineru-open-api flash-extract "专家知识辅助的强化学习研究及其在无人机路径规划中的应用_王国芳.pdf" --language ch --pages 1-20 -o "knowledge_base_new\王国芳_p1-20.md"
mineru-open-api flash-extract "专家知识辅助的强化学习研究及其在无人机路径规划中的应用_王国芳.pdf" --language ch --pages 21-40 -o "knowledge_base_new\王国芳_p21-40.md"

echo [10] 多无人机协同路径规划方法研究_刘希阳
mineru-open-api flash-extract "多无人机协同路径规划方法研究_刘希阳.pdf" --language ch --pages 1-20 -o "knowledge_base_new\刘希阳_p1-20.md"
mineru-open-api flash-extract "多无人机协同路径规划方法研究_刘希阳.pdf" --language ch --pages 21-40 -o "knowledge_base_new\刘希阳_p21-40.md"

echo [11] 基于强化学习的多无人机协同任务规划算法研究_樊龙涛
mineru-open-api flash-extract "基于强化学习的多无人机协同任务规划算法研究_樊龙涛.pdf" --language ch --pages 1-20 -o "knowledge_base_new\樊龙涛_p1-20.md"
mineru-open-api flash-extract "基于强化学习的多无人机协同任务规划算法研究_樊龙涛.pdf" --language ch --pages 21-40 -o "knowledge_base_new\樊龙涛_p21-40.md"

echo [12] 面向无人机自主避障导航的深度强化学习算法研究_张运涛
mineru-open-api flash-extract "面向无人机自主避障导航的深度强化学习算法研究_张运涛.pdf" --language ch --pages 1-20 -o "knowledge_base_new\张运涛_p1-20.md"
mineru-open-api flash-extract "面向无人机自主避障导航的深度强化学习算法研究_张运涛.pdf" --language ch --pages 21-40 -o "knowledge_base_new\张运涛_p21-40.md"

echo [13] 无人机路径规划算法研究_段跃飞
mineru-open-api flash-extract "无人机路径规划算法研究_段跃飞.pdf" --language ch --pages 1-20 -o "knowledge_base_new\段跃飞_p1-20.md"
mineru-open-api flash-extract "无人机路径规划算法研究_段跃飞.pdf" --language ch --pages 21-40 -o "knowledge_base_new\段跃飞_p21-40.md"

echo [14] 基于遗传算法和深度强化学习的多无人机协同区域监视的航路规划_李艳庆
mineru-open-api flash-extract "基于遗传算法和深度强化学习的多无人机协同区域监视的航路规划_李艳庆.pdf" --language ch --pages 1-20 -o "knowledge_base_new\李艳庆_p1-20.md"
mineru-open-api flash-extract "基于遗传算法和深度强化学习的多无人机协同区域监视的航路规划_李艳庆.pdf" --language ch --pages 21-40 -o "knowledge_base_new\李艳庆_p21-40.md"

echo [15] 动态环境下多无人机协同任务规划方法_竺殊荣
mineru-open-api flash-extract "动态环境下多无人机协同任务规划方法_竺殊荣.pdf" --language ch --pages 1-20 -o "knowledge_base_new\竺殊荣_p1-20.md"
mineru-open-api flash-extract "动态环境下多无人机协同任务规划方法_竺殊荣.pdf" --language ch --pages 21-40 -o "knowledge_base_new\竺殊荣_p21-40.md"

echo [16] 基于强化学习多无人机路径规划算法研究及实现_于盛
mineru-open-api flash-extract "基于强化学习多无人机路径规划算法研究及实现_于盛.pdf" --language ch --pages 1-20 -o "knowledge_base_new\于盛_p1-20.md"
mineru-open-api flash-extract "基于强化学习多无人机路径规划算法研究及实现_于盛.pdf" --language ch --pages 21-40 -o "knowledge_base_new\于盛_p21-40.md"

echo [17] 基于强化学习的无人机空中格斗算法研究_魏航
mineru-open-api flash-extract "基于强化学习的无人机空中格斗算法研究_魏航.pdf" --language ch --pages 1-20 -o "knowledge_base_new\魏航_p1-20.md"
mineru-open-api flash-extract "基于强化学习的无人机空中格斗算法研究_魏航.pdf" --language ch --pages 21-40 -o "knowledge_base_new\魏航_p21-40.md"

echo [18] 基于强化学习的四旋翼无人机路径规划方法研究_张天泽
mineru-open-api flash-extract "基于强化学习的四旋翼无人机路径规划方法研究_张天泽.pdf" --language ch --pages 1-20 -o "knowledge_base_new\张天泽_p1-20.md"
mineru-open-api flash-extract "基于强化学习的四旋翼无人机路径规划方法研究_张天泽.pdf" --language ch --pages 21-40 -o "knowledge_base_new\张天泽_p21-40.md"

echo.
echo ========================================
echo Now processing shorter papers (under 20 pages)
echo ========================================
echo.

REM Shorter papers - try direct extraction
echo [19] 机器人深度强化学习控制方法研究_王远大
mineru-open-api flash-extract "机器人深度强化学习控制方法研究_王远大.pdf" --language ch -o "knowledge_base_new\王远大.md"

echo [20] 基于深度强化学习的目标跟踪算法研究_杨宇翔
mineru-open-api flash-extract "基于深度强化学习的目标跟踪算法研究_杨宇翔.pdf" --language ch -o "knowledge_base_new\杨宇翔.md"

echo [21] 基于深度强化学习的认知物联网资源分配研究_郭少艾
mineru-open-api flash-extract "基于深度强化学习的认知物联网资源分配研究_郭少艾.pdf" --language ch -o "knowledge_base_new\郭少艾.md"

echo [22] 无人机通信系统的无线资源智能管理技术研究_周世阳
mineru-open-api flash-extract "无人机通信系统的无线资源智能管理技术研究_周世阳.pdf" --language ch -o "knowledge_base_new\周世阳.md"

echo [23] 深度强化学习SAC算法在运动控制中的应用研究_翁士博
mineru-open-api flash-extract "深度强化学习SAC算法在运动控制中的应用研究_翁士博.pdf" --language ch -o "knowledge_base_new\翁士博.md"

echo [24] 无人机自组织网络中的高效智能接入机制研究_张庭辉
mineru-open-api flash-extract "无人机自组织网络中的高效智能接入机制研究_张庭辉.pdf" --language ch -o "knowledge_base_new\张庭辉.md"

echo [25] 基于深度强化学习的无人机对战战术决策的研究_胡真财
mineru-open-api flash-extract "基于深度强化学习的无人机对战战术决策的研究_胡真财.pdf" --language ch -o "knowledge_base_new\胡真财.md"

echo [26] 基于深度强化学习的无人机自主降落方法研究_王帅康
mineru-open-api flash-extract "基于深度强化学习的无人机自主降落方法研究_王帅康.pdf" --language ch -o "knowledge_base_new\王帅康.md"

echo [27] 基于深度强化学习和迁移学习的反应式避障方法研究_吴宇豪
mineru-open-api flash-extract "基于深度强化学习和迁移学习的反应式避障方法研究_吴宇豪.pdf" --language ch -o "knowledge_base_new\吴宇豪.md"

echo [28] 基于RRT改进的机器人路径规划算法研究_蒋环宇
mineru-open-api flash-extract "基于RRT改进的机器人路径规划算法研究_蒋环宇.pdf" --language ch -o "knowledge_base_new\蒋环宇.md"

echo [29] 基于强化学习的多无人机路径规划_张仕充
mineru-open-api flash-extract "基于强化学习的多无人机路径规划_张仕充.pdf" --language ch -o "knowledge_base_new\张仕充.md"

echo [30] 基于深度强化学习的无人机群智感知调度_杨桂松
mineru-open-api flash-extract "基于深度强化学习的无人机群智感知：多模态融合增益与资源消耗的显式耦合调度策略_杨桂松.pdf" --language ch -o "knowledge_base_new\杨桂松.md"

echo.
echo ========================================
echo Done! All 30 PDFs processed
echo Output: knowledge_base_new\
echo ========================================
pause
