export const meta = {
  name: 'batch-literature-review',
  description: '批量文献研读工作流 - 按方向并行研读，交叉关联分析，生成综述报告',
  phases: [
    { title: '分类排序', detail: '按方向和类型对文献分类，确定研读优先级' },
    { title: '并行研读', detail: '按方向分组并行研读，生成知识卡片' },
    { title: '关联分析', detail: '跨文献关联分析，技术演进图谱' },
    { title: '知识整合', detail: '生成综述报告和项目借鉴清单' },
  ],
};

// ============================================================
// 配置区：根据实际需求修改
// ============================================================

// 文献清单（从参考文献清单.md中提取）
const LITERATURE_LIST = [
  // 多无人机协同
  { id: 1, title: 'Cooperative Multi-UAV Path Planning Using DRL', authors: 'Phan et al.', year: 2023, source: 'EAAI', category: '协同', type: '期刊' },
  { id: 2, title: 'Multi-Agent RL for Decentralized Cooperative Obstacle Avoidance', authors: 'Chen et al.', year: 2022, source: 'IEEE RA-L', category: '协同', type: '期刊' },
  { id: 3, title: 'Multi-UAV Cooperative Formation Flight Control Using DRL', authors: 'Wang et al.', year: 2020, source: 'JGCD', category: '协同', type: '期刊' },
  { id: 4, title: 'A Comprehensive Survey on Multi-UAV Systems', authors: 'Mozaffari et al.', year: 2019, source: 'IEEE COMST', category: '协同', type: '综述' },
  { id: 5, title: 'Distributed Multi-Agent Path Finding Using DRL', authors: 'Sartoretti et al.', year: 2019, source: 'IEEE RA-L', category: '协同', type: '期刊' },
  { id: 6, title: 'Multi-UAV Path Planning Using Distributed MPC', authors: 'Kamel et al.', year: 2020, source: 'Autonomous Robots', category: '协同', type: '期刊' },

  // 多智能体强化学习
  { id: 7, title: 'The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games', authors: 'Yu et al.', year: 2022, source: 'NeurIPS', category: 'MARL', type: '会议' },
  { id: 8, title: 'Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments', authors: 'Lowe et al.', year: 2017, source: 'NeurIPS', category: 'MARL', type: '会议' },
  { id: 9, title: 'QMIX: Monotonic Value Function Factorisation', authors: 'Rashid et al.', year: 2018, source: 'ICML', category: 'MARL', type: '会议' },
  { id: 10, title: 'Value-Decomposition Networks For Cooperative Multi-Agent Learning', authors: 'Sunehag et al.', year: 2018, source: 'AAMAS', category: 'MARL', type: '会议' },
  { id: 11, title: 'QPLEX: Duplex Dueling Multi-Agent Q-Learning', authors: 'Wang et al.', year: 2021, source: 'ICLR', category: 'MARL', type: '会议' },
  { id: 12, title: 'Learning Multiagent Communication with Backpropagation', authors: 'Sukhbaatar et al.', year: 2016, source: 'NeurIPS', category: 'MARL', type: '会议' },
  { id: 13, title: 'Learning to Communicate with Deep Multi-Agent Reinforcement Learning', authors: 'Foerster et al.', year: 2016, source: 'NeurIPS', category: 'MARL', type: '会议' },
  { id: 14, title: 'TarMAC: Targeted Multi-Agent Communication', authors: 'Das et al.', year: 2019, source: 'ICML', category: 'MARL', type: '会议' },

  // 任务分配算法
  { id: 15, title: 'The Hungarian Method for the Assignment Problem', authors: 'Kuhn', year: 1955, source: 'Naval Research', category: '任务分配', type: '期刊' },
  { id: 16, title: 'Auction Algorithms for Network Flows', authors: 'Bertsekas', year: 1992, source: 'Mathematical Programming', category: '任务分配', type: '期刊' },
  { id: 17, title: 'Attention, Learn to Solve Routing Problems!', authors: 'Kool et al.', year: 2019, source: 'ICLR', category: '任务分配', type: '会议' },
  { id: 18, title: 'Multi-Agent Reinforcement Learning for Combinatorial Optimization', authors: 'Li et al.', year: 2022, source: 'NeurIPS', category: '任务分配', type: '会议' },
  { id: 19, title: 'Multi-UAV Cooperative Search and Task Allocation', authors: 'Li et al.', year: 2023, source: 'IEEE TASE', category: '任务分配', type: '期刊' },
  { id: 20, title: 'Multi-Agent Reinforcement Learning: A Survey', authors: 'Zhang et al.', year: 2021, source: 'arXiv', category: '任务分配', type: '综述' },

  // 深度强化学习路径规划
  { id: 21, title: 'Path Planning using Deep Reinforcement Learning', authors: 'Tai et al.', year: 2017, source: 'arXiv', category: 'DRL规划', type: '会议' },
  { id: 22, title: 'Mobile Robot Path Planning Using DRL in Unknown Environment', authors: 'Li et al.', year: 2021, source: 'IEEE Access', category: 'DRL规划', type: '期刊' },
  { id: 23, title: 'Autonomous Navigation via Deep Reinforcement Learning', authors: 'Zhu et al.', year: 2020, source: 'IEEE TNNLS', category: 'DRL规划', type: '期刊' },
  { id: 24, title: 'Deep Reinforcement Learning for UAV Navigation', authors: 'Polvara et al.', year: 2019, source: 'IEEE RA-L', category: 'DRL规划', type: '期刊' },
  { id: 25, title: 'Proximal Policy Optimization Algorithms', authors: 'Schulman et al.', year: 2017, source: 'arXiv', category: 'DRL规划', type: '会议' },
  { id: 26, title: 'Soft Actor-Critic: Off-Policy Maximum Entropy DRL', authors: 'Haarnoja et al.', year: 2018, source: 'ICML', category: 'DRL规划', type: '会议' },

  // 无人机物流配送与能耗
  { id: 27, title: 'The Vehicle Routing Problem with Drones', authors: 'Murray & Chu', year: 2015, source: 'Transportation Research', category: '物流配送', type: '期刊' },
];

// 按方向分组
const CATEGORIES = {
  '协同': LITERATURE_LIST.filter(l => l.category === '协同'),
  'MARL': LITERATURE_LIST.filter(l => l.category === 'MARL'),
  '任务分配': LITERATURE_LIST.filter(l => l.category === '任务分配'),
  'DRL规划': LITERATURE_LIST.filter(l => l.category === 'DRL规划'),
  '物流配送': LITERATURE_LIST.filter(l => l.category === '物流配送'),
};

// ============================================================
// 工作流主体
// ============================================================

// Phase 1: 文献分类与优先级排序
phase('分类排序');

const classification = await agent(
  `你是文献研读总控调度专家。

项目背景：基于深度强化学习的多无人机协同物流配送路径规划

请对以下文献进行分类和优先级排序：

${LITERATURE_LIST.map(l => `[${l.id}] ${l.title} (${l.authors}, ${l.year}, ${l.source}) - ${l.category} - ${l.type}`).join('\n')}

分类维度：
1. 按研究方向：${Object.keys(CATEGORIES).join('、')}
2. 按文献类型：期刊、会议、综述
3. 按相关度：高/中/低（与本项目的关联程度）

输出格式（JSON）：
{
  "high_priority": [{id, title, reason}],
  "medium_priority": [{id, title, reason}],
  "low_priority": [{id, title, reason}],
  "reading_order": [按推荐研读顺序排列的文献id],
  "summary": "分类统计和研读计划概述"
}`,
  { label: '分类调度', phase: '分类排序', schema: {
    type: 'object',
    properties: {
      high_priority: { type: 'array', items: { type: 'object', properties: { id: { type: 'number' }, title: { type: 'string' }, reason: { type: 'string' } } } },
      medium_priority: { type: 'array', items: { type: 'object', properties: { id: { type: 'number' }, title: { type: 'string' }, reason: { type: 'string' } } } },
      low_priority: { type: 'array', items: { type: 'object', properties: { id: { type: 'number' }, title: { type: 'string' }, reason: { type: 'string' } } } },
      reading_order: { type: 'array', items: { type: 'number' } },
      summary: { type: 'string' }
    },
    required: ['high_priority', 'medium_priority', 'low_priority', 'reading_order', 'summary']
  }}
);

log(`分类完成: 高相关 ${classification?.high_priority?.length || 0} 篇, 中相关 ${classification?.medium_priority?.length || 0} 篇, 低相关 ${classification?.low_priority?.length || 0} 篇`);

// Phase 2: 按方向并行研读
phase('并行研读');

const REVIEW_PROMPT_TEMPLATE = (category, literatureList) => `你是${category}方向的文献研读专家。

项目背景：基于深度强化学习的多无人机协同物流配送路径规划

请对以下文献进行深度研读：

${literatureList.map(l => `[${l.id}] ${l.title} (${l.authors}, ${l.year}, ${l.source})`).join('\n')}

对每篇文献，按以下结构输出知识卡片：

## [编号] [标题]
> 作者: ... | 年份: ... | 来源: ... | 类型: ... | 相关度: ★/★★/★★★

### 一句话总结
> [核心贡献]

### 研究问题
- 核心问题：[精确定义]
- 研究动机：[为什么重要]

### 核心方法
- 技术路线：[概述]
- 关键创新点：[逐条]
- 核心公式：$[formula]$
- 算法流程：[伪代码或步骤]

### 实验结果
| 方法 | 指标1 | 指标2 |
|------|-------|-------|
| 本文 | [值] | [值] |
| 基线 | [值] | [值] |

### 批判性分析
- 优势：[列出]
- 局限：[列出]

### 与本项目关联
- 关联模块：[环境建模/奖励设计/算法选择/任务分配]
- 借鉴要点：[具体说明]
- 需要的适配：[说明]

输出要求：
- 公式用LaTeX格式
- 数据用表格
- 每篇文献独立一个知识卡片
- 标注不确定内容：[待确认]`;

// 并行研读5个方向
const directionResults = await parallel(Object.entries(CATEGORIES).map(([category, lits]) =>
  () => agent(
    REVIEW_PROMPT_TEMPLATE(category, lits),
    { label: `研读:${category}`, phase: '并行研读' }
  )
));

const allCards = directionResults.filter(Boolean).join('\n\n---\n\n');
log(`并行研读完成，共处理 ${LITERATURE_LIST.length} 篇文献`);

// Phase 3: 交叉关联分析
phase('关联分析');

const linkage = await agent(
  `你是跨文献知识关联分析专家。

项目背景：基于深度强化学习的多无人机协同物流配送路径规划

以下是各方向文献的研读结果：

${allCards}

请进行以下关联分析：

## 1. 文献-方法矩阵
构建一个矩阵，行是文献，列是关键技术维度：
| 文献 | CTDE | 值分解 | Actor-Critic | 通信学习 | 注意力机制 | 能耗建模 | 任务分配 |
|------|------|--------|-------------|---------|-----------|---------|---------|

## 2. 技术演进图
描述以下技术的发展脉络：
- MARL算法演进
- 通信学习演进
- 路径规划方法演进

## 3. 方法对比分析
| 方法类别 | 代表方法 | 优势 | 劣势 | 适用场景 | 推荐度 |
|---------|---------|------|------|---------|--------|

## 4. 研究空白识别
- 哪些组合未被探索？
- 哪些假设未被验证？
- 本项目可以切入的方向？

## 5. 项目借鉴清单
按本项目模块分类：
- 环境建模：参考哪些文献，具体借鉴什么
- 奖励设计：参考哪些文献，具体借鉴什么
- 算法选择：推荐哪些方法，为什么
- 任务分配：推荐哪些方法，为什么

## 6. 推荐技术路线
基于以上分析，为本项目推荐最优的技术组合方案。

输出格式：结构化Markdown，包含表格和图示。`,
  { label: '关联分析', phase: '关联分析' }
);

log('关联分析完成');

// Phase 4: 知识整合与输出
phase('知识整合');

const synthesis = await agent(
  `你是学术综述撰写专家。

项目背景：基于深度强化学习的多无人机协同物流配送路径规划

基于以下研读结果和关联分析，请生成完整的文献综述：

## 研读结果
${allCards}

## 关联分析
${linkage}

请输出以下内容：

### 1. 文献综述初稿（2000-3000字）
按以下结构组织：
1.1 研究背景与意义
1.2 多无人机协同技术研究现状
1.3 多智能体强化学习研究现状
1.4 任务分配算法研究现状
1.5 路径规划方法研究现状
1.6 现有研究不足与本文切入点

### 2. 项目技术路线建议
基于文献分析，推荐本项目的技术方案：
- 环境建模方案
- 算法选择建议
- 奖励函数设计思路
- 任务分配机制

### 3. 需补读的文献
列出研读过程中发现的、但不在当前清单中的相关文献。

### 4. 研究创新点建议
基于研究空白，提出本项目可能的创新方向。

输出格式：结构化Markdown。`,
  { label: '知识整合', phase: '知识整合' }
);

log('知识整合完成');

return {
  classification,
  totalLiterature: LITERATURE_LIST.length,
  categories: Object.keys(CATEGORIES).length,
  linkage,
  synthesis,
};
