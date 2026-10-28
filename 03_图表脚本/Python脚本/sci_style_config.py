"""
SCI论文风格绘图配置
参考Nature/Science期刊配色方案
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# ==================== SCI风格配色方案 ====================

# 主色调 - 清新淡绿色系
SCI_COLORS = {
    'primary': '#4CAF50',        # 主色 - 清新绿
    'secondary': '#81C784',      # 次要色 - 浅绿
    'tertiary': '#A5D6A7',       # 第三色 - 更浅绿
    'accent': '#2E7D32',         # 强调色 - 深绿
    'light': '#E8F5E9',          # 浅背景 - 极浅绿
    'background': '#F1F8E9',     # 背景色 - 淡绿白
}

# 多系列配色（用于对比图）
SCI_SERIES = [
    '#4CAF50',  # 绿色 - 主要方法
    '#2196F3',  # 蓝色 - 对比方法1
    '#FF9800',  # 橙色 - 对比方法2
    '#9C27B0',  # 紫色 - 对比方法3
    '#F44336',  # 红色 - 对比方法4
    '#00BCD4',  # 青色 - 对比方法5
]

# 渐变色（用于填充区域）
SCI_GRADIENT = ['#E8F5E9', '#C8E6C9', '#A5D6A7', '#81C784', '#66BB6A', '#4CAF50']

# ==================== 样式配置 ====================

def set_sci_style():
    """设置SCI论文风格"""
    plt.style.use('seaborn-v0_8-whitegrid')

    # 字体设置
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'SimSun']
    plt.rcParams['font.sans-serif'] = ['Arial', 'SimHei']
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 9

    # 坐标轴设置
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['axes.labelcolor'] = '#333333'
    plt.rcParams['axes.unicode_minus'] = False

    # 网格设置
    plt.rcParams['grid.color'] = '#E0E0E0'
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.linewidth'] = 0.5
    plt.rcParams['grid.alpha'] = 0.7

    # 刻度设置
    plt.rcParams['xtick.major.width'] = 0.8
    plt.rcParams['ytick.major.width'] = 0.8
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    # 图例设置
    plt.rcParams['legend.frameon'] = True
    plt.rcParams['legend.framealpha'] = 0.9
    plt.rcParams['legend.edgecolor'] = '#CCCCCC'

    # 保存设置
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.pad_inches'] = 0.1

    print("SCI风格配置已加载", flush=True)


def get_figure(figsize=(8, 6), dpi=150):
    """创建SCI风格的图形"""
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    return fig, ax


def style_axis(ax, xlabel='', ylabel='', title='', grid=True):
    """设置坐标轴样式"""
    if xlabel:
        ax.set_xlabel(xlabel, fontweight='bold')
    if ylabel:
        ax.set_ylabel(ylabel, fontweight='bold')
    if title:
        ax.set_title(title, fontweight='bold', pad=10)
    if grid:
        ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return ax


def add_legend(ax, loc='best', ncol=1):
    """添加图例"""
    legend = ax.legend(loc=loc, ncol=ncol, frameon=True,
                       framealpha=0.9, edgecolor='#CCCCCC')
    return legend


# ==================== 绘图函数 ====================

def plot_line(ax, x, y, label='', color=None, linewidth=2, alpha=1.0):
    """绘制折线"""
    if color is None:
        color = SCI_COLORS['primary']
    line, = ax.plot(x, y, color=color, linewidth=linewidth, alpha=alpha, label=label)
    return line


def plot_fill(ax, x, y_mean, y_std, color=None, alpha=0.3):
    """绘制填充区域"""
    if color is None:
        color = SCI_COLORS['primary']
    ax.fill_between(x,
                    y_mean - y_std,
                    y_mean + y_std,
                    color=color, alpha=alpha)


def plot_bar(ax, x, height, width=0.6, color=None, edgecolor='white', label=''):
    """绘制柱状图"""
    if color is None:
        color = SCI_COLORS['primary']
    bars = ax.bar(x, height, width=width, color=color,
                  edgecolor=edgecolor, linewidth=0.5, label=label)
    return bars


def add_value_labels(ax, bars, fmt='{:.2f}', fontsize=8):
    """为柱状图添加数值标签"""
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                fmt.format(height),
                ha='center', va='bottom', fontsize=fontsize)


# ==================== 配色应用示例 ====================

def demo_plot():
    """演示SCI风格绘图"""
    set_sci_style()

    # 创建示例数据
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    y3 = np.sin(x) * np.exp(-x/10)

    # 创建图形
    fig, ax = get_figure(figsize=(10, 6))
    style_axis(ax, xlabel='Training Episodes', ylabel='Performance',
               title='SCI Style Demo')

    # 绘制曲线
    plot_line(ax, x, y1, label='Method A', color=SCI_SERIES[0])
    plot_line(ax, x, y2, label='Method B', color=SCI_SERIES[1])
    plot_line(ax, x, y3, label='Method C', color=SCI_SERIES[2])

    # 添加图例
    add_legend(ax, loc='upper right')

    plt.tight_layout()
    plt.savefig('sci_style_demo.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("演示图已保存: sci_style_demo.png")


if __name__ == '__main__':
    demo_plot()
