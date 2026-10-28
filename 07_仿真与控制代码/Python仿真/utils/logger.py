"""
日志工具
项目：基于深度强化学习的多无人机协同物流配送路径规划
"""

import os
import csv
from datetime import datetime


class Logger:
    """训练日志记录器"""

    def __init__(self, log_dir, experiment_name):
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        os.makedirs(log_dir, exist_ok=True)

        # CSV日志文件
        self.csv_path = os.path.join(log_dir, f'{experiment_name}_log.csv')
        self.csv_file = open(self.csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'episode', 'reward', 'delivered', 'collisions',
            'actor_loss', 'critic_loss', 'timestamp'
        ])

    def log_episode(self, episode, reward, delivered, collisions,
                    actor_loss=0, critic_loss=0):
        """记录一个回合的日志"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.csv_writer.writerow([
            episode, f'{reward:.2f}', delivered, collisions,
            f'{actor_loss:.4f}', f'{critic_loss:.4f}', timestamp
        ])
        self.csv_file.flush()

    def close(self):
        """关闭日志文件"""
        self.csv_file.close()

    def get_log_path(self):
        """获取日志文件路径"""
        return self.csv_path
