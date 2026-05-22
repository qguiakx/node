"""
日志配置文件
"""

import logging
import os
import sys
from datetime import datetime

from src.utils.path_tool import get_abs_path

# 日志保存的根目录
LOG_ROOT = get_abs_path("logs")

# 确保目录的文件存在
os.makedirs(LOG_ROOT, exist_ok=True)

# 日志的配置格式
DEFAULT_LOG_FORMAT = logging.Formatter(
    "%(asctime)s - [%(name)s] - [%(levelname)s] - %(filename)s:%(lineno)d - %(message)s"
)


def get_logger(
        name: str = "agent",
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG,
        log_file=None
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # 避免重复添加Handler
    if logger.handlers:
        return logger

    # 控制台Handler
    handler = logging.StreamHandler()
    handler.setLevel(console_level)
    handler.setFormatter(DEFAULT_LOG_FORMAT)

    logger.addHandler(handler)

    # 文件Handler
    if not log_file:
        log_file = os.path.join(LOG_ROOT, f"{name}_{datetime.now().strftime('%Y%m%d')}.log")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(DEFAULT_LOG_FORMAT)

    logger.addHandler(file_handler)

    return logger


# 快捷获取日志器
logger = get_logger()

if __name__ == '__main__':
    logger.info("信息日志")
    logger.error("错误日志")
    logger.warning("警告日志")



