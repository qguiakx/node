"""
为项目提供统一的绝对路径
"""

import os


def get_project_path() -> str:
    """
    获取工程所在的目录
    :return 工程路径字符串
    """
    abspath = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(abspath)))
    return project_root


def get_abs_path(relative: str) -> str:
    """
    传递相对路径，得到绝对路径
    :param relative: 相对路径
    :return: 绝对路径
    """
    return os.path.join(get_project_path(), relative)


if __name__ == '__main__':
    print(get_abs_path("logs"))
