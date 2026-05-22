import os
import hashlib
import src.config.config as config


def check_md5(md5_str: str) -> bool:
    """
    检查传入的md5文件是否已经被处理过
    return False 表示md5文件未被处理 True 表示已经被处理过
    """
    if not os.path.exists(config.MD5_PATH):
        # 创建路径并打开
        open(config.MD5_PATH, 'w', encoding="utf-8").close()
        return False
    with open(config.MD5_PATH, 'r', encoding="utf-8") as f:
        return md5_str in [line.strip() for line in f.readlines()]


def save_md5(md_str: str):
    """将传入的md5字符串，保存到文件中"""
    with open(config.MD5_PATH, 'a', encoding="utf-8") as f:
        f.write(md_str + '\n')


def get_md5(input_str: str) -> str:
    """将传入的字符串转换为md5字符串"""
    str_bytes = input_str.encode(encoding="utf-8")
    # 得到md516进制字符串
    return hashlib.md5(str_bytes).hexdigest()