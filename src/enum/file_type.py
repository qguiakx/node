from enum import Enum


class FileType(Enum):
    """文件类型枚举类"""
    CSV = ("csv",)
    PDF = ("pdf",)
    PNG = ("png",)
    JPG = ("jpg", "jpeg")
    MD = ("md", "markdown")
    EXCEL = ("xlsx", "xls")
    UNKNOWN = ()

    def __init__(self, *extensions):
        self.extensions = extensions

    @classmethod
    def from_extension(cls, extension: str):
        """根据文件扩展名获取文件类型"""
        extension = extension.lower().lstrip('.')
        for file_type in cls:
            if extension in file_type.extensions:
                return file_type
        return cls.UNKNOWN

    @classmethod
    def get_type_name(cls, extension: str) -> str:
        """根据文件扩展名获取文件类型名称（小写）"""
        file_type = cls.from_extension(extension)
        return file_type.name.lower() if file_type != cls.UNKNOWN else "unknown"


# 使用示例
if __name__ == "__main__":
    # 原来的逻辑可以替换为：
    file_extension = "csv"
    file_type = FileType.get_type_name(file_extension)
    print(f"文件类型: {file_type}")  # 输出: 文件类型: csv

    # 或者直接获取枚举成员
    file_type_enum = FileType.from_extension("xlsx")
    print(f"枚举类型: {file_type_enum}")  # 输出: 枚举类型: FileType.EXCEL

    # 判断文件类型
    if file_type_enum == FileType.EXCEL:
        print("这是Excel文件")