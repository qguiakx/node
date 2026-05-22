from pydantic import BaseModel


class UploadFileModel(BaseModel):
    fileId: str
    filePath: str
    fileType: str

    @classmethod
    def build(cls, file_id, file_path, file_type):
        return cls(fileId=file_id, filePath=file_path, fileType=file_type)
