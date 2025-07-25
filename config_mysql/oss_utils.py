# 文件名: oss_utils.py
# 描述: 负责将文件上传到阿里云OSS

import os
import oss2
import logging
from uuid import uuid4
from config_mysql.configs import OSS_CONFIG  # 从配置文件导入Endpoint


def upload_to_oss(local_file_path: str) -> str or None:
    """
    上传本地文件到阿里云OSS的 'data_Re_ID' 目录下，并返回公开访问的URL。

    Args:
        local_file_path (str): 本地文件的完整路径。

    Returns:
        str or None: 如果上传成功，返回文件的公开URL；否则返回None。
    """
    try:
        access_key_id = os.environ['OSS_ACCESS_KEY_ID']
        access_key_secret = os.environ['OSS_ACCESS_KEY_SECRET']
        bucket_name = OSS_CONFIG['bucket_name']
        Path = OSS_CONFIG['Path']
    except KeyError as e:
        logging.error(f"OSS配置错误: 缺少环境变量 {e}。")
        return None

    endpoint = OSS_CONFIG['endpoint']
    auth = oss2.Auth(access_key_id, access_key_secret)
    bucket = oss2.Bucket(auth, f"http://{endpoint}", bucket_name)

    file_extension = os.path.splitext(local_file_path)[1]
    # 【核心修改】指定上传到 'data_Re_ID' 目录下
    object_name = f"{Path}/{uuid4().hex}{file_extension}"

    try:
        bucket.put_object_from_file(object_name, local_file_path)
        logging.info(f"文件 '{local_file_path}' 成功上传到OSS，路径为: '{object_name}'")

        public_url = f"https://{bucket_name}.{endpoint}/{object_name}"
        return public_url

    except oss2.exceptions.OssError as e:
        logging.error(f"上传到OSS时发生错误: {e}")
        return None
    except Exception as e:
        logging.error(f"上传文件时发生未知错误: {e}", exc_info=True)
        return None
