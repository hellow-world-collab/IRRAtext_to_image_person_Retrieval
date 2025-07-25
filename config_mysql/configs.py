MYSQL_CONFIG = {
    'host': '127.0.0.1',       # 您的MySQL服务器地址 (如果是本机，通常是这个)
    'port': 3306,              # 您的MySQL服务器端口 (默认是3306)
    'user': 'root',            # 您的MySQL用户名
    'password': '123456', # 您的MySQL密码
    'database': 'vision_platform_db', # 我们创建的数据库名
    'charset': 'utf8mb4'
}
OSS_CONFIG = {
    # 将 'oss-cn-beijing.aliyuncs.com' 替换为您Bucket的实际Endpoint
    'endpoint': 'oss-cn-beijing.aliyuncs.com',
    'bucket_name':'webwjf',
    'Path':'data_Re_ID'
}
