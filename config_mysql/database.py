# 文件名: database.py (已优化为使用连接池)
# 描述: 引入了数据库连接池来提升性能

import pymysql.cursors
import logging
from threading import Lock
from dbutils.pooled_db import PooledDB
from .configs import MYSQL_CONFIG

# --- 1. 创建数据库连接池 ---
# 应用启动时，这个池就会被创建一次，并在整个生命周期中被复用
try:
    pool = PooledDB(
        creator=pymysql,  # 使用 PyMySQL 作为驱动
        maxconnections=10,  # 池中保留的最大连接数
        mincached=2,      # 启动时初始化的最小空闲连接数
        blocking=True,    # 连接数达到上限时，新的请求会等待
        **MYSQL_CONFIG    # 传入您的数据库配置
    )
    logging.info("数据库连接池创建成功。")
except Exception as e:
    logging.error(f"无法创建数据库连接池: {e}")
    pool = None

db_lock = Lock()

# --- 2. 修改获取连接的函数 ---
def get_db_connection():
    """从连接池中获取一个数据库连接。"""
    if pool:
        return pool.connection()
    logging.error("无法从连接池获取连接，因为连接池未初始化。")
    return None

# --- 3. 所有数据库操作函数的逻辑保持不变 ---
#    现在它们获取和关闭连接的开销极小

def add_history_record(operation_type: str, query_text: str, result_url: str, timestamp: str, details: str = "{}"):
    """向MySQL数据库中添加一条新的历史记录。"""
    sql = "INSERT INTO `operation_history` (`operation_type`, `query_text`, `result_url`, `timestamp`, `details`) VALUES (%s, %s, %s, %s, %s)"
    with db_lock:
        connection = get_db_connection() # 从池中借用连接
        if connection:
            try:
                with connection.cursor(pymysql.cursors.DictCursor) as cursor: # 注意这里要指定DictCursor
                    cursor.execute(sql, (operation_type, query_text, result_url, timestamp, details))
                connection.commit()
                logging.info(f"成功添加一条历史记录到MySQL: {operation_type}")
            except pymysql.MySQLError as e:
                logging.error(f"添加历史记录到MySQL时出错: {e}")
            finally:
                connection.close() # 将连接还回池中

def get_history_paginated(page: int, limit: int):
    """从MySQL数据库中分页获取历史记录。"""
    count_sql = "SELECT COUNT(*) as total FROM `operation_history`"
    data_sql = "SELECT * FROM `operation_history` ORDER BY `timestamp` DESC LIMIT %s OFFSET %s"
    offset = (page - 1) * limit

    with db_lock:
        connection = get_db_connection() # 从池中借用连接
        if connection:
            try:
                with connection.cursor(pymysql.cursors.DictCursor) as cursor:
                    cursor.execute(count_sql)
                    total_items = cursor.fetchone()['total']
                    cursor.execute(data_sql, (limit, offset))
                    items = cursor.fetchall()
                    return {"items": items, "total": total_items}
            except pymysql.MySQLError as e:
                logging.error(f"从MySQL分页获取历史记录时出错: {e}")
                return {"items": [], "total": 0}
            finally:
                connection.close() # 将连接还回池中
        return {"items": [], "total": 0}

def clear_all_history():
    """清空MySQL数据库中的所有历史记录。"""
    sql = "DELETE FROM `operation_history`"
    with db_lock:
        connection = get_db_connection() # 从池中借用连接
        if connection:
            try:
                with connection.cursor() as cursor:
                    cursor.execute(sql)
                connection.commit()
                logging.info("MySQL历史记录已清空。")
            finally:
                connection.close() # 将连接还回池中

def delete_history_ids(ids):
    """根据提供的ID列表删除历史记录。"""
    if not ids:
        return False
    placeholders = ', '.join(['%s'] * len(ids))
    sql = f"DELETE FROM `operation_history` WHERE id IN ({placeholders})"
    with db_lock:
        connection = get_db_connection() # 从池中借用连接
        if connection:
            try:
                with connection.cursor() as cursor:
                    cursor.execute(sql, tuple(ids))
                connection.commit()
                logging.info(f"成功删除了ID为 {ids} 的历史记录。")
                return True
            except pymysql.MySQLError as e:
                logging.error(f"删除历史记录时出错: {e}")
                return False
            finally:
                connection.close() # 将连接还回池中
    return False