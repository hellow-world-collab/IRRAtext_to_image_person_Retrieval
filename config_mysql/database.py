# 文件名: database.py (MySQL版本 - 支持高效分页)
# 描述: 负责所有 MySQL 数据库操作，增加了高效分页查询功能

import pymysql.cursors
import logging
from threading import Lock
from .configs import MYSQL_CONFIG  # 从新配置文件导入连接信息

db_lock = Lock()


def get_db_connection():
    """建立并返回一个MySQL数据库连接。"""
    try:
        connection = pymysql.connect(**MYSQL_CONFIG, cursorclass=pymysql.cursors.DictCursor)
        return connection
    except pymysql.MySQLError as e:
        logging.error(f"无法连接到MySQL数据库: {e}")
        return None


def add_history_record(operation_type: str, query_text: str, result_url: str, timestamp: str, details: str = "{}"):
    """向MySQL数据库中添加一条新的历史记录。"""
    # ... 此函数保持不变 ...
    sql = "INSERT INTO `operation_history` (`operation_type`, `query_text`, `result_url`, `timestamp`, `details`) VALUES (%s, %s, %s, %s, %s)"
    with db_lock:
        connection = get_db_connection()
        if connection:
            try:
                with connection.cursor() as cursor:
                    cursor.execute(sql, (operation_type, query_text, result_url, timestamp, details))
                connection.commit()
                logging.info(f"成功添加一条历史记录到MySQL: {operation_type}")
            except pymysql.MySQLError as e:
                logging.error(f"添加历史记录到MySQL时出错: {e}")
            finally:
                connection.close()


# ==================== 【核心修改点：新增分页查询函数】 ====================
def get_history_paginated(page: int, limit: int):
    """
    从MySQL数据库中分页获取历史记录。
    返回包含项目列表和总项目数的字典。
    """
    count_sql = "SELECT COUNT(*) as total FROM `operation_history`"
    data_sql = "SELECT * FROM `operation_history` ORDER BY `timestamp` DESC LIMIT %s OFFSET %s"

    offset = (page - 1) * limit

    with db_lock:
        connection = get_db_connection()
        if connection:
            try:
                with connection.cursor() as cursor:
                    # 1. 获取总记录数
                    cursor.execute(count_sql)
                    total_items = cursor.fetchone()['total']

                    # 2. 获取当前页的数据
                    cursor.execute(data_sql, (limit, offset))
                    items = cursor.fetchall()

                    return {"items": items, "total": total_items}
            except pymysql.MySQLError as e:
                logging.error(f"从MySQL分页获取历史记录时出错: {e}")
                return {"items": [], "total": 0}
            finally:
                connection.close()
        return {"items": [], "total": 0}


def clear_all_history():
    """清空MySQL数据库中的所有历史记录。"""
    # ... 此函数保持不变 ...
    sql = "DELETE FROM `operation_history`"
    with db_lock:
        connection = get_db_connection()
        if connection:
            try:
                with connection.cursor() as cursor:
                    cursor.execute(sql)
                connection.commit()
                logging.info("MySQL历史记录已清空。")
            finally:
                connection.close()