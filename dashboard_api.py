# filename: dashboard_api.py (Final Fix)
# description: This version correctly handles database results as tuples,
#              fixing the "'tuple' object has no attribute 'get'" error.

from fastapi import APIRouter, HTTPException
from config_mysql.database import get_db_connection
import logging
from datetime import datetime, timedelta

router = APIRouter()

@router.get("/dashboard/stats")
async def get_dashboard_stats():
    """
    Provides core statistics for the dashboard.
    """
    total_searches_sql = "SELECT COUNT(*) FROM operation_history"
    today_searches_sql = "SELECT COUNT(*) FROM operation_history WHERE DATE(timestamp) = CURDATE()"
    type_distribution_sql = "SELECT operation_type, COUNT(*) FROM operation_history GROUP BY operation_type"
    daily_activity_sql = "SELECT DATE(timestamp), COUNT(*) FROM operation_history WHERE timestamp >= CURDATE() - INTERVAL 6 DAY GROUP BY DATE(timestamp) ORDER BY DATE(timestamp) ASC"

    connection = get_db_connection()
    if not connection:
        raise HTTPException(status_code=503, detail="Database connection unavailable.")

    try:
        with connection.cursor() as cursor:
            # --- FIX for total_searches ---
            cursor.execute(total_searches_sql)
            total_searches_result = cursor.fetchone()
            total_searches = total_searches_result[0] if total_searches_result else 0

            # --- FIX for today_searches ---
            cursor.execute(today_searches_sql)
            today_searches_result = cursor.fetchone()
            today_searches = today_searches_result[0] if today_searches_result else 0

            # --- FIX for type_distribution ---
            cursor.execute(type_distribution_sql)
            type_distribution_raw = cursor.fetchall()
            # Convert list of tuples to list of dictionaries for the frontend
            type_distribution = [{'operation_type': row[0], 'count': row[1]} for row in type_distribution_raw]

            # --- FIX for daily_activity ---
            cursor.execute(daily_activity_sql)
            daily_activity_raw = cursor.fetchall()
            # Prepare a dictionary for the last 7 days
            daily_activity = {}
            for i in range(7):
                date_key = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
                daily_activity[date_key] = 0
            # Populate with data from the database
            for row in daily_activity_raw:
                if row and row[0]:
                    date_str = row[0].strftime('%Y-%m-%d')
                    daily_activity[date_str] = row[1]

    except Exception as e:
        logging.error(f"Error querying dashboard stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Database query failed.")
    finally:
        if connection:
            connection.close()

    return {
        "total_searches": total_searches,
        "today_searches": today_searches,
        "type_distribution": type_distribution,
        "daily_activity": daily_activity
    }


@router.get("/dashboard/top-queries")
async def get_top_queries(limit: int = 15):
    """
    Gets the most popular search queries.
    """
    sql = "SELECT query_text, COUNT(*) FROM operation_history WHERE query_text IS NOT NULL AND query_text != '' GROUP BY query_text ORDER BY COUNT(*) DESC LIMIT %s"

    connection = get_db_connection()
    if not connection:
        raise HTTPException(status_code=503, detail="Database connection unavailable.")

    try:
        with connection.cursor() as cursor:
            # --- FIX for top_queries ---
            cursor.execute(sql, (limit,))
            top_queries_raw = cursor.fetchall()
            # Convert list of tuples to list of dictionaries for the frontend
            top_queries = [{'query_text': row[0], 'count': row[1]} for row in top_queries_raw]

    except Exception as e:
        logging.error(f"Error querying top queries: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Database query failed.")
    finally:
        if connection:
            connection.close()

    return top_queries