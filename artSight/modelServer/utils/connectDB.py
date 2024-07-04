from django.db import connection

def execute_query(query, params=None):
    with connection.cursor() as cursor:
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        
        if cursor.description:  # SELECT 쿼리인 경우
            columns = [col[0] for col in cursor.description]
            return [
                dict(zip(columns, row))
                for row in cursor.fetchall()
            ]
        return None  # INSERT, UPDATE, DELETE 등의 경우
