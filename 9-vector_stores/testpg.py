import psycopg

try:
    conn = psycopg.connect(
        "postgresql://postgres:23@localhost:5432/demo_db"
    )
    print("✅ Database connection successful!")
    conn.close()
except Exception as e:
    print("❌ Failed to connect to the database:")
    print(e)
