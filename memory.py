"""
memory.py

Customer memory layer for SupportIQ.
Stores and retrieves customer interaction history using SQLite.

Three jobs:
    1. Initialize database tables (runs once at startup)
    2. Read customer history (called by Agent 3 before generating response)
    3. Write ticket record (called after Agent 5 approves response)

SQLite chosen over flat JSON for:
    - Structured queries (SELECT by customer_id, ORDER BY date)
    - No external server needed (single file: data/customers/customers.db)
    - Proper referential integrity (tickets linked to customers via FK)
"""

import sqlite3
from pathlib import Path
from datetime import datetime

# Database file location
DB_PATH = Path("data/customers/customers.db")


def init_db():
    """
    Creates database tables if they don't exist.
    Safe to call multiple times — IF NOT EXISTS prevents duplication.
    Called once at application startup.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS customers (
                customer_id    TEXT PRIMARY KEY,
                name           TEXT,
                vip_status     INTEGER DEFAULT 0,
                sentiment      TEXT DEFAULT 'neutral',
                total_tickets  INTEGER DEFAULT 0,
                created_at     TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tickets (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id    TEXT,
                ticket_text    TEXT,
                intent         TEXT,
                priority       TEXT,
                resolution     TEXT,
                created_at     TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
            )
        """)

        conn.commit()
        print(f"Database initialized at {DB_PATH}")


def get_customer_history(customer_id: str) -> dict:
    """
    Retrieves customer profile and last 5 tickets for a given customer_id.
    Called by Agent 3 before generating a response.

    Returns a dict with:
        - found: bool (False if customer doesn't exist yet)
        - name, vip_status, sentiment, total_tickets
        - recent_tickets: list of last 5 ticket records
    """
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM customers WHERE customer_id = ?",
            (customer_id,)
        )
        customer = cursor.fetchone()

        if not customer:
            return {
                "found": False,
                "customer_id": customer_id,
                "name": "Valued Customer",
                "vip_status": False,
                "sentiment": "neutral",
                "total_tickets": 0,
                "recent_tickets": [],
            }

        cursor.execute(
            """
            SELECT ticket_text, intent, priority, resolution, created_at
            FROM tickets
            WHERE customer_id = ?
            ORDER BY created_at DESC
            LIMIT 5
            """,
            (customer_id,)
        )
        recent_tickets = [dict(row) for row in cursor.fetchall()]

        return {
            "found": True,
            "customer_id": customer_id,
            "name": customer["name"],
            "vip_status": bool(customer["vip_status"]),
            "sentiment": customer["sentiment"],
            "total_tickets": customer["total_tickets"],
            "recent_tickets": recent_tickets,
        }

def get_all_tickets(limit: int = 50) -> list:
    """
    Retrieves all tickets across all customers, most recent first.
    Used for the Streamlit ticket history/dashboard view.
    """
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT t.id, t.customer_id, c.name, t.ticket_text,
                   t.intent, t.priority, t.resolution, t.created_at
            FROM tickets t
            LEFT JOIN customers c ON t.customer_id = c.customer_id
            ORDER BY t.created_at DESC
            LIMIT ?
            """,
            (limit,)
        )

        return [dict(row) for row in cursor.fetchall()]

def save_ticket(
    customer_id: str,
    name: str,
    ticket_text: str,
    intent: str,
    priority: str,
    resolution: str,
) -> None:
    """
    Saves a resolved ticket to the database.
    Creates customer record if first time contact.
    Updates sentiment and total_tickets count.
    Called after Agent 5 approves the final response.
    """
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()

        cursor.execute(
            "SELECT customer_id, total_tickets FROM customers WHERE customer_id = ?",
            (customer_id,)
        )
        existing = cursor.fetchone()

        if not existing:
            cursor.execute(
                """
                INSERT INTO customers (customer_id, name, vip_status, sentiment, total_tickets)
                VALUES (?, ?, 0, 'neutral', 1)
                """,
                (customer_id, name)
            )
        else:
            cursor.execute(
                """
                UPDATE customers
                SET total_tickets = total_tickets + 1,
                    sentiment = ?
                WHERE customer_id = ?
                """,
                (_calculate_sentiment(existing[1] + 1, priority), customer_id)
            )

        cursor.execute(
            """
            INSERT INTO tickets (customer_id, ticket_text, intent, priority, resolution, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (customer_id, ticket_text, intent, priority, resolution,
             datetime.now().isoformat())
        )

        conn.commit()
        print(f"Ticket saved for customer {customer_id}")


def _calculate_sentiment(total_tickets: int, latest_priority: str) -> str:
    """
    Simple heuristic to update customer sentiment based on
    ticket history. More tickets + higher priority = more frustrated.

    Private function (underscore prefix) — internal use only.
    """
    if latest_priority == "critical":
        return "angry"
    elif latest_priority == "high" and total_tickets >= 3:
        return "frustrated"
    elif total_tickets >= 5:
        return "frustrated"
    elif latest_priority in ("low", "medium") and total_tickets <= 2:
        return "positive"
    else:
        return "neutral"


def seed_test_customers() -> None:
    """
    Adds synthetic test customers to the database.
    Run once to populate test data for development.
    """
    test_customers = [
        {
            "customer_id": "CUST-001",
            "name": "Rahul Sharma",
            "tickets": [
                {
                    "text": "My UPI payment failed last month",
                    "intent": "technical",
                    "priority": "high",
                    "resolution": "Refund processed within 3 days"
                },
                {
                    "text": "KYC documents not accepted",
                    "intent": "complaint",
                    "priority": "medium",
                    "resolution": "KYC re-submitted and approved"
                },
            ]
        },
        {
            "customer_id": "CUST-002",
            "name": "Priya Patel",
            "tickets": [
                {
                    "text": "What is the interest rate on FD?",
                    "intent": "inquiry",
                    "priority": "low",
                    "resolution": "Information provided"
                },
            ]
        },
        {
            "customer_id": "CUST-003",
            "name": "Amit Singh",
            "tickets": [
                {
                    "text": "Unauthorized transaction on my credit card",
                    "intent": "technical",
                    "priority": "critical",
                    "resolution": "Card blocked, fraud investigation initiated"
                },
                {
                    "text": "Still waiting for fraud refund after 2 weeks",
                    "intent": "escalation",
                    "priority": "critical",
                    "resolution": "Escalated to senior team, refund in 2 days"
                },
                {
                    "text": "Why is my account still frozen?",
                    "intent": "escalation",
                    "priority": "critical",
                    "resolution": "Account unfrozen, compensation offered"
                },
            ]
        },
    ]

    for customer in test_customers:
        for ticket in customer["tickets"]:
            save_ticket(
                customer_id=customer["customer_id"],
                name=customer["name"],
                ticket_text=ticket["text"],
                intent=ticket["intent"],
                priority=ticket["priority"],
                resolution=ticket["resolution"],
            )
    print("Test customers seeded successfully")


if __name__ == "__main__":
    print("Initializing database...")
    init_db()

    print("\nSeeding test customers...")
    seed_test_customers()

    print("\nTesting get_customer_history...")
    for customer_id in ["CUST-001", "CUST-002", "CUST-003", "CUST-999"]:
        history = get_customer_history(customer_id)
        print(f"\n{customer_id}:")
        print(f"  Found:          {history['found']}")
        print(f"  Name:           {history['name']}")
        print(f"  VIP:            {history['vip_status']}")
        print(f"  Sentiment:      {history['sentiment']}")
        print(f"  Total tickets:  {history['total_tickets']}")
        print(f"  Recent tickets: {len(history['recent_tickets'])}")