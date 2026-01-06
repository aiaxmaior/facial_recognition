#!/usr/bin/env python3
"""
Face Database Admin Tool
========================
CLI and web interface for managing the enrolled faces SQLite database.

Usage:
    python face_admin.py                    # Interactive CLI
    python face_admin.py --web              # Launch web interface
    python face_admin.py list               # List all enrolled users
    python face_admin.py search "John"      # Search for users
    python face_admin.py delete "John Doe"  # Delete a user
    python face_admin.py info "John Doe"    # Show user details
    python face_admin.py query "SELECT * FROM faces"  # Run custom SQL
    python face_admin.py export users.csv   # Export to CSV
"""

import os
import sys
import sqlite3
import argparse
import numpy as np
from datetime import datetime
from typing import Optional, List, Dict, Any
import csv

# Default database path
DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "enrolled_faces",
    "faces.db"
)


class FaceDatabase:
    """SQLite database interface for enrolled faces."""
    
    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        if not os.path.exists(db_path):
            print(f"⚠️  Database not found: {db_path}")
            print("   Run the enrollment system first to create the database.")
            self.exists = False
        else:
            self.exists = True
    
    def _connect(self):
        """Create a database connection."""
        return sqlite3.connect(self.db_path)
    
    def list_users(self, limit: int = 100) -> List[Dict[str, Any]]:
        """List all enrolled users."""
        if not self.exists:
            return []
        
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, name, model, detector, image_count, enrolled_at
            FROM faces
            ORDER BY enrolled_at DESC
            LIMIT ?
        """, (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def search_users(self, query: str) -> List[Dict[str, Any]]:
        """Search for users by name."""
        if not self.exists:
            return []
        
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, name, model, detector, image_count, enrolled_at
            FROM faces
            WHERE name LIKE ?
            ORDER BY name
        """, (f"%{query}%",))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def get_user(self, name: str) -> Optional[Dict[str, Any]]:
        """Get detailed info for a specific user."""
        if not self.exists:
            return None
        
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM faces WHERE name = ?", (name,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            data = dict(row)
            # Decode embedding info
            if data.get('embedding'):
                emb = np.frombuffer(data['embedding'], dtype=np.float64)
                data['embedding_dim'] = len(emb)
                data['embedding_preview'] = emb[:5].tolist()  # First 5 values
                del data['embedding']  # Don't include raw bytes
            if data.get('embedding_normalized'):
                del data['embedding_normalized']  # Don't include raw bytes
            return data
        return None
    
    def delete_user(self, name: str) -> bool:
        """Delete a user by name."""
        if not self.exists:
            return False
        
        conn = self._connect()
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM faces WHERE name = ?", (name,))
        deleted = cursor.rowcount > 0
        conn.commit()
        conn.close()
        
        return deleted
    
    def run_query(self, sql: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """Run a custom SQL query (SELECT only for safety)."""
        if not self.exists:
            return []
        
        # Safety check - only allow SELECT
        sql_upper = sql.strip().upper()
        if not sql_upper.startswith("SELECT"):
            raise ValueError("Only SELECT queries are allowed. Use delete_user() for deletions.")
        
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        conn.close()
        
        # Convert rows, handling BLOB columns
        results = []
        for row in rows:
            d = dict(row)
            # Replace BLOB columns with info
            for key, value in d.items():
                if isinstance(value, bytes):
                    d[key] = f"<BLOB {len(value)} bytes>"
            results.append(d)
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        if not self.exists:
            return {"error": "Database not found"}
        
        conn = self._connect()
        cursor = conn.cursor()
        
        stats = {}
        
        # Total users
        cursor.execute("SELECT COUNT(*) FROM faces")
        stats['total_users'] = cursor.fetchone()[0]
        
        # Users by model
        cursor.execute("SELECT model, COUNT(*) as count FROM faces GROUP BY model")
        stats['by_model'] = dict(cursor.fetchall())
        
        # Users by detector
        cursor.execute("SELECT detector, COUNT(*) as count FROM faces GROUP BY detector")
        stats['by_detector'] = dict(cursor.fetchall())
        
        # Recent enrollments (last 7 days)
        cursor.execute("""
            SELECT COUNT(*) FROM faces 
            WHERE enrolled_at > datetime('now', '-7 days')
        """)
        stats['enrolled_last_7_days'] = cursor.fetchone()[0]
        
        # Database file size
        stats['db_size_kb'] = round(os.path.getsize(self.db_path) / 1024, 2)
        
        conn.close()
        return stats
    
    def export_csv(self, output_path: str) -> int:
        """Export users to CSV (excluding embeddings)."""
        if not self.exists:
            return 0
        
        users = self.list_users(limit=10000)
        
        if not users:
            return 0
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=users[0].keys())
            writer.writeheader()
            writer.writerows(users)
        
        return len(users)


# Default events database path
DEFAULT_EVENTS_DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "enrolled_faces",
    "events.db"
)


class EventDatabase:
    """SQLite database interface for recognition events."""
    
    def __init__(self, db_path: str = DEFAULT_EVENTS_DB_PATH):
        self.db_path = db_path
        if not os.path.exists(db_path):
            print(f"⚠️  Events database not found: {db_path}")
            print("   Run the live stream recognizer to start logging events.")
            self.exists = False
        else:
            self.exists = True
    
    def _connect(self):
        """Create a database connection."""
        return sqlite3.connect(self.db_path)
    
    def get_recent_events(self, limit: int = 100, event_type: str = None) -> List[Dict[str, Any]]:
        """Get recent events, optionally filtered by type."""
        if not self.exists:
            return []
        
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        if event_type:
            cursor.execute(
                "SELECT * FROM events WHERE event_type = ? ORDER BY timestamp DESC LIMIT ?",
                (event_type, limit)
            )
        else:
            cursor.execute(
                "SELECT * FROM events ORDER BY timestamp DESC LIMIT ?",
                (limit,)
            )
        
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]
    
    def get_recognitions(self, name: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recognition events, optionally filtered by name."""
        if not self.exists:
            return []
        
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        if name:
            cursor.execute(
                """SELECT * FROM events 
                   WHERE event_type = 'RECOGNITION' AND name LIKE ?
                   ORDER BY timestamp DESC LIMIT ?""",
                (f"%{name}%", limit)
            )
        else:
            cursor.execute(
                """SELECT * FROM events 
                   WHERE event_type = 'RECOGNITION'
                   ORDER BY timestamp DESC LIMIT ?""",
                (limit,)
            )
        
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get event statistics."""
        if not self.exists:
            return {"error": "Database not found"}
        
        conn = self._connect()
        cursor = conn.cursor()
        
        stats = {}
        
        # Total events by type
        cursor.execute("""
            SELECT event_type, COUNT(*) as count 
            FROM events GROUP BY event_type
        """)
        stats['by_type'] = dict(cursor.fetchall())
        
        # Total recognitions by person
        cursor.execute("""
            SELECT name, COUNT(*) as count 
            FROM events 
            WHERE event_type = 'RECOGNITION'
            GROUP BY name
            ORDER BY count DESC
            LIMIT 20
        """)
        stats['by_person'] = dict(cursor.fetchall())
        
        # Total events
        cursor.execute("SELECT COUNT(*) FROM events")
        stats['total_events'] = cursor.fetchone()[0]
        
        # Time range
        cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM events")
        row = cursor.fetchone()
        stats['first_event'] = row[0]
        stats['last_event'] = row[1]
        
        # Events today
        cursor.execute("""
            SELECT COUNT(*) FROM events 
            WHERE date(timestamp) = date('now')
        """)
        stats['events_today'] = cursor.fetchone()[0]
        
        # Database size
        stats['db_size_kb'] = round(os.path.getsize(self.db_path) / 1024, 2)
        
        conn.close()
        return stats
    
    def run_query(self, sql: str) -> List[Dict[str, Any]]:
        """Run a custom SELECT query."""
        if not self.exists:
            return []
        
        if not sql.strip().upper().startswith("SELECT"):
            raise ValueError("Only SELECT queries allowed")
        
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]
    
    def export_csv(self, output_path: str, event_type: str = None) -> int:
        """Export events to CSV file."""
        if not self.exists:
            return 0
        
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        if event_type:
            cursor.execute(
                "SELECT * FROM events WHERE event_type = ? ORDER BY timestamp",
                (event_type,)
            )
        else:
            cursor.execute("SELECT * FROM events ORDER BY timestamp")
        
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return 0
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            for row in rows:
                writer.writerow(dict(row))
        
        return len(rows)
    
    def clear_old_events(self, days: int = 30) -> int:
        """Delete events older than specified days."""
        if not self.exists:
            return 0
        
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM events WHERE timestamp < datetime('now', ?)",
            (f"-{days} days",)
        )
        deleted = cursor.rowcount
        conn.commit()
        conn.close()
        return deleted


def format_timestamp(ts: str) -> str:
    """Format a timestamp for display."""
    if not ts:
        return "N/A"
    try:
        # Handle Unix timestamp
        if isinstance(ts, (int, float)) or ts.replace('.', '').isdigit():
            dt = datetime.fromtimestamp(float(ts))
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        return ts
    except:
        return str(ts)


def print_table(rows: List[Dict], max_width: int = 100):
    """Print rows as a formatted table."""
    if not rows:
        print("  (no results)")
        return
    
    # Get column widths
    columns = list(rows[0].keys())
    widths = {col: len(col) for col in columns}
    
    for row in rows:
        for col in columns:
            val = str(row.get(col, ''))
            if col == 'enrolled_at':
                val = format_timestamp(val)
            widths[col] = min(max(widths[col], len(val)), 40)
    
    # Print header
    header = " | ".join(col.ljust(widths[col])[:widths[col]] for col in columns)
    print(f"  {header}")
    print(f"  {'-' * len(header)}")
    
    # Print rows
    for row in rows:
        values = []
        for col in columns:
            val = str(row.get(col, ''))
            if col == 'enrolled_at':
                val = format_timestamp(val)
            values.append(val.ljust(widths[col])[:widths[col]])
        print(f"  {' | '.join(values)}")


def interactive_cli(db: FaceDatabase, events_db: EventDatabase = None):
    """Run interactive CLI mode."""
    print("\n" + "=" * 70)
    print("  Face Database Admin - Interactive Mode")
    print("=" * 70)
    print(f"  Faces DB:  {db.db_path}")
    if events_db:
        print(f"  Events DB: {events_db.db_path}")
    print("  Commands: list, search, info, delete, stats, query, export, help, quit")
    if events_db:
        print("  Events:   events, recognitions, event-stats, event-export, clear-old")
    print("=" * 70 + "\n")
    
    while True:
        try:
            cmd = input("face_admin> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        
        if not cmd:
            continue
        
        parts = cmd.split(maxsplit=1)
        action = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""
        
        if action in ('quit', 'exit', 'q'):
            print("Goodbye!")
            break
        
        elif action == 'help':
            print("""
  === Face Database Commands ===
  list              - List all enrolled users
  search <name>     - Search for users by name
  info <name>       - Show detailed info for a user
  delete <name>     - Delete a user (with confirmation)
  stats             - Show database statistics
  query <sql>       - Run a custom SELECT query
  export <file.csv> - Export users to CSV

  === Event Log Commands ===
  events [limit]           - Show recent events (default: 50)
  recognitions [name]      - Show recognition events (optionally filter by name)
  event-stats              - Show event statistics
  event-export <file.csv>  - Export events to CSV
  clear-old <days>         - Clear events older than N days

  quit              - Exit the program
            """)
        
        elif action == 'list':
            print("\n📋 Enrolled Users:")
            users = db.list_users()
            print_table(users)
            print(f"\n  Total: {len(users)} users\n")
        
        elif action == 'search':
            if not arg:
                print("  Usage: search <name>")
                continue
            print(f"\n🔍 Searching for '{arg}':")
            users = db.search_users(arg)
            print_table(users)
            print(f"\n  Found: {len(users)} users\n")
        
        elif action == 'info':
            if not arg:
                print("  Usage: info <name>")
                continue
            user = db.get_user(arg)
            if user:
                print(f"\n👤 User: {arg}")
                print("  " + "-" * 40)
                for key, value in user.items():
                    if key == 'enrolled_at':
                        value = format_timestamp(value)
                    print(f"  {key}: {value}")
                print()
            else:
                print(f"  User '{arg}' not found.\n")
        
        elif action == 'delete':
            if not arg:
                print("  Usage: delete <name>")
                continue
            confirm = input(f"  Delete '{arg}'? (yes/no): ").strip().lower()
            if confirm == 'yes':
                if db.delete_user(arg):
                    print(f"  ✅ Deleted '{arg}'\n")
                else:
                    print(f"  ❌ User '{arg}' not found.\n")
            else:
                print("  Cancelled.\n")
        
        elif action == 'stats':
            print("\n📊 Database Statistics:")
            stats = db.get_stats()
            print("  " + "-" * 40)
            for key, value in stats.items():
                print(f"  {key}: {value}")
            print()
        
        elif action == 'query':
            if not arg:
                print("  Usage: query <SELECT ...>")
                continue
            try:
                results = db.run_query(arg)
                print(f"\n📝 Query Results:")
                print_table(results)
                print(f"\n  Rows: {len(results)}\n")
            except Exception as e:
                print(f"  ❌ Error: {e}\n")
        
        elif action == 'export':
            if not arg:
                print("  Usage: export <filename.csv>")
                continue
            count = db.export_csv(arg)
            print(f"  ✅ Exported {count} users to {arg}\n")
        
        # === Event Log Commands ===
        elif action == 'events':
            if not events_db or not events_db.exists:
                print("  ❌ Events database not available.\n")
                continue
            limit = int(arg) if arg.isdigit() else 50
            print(f"\n📋 Recent Events (last {limit}):")
            events = events_db.get_recent_events(limit=limit)
            print_table(events)
            print(f"\n  Total shown: {len(events)}\n")
        
        elif action == 'recognitions':
            if not events_db or not events_db.exists:
                print("  ❌ Events database not available.\n")
                continue
            print(f"\n✅ Recognition Events" + (f" for '{arg}':" if arg else ":"))
            events = events_db.get_recognitions(name=arg if arg else None, limit=100)
            print_table(events)
            print(f"\n  Total shown: {len(events)}\n")
        
        elif action == 'event-stats':
            if not events_db or not events_db.exists:
                print("  ❌ Events database not available.\n")
                continue
            print("\n📊 Event Statistics:")
            stats = events_db.get_stats()
            print("  " + "-" * 40)
            for key, value in stats.items():
                print(f"  {key}: {value}")
            print()
        
        elif action == 'event-export':
            if not events_db or not events_db.exists:
                print("  ❌ Events database not available.\n")
                continue
            if not arg:
                print("  Usage: event-export <filename.csv>")
                continue
            count = events_db.export_csv(arg)
            print(f"  ✅ Exported {count} events to {arg}\n")
        
        elif action == 'clear-old':
            if not events_db or not events_db.exists:
                print("  ❌ Events database not available.\n")
                continue
            if not arg or not arg.isdigit():
                print("  Usage: clear-old <days>")
                continue
            days = int(arg)
            confirm = input(f"  Delete events older than {days} days? (yes/no): ").strip().lower()
            if confirm == 'yes':
                deleted = events_db.clear_old_events(days)
                print(f"  ✅ Deleted {deleted} old events.\n")
            else:
                print("  Cancelled.\n")
        
        else:
            print(f"  Unknown command: {action}. Type 'help' for available commands.\n")


def create_web_interface(db: FaceDatabase, events_db: EventDatabase = None):
    """Create a Gradio web interface for database management."""
    try:
        import gradio as gr
    except ImportError:
        print("Error: Gradio is required for web interface. Install with: pip install gradio")
        sys.exit(1)
    
    def list_all_users():
        users = db.list_users()
        if not users:
            return "No users enrolled."
        
        lines = ["| ID | Name | Model | Detector | Images | Enrolled |",
                 "|:---|:-----|:------|:---------|:-------|:---------|"]
        for u in users:
            enrolled = format_timestamp(u.get('enrolled_at', ''))
            lines.append(f"| {u['id']} | {u['name']} | {u['model']} | {u['detector']} | {u['image_count']} | {enrolled} |")
        return "\n".join(lines)
    
    def search_users(query):
        if not query:
            return "Enter a search term."
        users = db.search_users(query)
        if not users:
            return f"No users found matching '{query}'."
        
        lines = ["| ID | Name | Model | Images | Enrolled |",
                 "|:---|:-----|:------|:-------|:---------|"]
        for u in users:
            enrolled = format_timestamp(u.get('enrolled_at', ''))
            lines.append(f"| {u['id']} | {u['name']} | {u['model']} | {u['image_count']} | {enrolled} |")
        return "\n".join(lines)
    
    def get_user_info(name):
        if not name:
            return "Enter a user name."
        user = db.get_user(name)
        if not user:
            return f"User '{name}' not found."
        
        lines = [f"## 👤 {name}", ""]
        for key, value in user.items():
            if key == 'enrolled_at':
                value = format_timestamp(value)
            lines.append(f"- **{key}**: {value}")
        return "\n".join(lines)
    
    def delete_user(name):
        if not name:
            return "Enter a user name to delete."
        if db.delete_user(name):
            return f"✅ Successfully deleted '{name}'"
        return f"❌ User '{name}' not found."
    
    def run_sql_query(sql):
        if not sql:
            return "Enter a SQL query."
        try:
            results = db.run_query(sql)
            if not results:
                return "Query returned no results."
            
            # Build markdown table
            cols = list(results[0].keys())
            lines = ["| " + " | ".join(cols) + " |",
                     "|" + "|".join(["---"] * len(cols)) + "|"]
            for row in results[:100]:  # Limit to 100 rows
                vals = [str(row.get(c, ''))[:50] for c in cols]
                lines.append("| " + " | ".join(vals) + " |")
            
            if len(results) > 100:
                lines.append(f"\n*...and {len(results) - 100} more rows*")
            
            return "\n".join(lines)
        except Exception as e:
            return f"❌ Error: {e}"
    
    def get_stats():
        stats = db.get_stats()
        lines = ["## 📊 Database Statistics", ""]
        for key, value in stats.items():
            lines.append(f"- **{key}**: {value}")
        return "\n".join(lines)
    
    # Event log functions
    def get_recent_events(limit_str):
        if not events_db or not events_db.exists:
            return "Events database not available."
        limit = int(limit_str) if limit_str.isdigit() else 50
        events = events_db.get_recent_events(limit=limit)
        if not events:
            return "No events found."
        
        cols = ["id", "timestamp", "event_type", "name", "confidence", "is_match", "camera_id"]
        lines = ["| " + " | ".join(cols) + " |",
                 "|" + "|".join(["---"] * len(cols)) + "|"]
        for e in events:
            vals = []
            for c in cols:
                v = e.get(c, '')
                if c == 'confidence' and v is not None:
                    v = f"{v:.3f}"
                elif c == 'is_match':
                    v = "✅" if v == 1 else "❌" if v == 0 else ""
                vals.append(str(v)[:30])
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)
    
    def get_recognitions_web(name_filter):
        if not events_db or not events_db.exists:
            return "Events database not available."
        events = events_db.get_recognitions(name=name_filter if name_filter else None, limit=100)
        if not events:
            return "No recognition events found."
        
        cols = ["timestamp", "name", "confidence", "distance", "camera_id"]
        lines = ["| " + " | ".join(cols) + " |",
                 "|" + "|".join(["---"] * len(cols)) + "|"]
        for e in events:
            vals = []
            for c in cols:
                v = e.get(c, '')
                if c in ('confidence', 'distance') and v is not None:
                    v = f"{v:.4f}"
                vals.append(str(v)[:40])
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)
    
    def get_event_stats():
        if not events_db or not events_db.exists:
            return "Events database not available."
        stats = events_db.get_stats()
        lines = ["## 📊 Event Statistics", ""]
        for key, value in stats.items():
            lines.append(f"- **{key}**: {value}")
        return "\n".join(lines)
    
    # Build interface
    with gr.Blocks(
        title="Face Database Admin",
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
        css="footer { display: none !important; }"
    ) as interface:
        gr.Markdown("# 🗃️ Face Database Admin")
        gr.Markdown(f"Faces: `{db.db_path}`" + (f" | Events: `{events_db.db_path}`" if events_db else ""))
        
        with gr.Tabs():
            with gr.Tab("📋 Users"):
                list_btn = gr.Button("Refresh List", variant="primary")
                users_output = gr.Markdown()
                list_btn.click(fn=list_all_users, outputs=users_output)
            
            with gr.Tab("🔍 Search"):
                search_input = gr.Textbox(label="Search Name", placeholder="Enter name to search...")
                search_btn = gr.Button("Search", variant="primary")
                search_output = gr.Markdown()
                search_btn.click(fn=search_users, inputs=search_input, outputs=search_output)
            
            with gr.Tab("👤 User Info"):
                info_input = gr.Textbox(label="User Name", placeholder="Enter exact user name...")
                info_btn = gr.Button("Get Info", variant="primary")
                info_output = gr.Markdown()
                info_btn.click(fn=get_user_info, inputs=info_input, outputs=info_output)
            
            with gr.Tab("🗑️ Delete"):
                delete_input = gr.Textbox(label="User Name to Delete", placeholder="Enter exact user name...")
                delete_btn = gr.Button("Delete User", variant="stop")
                delete_output = gr.Markdown()
                delete_btn.click(fn=delete_user, inputs=delete_input, outputs=delete_output)
            
            with gr.Tab("📝 SQL Query"):
                query_input = gr.Textbox(
                    label="SQL Query",
                    placeholder="SELECT * FROM faces WHERE name LIKE '%John%'",
                    lines=3
                )
                query_btn = gr.Button("Execute Query", variant="primary")
                query_output = gr.Markdown()
                query_btn.click(fn=run_sql_query, inputs=query_input, outputs=query_output)
            
            with gr.Tab("📊 Stats"):
                stats_btn = gr.Button("Refresh Stats", variant="primary")
                stats_output = gr.Markdown()
                stats_btn.click(fn=get_stats, outputs=stats_output)
            
            # Event Log Tabs (only if events_db is available)
            if events_db:
                with gr.Tab("📜 Event Log"):
                    with gr.Row():
                        event_limit = gr.Textbox(label="Limit", value="50", scale=1)
                        event_refresh_btn = gr.Button("Refresh Events", variant="primary", scale=2)
                    events_output = gr.Markdown()
                    event_refresh_btn.click(fn=get_recent_events, inputs=event_limit, outputs=events_output)
                
                with gr.Tab("✅ Recognitions"):
                    recog_filter = gr.Textbox(label="Filter by Name (optional)", placeholder="Enter name to filter...")
                    recog_btn = gr.Button("Show Recognitions", variant="primary")
                    recog_output = gr.Markdown()
                    recog_btn.click(fn=get_recognitions_web, inputs=recog_filter, outputs=recog_output)
                
                with gr.Tab("📈 Event Stats"):
                    event_stats_btn = gr.Button("Refresh Event Stats", variant="primary")
                    event_stats_output = gr.Markdown()
                    event_stats_btn.click(fn=get_event_stats, outputs=event_stats_output)
    
    return interface


def main():
    parser = argparse.ArgumentParser(
        description="Face Database Admin Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=["list", "search", "info", "delete", "query", "export", "stats",
                 "events", "recognitions", "event-stats", "event-export"],
        help="Command to run (or omit for interactive mode)"
    )
    parser.add_argument(
        "argument",
        nargs="?",
        help="Argument for the command (name, query, filename)"
    )
    parser.add_argument(
        "--db", "-d",
        default=DEFAULT_DB_PATH,
        help=f"Path to faces database file (default: {DEFAULT_DB_PATH})"
    )
    parser.add_argument(
        "--events-db",
        default=DEFAULT_EVENTS_DB_PATH,
        help=f"Path to events database file (default: {DEFAULT_EVENTS_DB_PATH})"
    )
    parser.add_argument(
        "--web", "-w",
        action="store_true",
        help="Launch web interface instead of CLI"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=7862,
        help="Port for web interface (default: 7862)"
    )
    
    args = parser.parse_args()
    
    # Initialize databases
    db = FaceDatabase(args.db)
    events_db = EventDatabase(args.events_db)
    
    # Web interface mode
    if args.web:
        interface = create_web_interface(db, events_db if events_db.exists else None)
        print(f"\n🌐 Launching Face Database Admin on http://localhost:{args.port}\n")
        interface.launch(
            server_name="0.0.0.0",
            server_port=args.port,
            share=False,
            inbrowser=True
        )
        return
    
    # Single command mode
    if args.command:
        if args.command == "list":
            users = db.list_users()
            print_table(users)
            print(f"\nTotal: {len(users)} users")
        
        elif args.command == "search":
            if not args.argument:
                print("Usage: face_admin.py search <name>")
                sys.exit(1)
            users = db.search_users(args.argument)
            print_table(users)
            print(f"\nFound: {len(users)} users")
        
        elif args.command == "info":
            if not args.argument:
                print("Usage: face_admin.py info <name>")
                sys.exit(1)
            user = db.get_user(args.argument)
            if user:
                for key, value in user.items():
                    if key == 'enrolled_at':
                        value = format_timestamp(value)
                    print(f"{key}: {value}")
            else:
                print(f"User '{args.argument}' not found.")
        
        elif args.command == "delete":
            if not args.argument:
                print("Usage: face_admin.py delete <name>")
                sys.exit(1)
            if db.delete_user(args.argument):
                print(f"Deleted '{args.argument}'")
            else:
                print(f"User '{args.argument}' not found.")
        
        elif args.command == "query":
            if not args.argument:
                print("Usage: face_admin.py query \"SELECT ...\"")
                sys.exit(1)
            try:
                results = db.run_query(args.argument)
                print_table(results)
                print(f"\nRows: {len(results)}")
            except Exception as e:
                print(f"Error: {e}")
        
        elif args.command == "export":
            if not args.argument:
                print("Usage: face_admin.py export <filename.csv>")
                sys.exit(1)
            count = db.export_csv(args.argument)
            print(f"Exported {count} users to {args.argument}")
        
        elif args.command == "stats":
            stats = db.get_stats()
            for key, value in stats.items():
                print(f"{key}: {value}")
        
        # Event log commands
        elif args.command == "events":
            if not events_db.exists:
                print("Events database not found.")
                sys.exit(1)
            limit = int(args.argument) if args.argument and args.argument.isdigit() else 50
            events = events_db.get_recent_events(limit=limit)
            print_table(events)
            print(f"\nTotal shown: {len(events)}")
        
        elif args.command == "recognitions":
            if not events_db.exists:
                print("Events database not found.")
                sys.exit(1)
            events = events_db.get_recognitions(name=args.argument if args.argument else None)
            print_table(events)
            print(f"\nTotal shown: {len(events)}")
        
        elif args.command == "event-stats":
            if not events_db.exists:
                print("Events database not found.")
                sys.exit(1)
            stats = events_db.get_stats()
            for key, value in stats.items():
                print(f"{key}: {value}")
        
        elif args.command == "event-export":
            if not events_db.exists:
                print("Events database not found.")
                sys.exit(1)
            if not args.argument:
                print("Usage: face_admin.py event-export <filename.csv>")
                sys.exit(1)
            count = events_db.export_csv(args.argument)
            print(f"Exported {count} events to {args.argument}")
        
        return
    
    # Interactive mode
    if not db.exists:
        sys.exit(1)
    
    interactive_cli(db, events_db if events_db.exists else None)


if __name__ == "__main__":
    main()

