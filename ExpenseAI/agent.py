# Expense AI : Personal Expense Tracking Agent.

import os
import sqlite3
import datetime
import asyncio
import re
from typing import List, Optional, Dict
from pydantic import BaseModel, Field
from dateutil.relativedelta import relativedelta

from google.adk.agents import LlmAgent,SequentialAgent
from google.adk.models.google_llm import Gemini
from google.adk.tools.agent_tool import AgentTool
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

# ============================================================================
# Configuration Setup
# ============================================================================

# Read API key from environment variable
API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("Please set GOOGLE_API_KEY or GEMINI_API_KEY environment variable")

# Model configuration with retry options
retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)

# Response schemas
class AgentResponse(BaseModel):
    """Schema to ensure agents always provide a response."""
    response: str = Field(description="A clear, user-friendly response explaining what was done or what information was found")
    status: str = Field(description="Status of the operation: 'success', 'error', or 'info'")

# ============================================================================
# DatabaseHandler Class
# ============================================================================

class DatabaseHandler:
    """Manages SQLite database operations for expenses."""
    
    def __init__(self, db_name: str = "expenses.db"):
        """Initialize database connection and create table if not exists."""
        self.db_name = db_name
        self._init_db()
    
    def _get_conn(self):
        """Return SQLite connection."""
        return sqlite3.connect(self.db_name)
    
    def _init_db(self):
        """Create expenses table if it doesn't exist."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS expenses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    category TEXT NOT NULL,
                    amount REAL NOT NULL,
                    date TEXT NOT NULL
                )
            ''')
            conn.commit()
    
    def add_record(self, category: str, amount: float, date: str) -> int:
        """Insert a new expense record. Returns expense ID."""
        try:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO expenses (category, amount, date) VALUES (?, ?, ?)",
                    (category, amount, date)
                )
                conn.commit()
                return cursor.lastrowid
        except sqlite3.OperationalError as e:
            raise Exception(f"Database operation failed: {str(e)}")
        except sqlite3.Error as e:
            raise Exception(f"Database error: {str(e)}")
    
    def update_record(self, expense_id: int, category: Optional[str] = None,
                     amount: Optional[float] = None, date: Optional[str] = None) -> bool:
        """Dynamically update expense fields. Returns True if successful."""
        try:
            fields = []
            values = []
            
            if category:
                fields.append("category = ?")
                values.append(category)
            if amount is not None:
                fields.append("amount = ?")
                values.append(amount)
            if date:
                fields.append("date = ?")
                values.append(date)
            
            if not fields:
                return False
            
            values.append(expense_id)
            query = f"UPDATE expenses SET {', '.join(fields)} WHERE id = ?"
            
            with self._get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute(query, tuple(values))
                conn.commit()
                return cursor.rowcount > 0
        except sqlite3.OperationalError as e:
            raise Exception(f"Database operation failed: {str(e)}")
        except sqlite3.Error as e:
            raise Exception(f"Database error: {str(e)}")
    
    def delete_record(self, expense_id: int) -> bool:
        """Delete expense by ID. Returns True if successful."""
        try:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM expenses WHERE id = ?", (expense_id,))
                conn.commit()
                return cursor.rowcount > 0
        except sqlite3.OperationalError as e:
            raise Exception(f"Database operation failed: {str(e)}")
        except sqlite3.Error as e:
            raise Exception(f"Database error: {str(e)}")
    
    def get_expenses(self, start_date: Optional[str] = None, 
                    end_date: Optional[str] = None,
                    category: Optional[str] = None) -> List[Dict]:
        """
        Fetch expenses with optional date range and category filter.
        Returns list of dictionaries with id, category, amount, date.
        """
        with self._get_conn() as conn:
            cursor = conn.cursor()
            
            # Build query with optional filters
            query = "SELECT id, category, amount, date FROM expenses WHERE 1=1"
            params = []
            
            if category:
                query += " AND category = ?"
                params.append(category)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
        
        results = []
        
        # Helper to parse DD-MM-YYYY
        def parse_date(d_str: str):
            try:
                return datetime.datetime.strptime(d_str, "%d-%m-%Y").date()
            except ValueError:
                return None
        
        start_dt = parse_date(start_date) if start_date else None
        end_dt = parse_date(end_date) if end_date else None
        
        for row in rows:
            r_id, r_cat, r_amt, r_date = row
            r_dt = parse_date(r_date)
            
            if not r_dt:
                continue  # Skip invalid dates
            
            # Filter by date range
            if start_dt and r_dt < start_dt:
                continue
            if end_dt and r_dt > end_dt:
                continue
            
            results.append({
                "id": r_id,
                "category": r_cat,
                "amount": r_amt,
                "date": r_date
            })
        
        return results
    
    def find_expense_by_criteria(self, amount: Optional[float] = None,
                                category: Optional[str] = None,
                                date: Optional[str] = None) -> List[Dict]:
        """
        Find expenses matching criteria. Used for identifying expenses before update/delete.
        Returns list of matching expenses.
        """
        all_expenses = self.get_expenses()
        matches = []
        
        for expense in all_expenses:
            match = True
            
            if amount is not None:
                # Allow small floating point differences
                if abs(expense['amount'] - amount) > 0.01:
                    match = False
            
            if category and expense['category'].lower() != category.lower():
                match = False
            
            if date and expense['date'] != date:
                match = False
            
            if match:
                matches.append(expense)
        
        return matches

# Initialize database
db = DatabaseHandler()

# ============================================================================
# Helper Functions
# ============================================================================

def parse_natural_date(date_str: Optional[str]) -> str:
    """
    Convert natural language date to DD-MM-YYYY format using datetime only.
    Handles: today, yesterday, last week, last month, last quarter, past 6 months, 
    this month, X months, and specific years.
    Defaults to today if None/empty.
    """

    today = datetime.date.today()
    
    # 1. Default case (None/empty input)
    if not date_str:
        return today.strftime("%d-%m-%Y")
    
    date_str = date_str.strip().lower()
    
    # --- Part 1: Direct Matches (Existing Logic) ---
    
    if date_str in ["today", "now", "current date"]:
        return today.strftime("%d-%m-%Y")
    
    if date_str == "yesterday":
        yesterday = today - datetime.timedelta(days=1)
        return yesterday.strftime("%d-%m-%Y")

    # --- Part 2: New Logic for Current Period Start ("This X") ---
    
    if "this month" in date_str:
        # Returns the first day of the current month (DD-MM-YYYY)
        start_of_month = today.replace(day=1)
        return start_of_month.strftime("%d-%m-%Y")

    if "this year" in date_str:
        # Returns the first day of the current year (01-01-YYYY)
        start_of_year = today.replace(month=1, day=1)
        return start_of_year.strftime("%d-%m-%Y")

    # --- Part 3: New Logic for Relative Periods ("X periods ago" / "X periods") ---
    
    # Helper for parsing "X units" where X is a number
    num_match = re.search(r'(\d+)\s*(\w+)', date_str)
    
    if num_match:
        number = int(num_match.group(1))
        unit = num_match.group(2)
        
        target_date = None
        
        if "months" in unit:
            # e.g., "3 months" -> 3 months ago
            target_date = today - relativedelta(months=number)
        
        elif "weeks" in unit:
            # e.g., "2 weeks" -> 2 weeks ago
            target_date = today - datetime.timedelta(weeks=number)
            
        elif "years" in unit:
            # e.g., "5 years" -> 5 years ago (accurate for leap years)
            target_date = today - relativedelta(years=number)

        elif "quarters" in unit:
            # e.g., "2 quarters" -> 6 months ago (accurate calendar subtraction)
            target_date = today - relativedelta(months=number * 3)

        if target_date:
            return target_date.strftime("%d-%m-%Y")
            
    # --- Part 4: Specific Year Parsing ("year XXXX") ---
    
    year_match = re.search(r'year\s+(\d{4})', date_str)
    if year_match:
        try:
            year = int(year_match.group(1))
            # Return January 1st of the specified year
            target_date = datetime.date(year, 1, 1)
            return target_date.strftime("%d-%m-%Y")
        except ValueError:
            # Handle invalid year number (e.g., too large)
            pass 
            
    # --- Part 5: Existing Relative Matches (Last X) ---
    
    # X days ago
    days_ago_match = re.search(r'(\d+)\s*days?\s*ago', date_str)
    if days_ago_match:
        days = int(days_ago_match.group(1))
        target_date = today - datetime.timedelta(days=days)
        return target_date.strftime("%d-%m-%Y")
    
    # last week (7 days ago)
    if "last week" in date_str:
        return (today - datetime.timedelta(weeks=1)).strftime("%d-%m-%Y")
    
    # last month (accurate subtraction)
    if "last month" in date_str:
        return (today - relativedelta(months=1)).strftime("%d-%m-%Y")
    
    # last quarter/past quarter (3 months ago)
    if "last quarter" in date_str or "past quarter" in date_str:
        return (today - relativedelta(months=3)).strftime("%d-%m-%Y")
    
    # past 6 months/last 6 months (6 months ago)
    if "past 6 months" in date_str or "last 6 months" in date_str:
        return (today - relativedelta(months=6)).strftime("%d-%m-%Y")
    
    # last year/past year (1 year ago)
    if "last year" in date_str or "past year" in date_str:
        return (today - relativedelta(years=1)).strftime("%d-%m-%Y")
    
    # --- Part 6: Explicit Date Format Parsing (Existing Logic) ---
    
    formats = ["%d-%m-%Y", "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"]
    for fmt in formats:
        try:
            parsed = datetime.datetime.strptime(date_str, fmt).date()
            return parsed.strftime("%d-%m-%Y")
        except ValueError:
            continue
    
    # 7. Final fallback
    return today.strftime("%d-%m-%Y")

def normalize_category(category: str) -> str:
    """
    Map vague category inputs to strict enum.
    Categories: Snacks, Food, Travel, Groceries, Fashion, Healthcare, Entertainment, Utilities, Rent, Wasted
    """
    if not category:
        return "Wasted"
    
    category_lower = category.strip().lower()
    print(category_lower) #2dl
    # Strict enum list
    valid_categories = [
        "Snacks", "Food", "Travel", "Groceries", "Fashion",
        "Healthcare", "Entertainment", "Utilities", "Rent", "Wasted"
    ]
    
    # Exact match (case-insensitive)
    for valid_cat in valid_categories:
        if category_lower == valid_cat.lower():
            return valid_cat
    
    # # Fuzzy matching
    category_mapping = {
        #Snacks category
        "snack": "Snacks",
        "snacks": "Snacks",
        "juice": "Snacks",
        "chocolate": "Snacks",
        "chips": "Snacks",
        "junk": "Snacks",
        #Food category
        "food": "Food",
        "meal": "Food",
        "meals": "Food",
        "lunch": "Food",
        "dinner": "Food",
        "breakfast": "Food",
        #Travel category
        "travel": "Travel",
        "fuel": "Travel",
        "petrol": "Travel",
        "deisel": "Travel",
        "bus": "Travel",
        "auto": "Travel",
        "rapido": "Travel",
        "uber": "Travel",
        "cab": "Travel",
        "taxi": "Travel",
        "transport": "Travel",
        "transportation": "Travel",
        #Groceries category
        "grocery": "Groceries",
        "groceries": "Groceries",
        "shopping": "Groceries",
        #Fashion category
        "fashion": "Fashion",
        "clothes": "Fashion",
        "clothing": "Fashion",
        "jacket": "Fashion",
        "shirt": "Fashion",
        "pants": "Fashion",
        "shorts": "Fashion",
        "cap": "Fashion",
        "watch": "Fashion",
        #Healthcare category
        "health": "Healthcare",
        "healthcare": "Healthcare",
        "medical": "Healthcare",
        "medicine": "Healthcare",
        "meds": "Healthcare",
        "hospital": "Healthcare",
        "clinic": "Healthcare",
        "dentist": "Healthcare",
        #Entertainment category
        "entertainment": "Entertainment",
        "movie": "Entertainment",
        "movies": "Entertainment",
        "amusement": "Entertainment",
        "entertainments": "Entertainment",
        #Utilities category
        "utility": "Utilities",
        "utilities": "Utilities",
        "electricity": "Utilities",
        "water": "Utilities",
        "gas": "Utilities",
        "lpg": "Utilities",
        "internet": "Utilities",
        #Rent category
        "rent": "Rent",
        #Wasted category
        "wasted": "Wasted",
        "waste": "Wasted",
        "fine": "Wasted",
        "fines": "Wasted",
        "lost": "Wasted"
    }
    
    # Check for partial matches
    for key, value in category_mapping.items():
        if key in category_lower:
            return value
    
    # Default to Wasted if no match
    return "spending un-categorizable"

def format_currency(amount: float) -> str:
    """Format amount with Indian Rupee symbol."""
    return f"₹{amount:.2f}"

# ============================================================================
# Tool Definitions
# ============================================================================

def add_expense(category: str, amount: float, date: str) -> str:
    """
    Adds a new expense to the database.
    Args:
        category: Category of expense (will be normalized).
        amount: Numeric amount spent.
        date: Date in DD-MM-YYYY format (will be parsed if natural language).
    Returns:
        Success message with expense ID.
    """
    try:
        # Validate amount
        if not isinstance(amount, (int, float)):
            return "Error: Amount must be a number."
        if amount < 0:
            return "Error: Amount cannot be negative."
        if amount == 0:
            return "Error: Amount must be greater than zero."
        if amount != amount:  # Check for NaN
            return "Error: Invalid amount provided (NaN)."
        if amount > 1e10:  # Prevent extremely large values
            return "Error: Amount is too large."
        
        # Validate category
        if not category or not category.strip():
            return "Error: Category cannot be empty."
        
        # Normalize category and parse date
        normalized_category = normalize_category(category)
        parsed_date = parse_natural_date(date)
        
        expense_id = db.add_record(normalized_category, amount, parsed_date)
        return f"Success: Expense added with ID {expense_id}. Spent {format_currency(amount)} for {normalized_category} on {parsed_date}."
    except sqlite3.Error as e:
        return f"Database error: {str(e)}"
    except Exception as e:
        return f"Error adding expense: {str(e)}"

def update_expense(expense_id: int, category: Optional[str] = None,
                  amount: Optional[float] = None, date: Optional[str] = None) -> str:
    """
    Updates an existing expense record.
    Args:
        expense_id: ID of expense to update.
        category: New category (optional, will be normalized).
        amount: New amount (optional).
        date: New date (optional, will be parsed).
    Returns:
        Success or error message.
    """
    try:
        # Validate expense_id
        if not isinstance(expense_id, int) or expense_id <= 0:
            return f"Error: Invalid expense ID {expense_id}. ID must be a positive integer."
        
        # Validate amount if provided
        if amount is not None:
            if not isinstance(amount, (int, float)):
                return "Error: Amount must be a number."
            if amount < 0:
                return "Error: Amount cannot be negative."
            if amount != amount:  # NaN check
                return "Error: Invalid amount provided (NaN)."
            if amount > 1e10:
                return "Error: Amount is too large."
        
        # Validate category if provided
        if category is not None and (not category or not category.strip()):
            return "Error: Category cannot be empty."
        
        # Check if at least one field to update
        if not any([category, amount is not None, date]):
            return "Error: At least one field (category, amount, or date) must be provided for update."
        
        # Normalize category and parse date if provided
        normalized_category = normalize_category(category) if category else None
        parsed_date = parse_natural_date(date) if date else None
        
        success = db.update_record(expense_id, normalized_category, amount, parsed_date)
        if success:
            return f"Success: Expense ID {expense_id} updated."
        else:
            return f"Error: Expense ID {expense_id} not found or no changes provided."
    except sqlite3.Error as e:
        return f"Database error: {str(e)}"
    except Exception as e:
        return f"Error updating expense: {str(e)}"

def delete_expense(expense_id: int) -> str:
    """
    Deletes an expense from the database.
    Args:
        expense_id: ID of expense to delete.
    Returns:
        Success or error message.
    """
    try:
        # Validate expense_id
        if not isinstance(expense_id, int) or expense_id <= 0:
            return f"Error: Invalid expense ID {expense_id}. ID must be a positive integer."
        
        success = db.delete_record(expense_id)
        if success:
            return f"Success: Expense ID {expense_id} deleted."
        else:
            return f"Error: Expense ID {expense_id} not found."
    except sqlite3.Error as e:
        return find_expense_by_criteria
        f"Database error: {str(e)}"
    except Exception as e:
        return f"Error deleting expense: {str(e)}"

def find_expenses(amount: Optional[float] = None,
                 category: Optional[str] = None,
                 date: Optional[str] = None) -> str:
    """
    Find expenses matching criteria and return details with IDs.
    Used by orchestrator to identify expenses before update/delete operations.
    
    Args:
        amount: Optional amount to match.
        category: Optional category to match (will be normalized).
        date: Optional date to match (DD-MM-YYYY format, will be parsed).
    
    Returns:
        Formatted string with matching expenses including IDs.
    """
    try:
        # Validate amount if provided
        if amount is not None:
            if not isinstance(amount, (int, float)):
                return "Error: Amount must be a number."
            if amount < 0:
                return "Error: Amount cannot be negative."
            if amount != amount:  # NaN check
                return "Error: Invalid amount provided (NaN)."
        
        normalized_category = normalize_category(category) if category else None
        parsed_date = parse_natural_date(date) if date else None
        
        matches = db.find_expense_by_criteria(amount, normalized_category, parsed_date)
        
        if not matches:
            criteria = []
            if amount is not None: criteria.append(f"amount={format_currency(amount)}")
            if category: criteria.append(f"category={normalized_category}")
            if date: criteria.append(f"date={parsed_date}")
            return f"No expenses found matching: {', '.join(criteria) if criteria else 'any criteria'}."
        
        # Format results with IDs
        result_lines = []
        for expense in matches:
            result_lines.append(
                f"ID: {expense['id']}, Category: {expense['category']}, "
                f"Amount: {format_currency(expense['amount'])}, Date: {expense['date']}"
            )
        
        return f"Found {len(matches)} matching expense(s):\n" + "\n".join(result_lines)
    
    except sqlite3.Error as e:
        return f"Database error: {str(e)}"
    except Exception as e:
        return f"Error finding expenses: {str(e)}"

def get_expenses_summary(start_date: str, end_date: str, category: Optional[str] = None) -> str:
    """
    Retrieves and summarizes expenses within a date range.
    Args:
        start_date: Start date string (DD-MM-YYYY format, will be parsed if natural language).
        end_date: End date string (DD-MM-YYYY format, will be parsed if natural language).
        category: Optional category filter (will be normalized).
    Returns:
        Formatted summary with total, category breakdown, and record count.
    """
    try:
        # Parse dates and normalize category
        parsed_start = parse_natural_date(start_date)
        parsed_end = parse_natural_date(end_date)
        
        # Validate date range
        try:
            start_dt = datetime.datetime.strptime(parsed_start, "%d-%m-%Y").date()
            end_dt = datetime.datetime.strptime(parsed_end, "%d-%m-%Y").date()
            
            if start_dt > end_dt:
                return f"Error: Start date ({parsed_start}) cannot be after end date ({parsed_end})."
        except ValueError:
            # If date parsing fails, continue anyway (parse_natural_date should handle it)
            pass
        
        normalized_category = normalize_category(category) if category else None
        
        records = db.get_expenses(parsed_start, parsed_end, normalized_category)
        
        if not records:
            return f"No expenses found between {parsed_start} and {parsed_end}" + \
                   (f" for category {normalized_category}." if normalized_category else ".")
        
        # Calculate summary
        total = sum(r['amount'] for r in records)
        breakdown = {}
        for r in records:
            cat = r['category']
            breakdown[cat] = breakdown.get(cat, 0) + r['amount']
        
        # Format breakdown with currency
        breakdown_str = ", ".join([f"{k}: {format_currency(v)}" for k, v in breakdown.items()])
        
        summary = f"Found {len(records)} record(s). Total: {format_currency(total)}.\n" \
                 f"Category Breakdown: {breakdown_str}\n" \
                 f"Date Range: {parsed_start} to {parsed_end}"
        
        return summary
    except sqlite3.Error as e:
        return f"Database error: {str(e)}"
    except Exception as e:
        return f"Error getting summary: {str(e)}"

# ============================================================================
# Sub-Agents
# ============================================================================

expense_manager = LlmAgent(
    name="expense_manager",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    description="Manages expense records: adding, updating, and deleting expenses.",
    instruction="""You are an expense management assistant. Your primary role is to add, update, and delete expense records in the database.

Responsibilities:
1. **Find Expenses**: Use the `find_expenses` tool to search for expenses by amount, category, or date. This is essential before update/delete operations.
2. **Add Expenses**: Use the `add_expense` tool to record new expenses. Ensure you have the category, amount, and date. If the date is not provided, use the current date.
3. **Update Expenses**: Use the `update_expense` tool to modify existing expense records. You need the expense_id to update. Use `find_expenses` first if you don't have the ID.
4. **Delete Expenses**: Use the `delete_expense` tool to remove expense records. You need the expense_id to delete. Use `find_expenses` first if you don't have the ID.

Rules:
1. When adding an expense, if the 'date' is not explicitly mentioned, assume the current date.
2. For 'update_expense' and 'delete_expense' operations, you must have the expense_id. If you don't have it, use `find_expenses` to search for the expense first.
3. If multiple expenses match your search criteria, inform the user and ask which one they want to modify.
4. If 'category' or 'amount' is missing for any operation, ask the user for the missing information.
5. Always use Indian Rupee (₹) when displaying amounts.
6. Always provide a clear, user-friendly response in the 'response' field explaining what was done.

Your Tools:
- `find_expenses`: To search for expenses by criteria (amount, category, date) and get their IDs.
- `add_expense`: To add a new expense record.
- `update_expense`: To modify an existing expense record.
- `delete_expense`: To remove an expense record.
""",
    tools=[find_expenses, add_expense, update_expense, delete_expense],
    output_key= "ManagerAgentResponse"
)

expense_summarizer = LlmAgent(
    name="expense_summarizer",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    description="Summarizes expenses based on date ranges and categories.",
    instruction="""Use the `get_expenses_summary` tool to retrieve and summarize expense information.

Rules:
1. ALWAYS convert natural language dates (e.g., "today", "last month", "yesterday", "3 days ago", "last 2 quarters", "last year", etc.) into strictly 'DD-MM-YYYY' format before calling the tool.
2. If only one date is provided, determine the appropriate date range (e.g., if "last month" is mentioned, calculate start and end dates for that month).
3. Provide a clear summary of the expenses found, including total amount in Indian Rupees (₹) and category breakdown.
4. If the user asks for expenses in specific categories, pass the category parameter to the tool.
5. Always provide your summary in the 'response' field in a conversational, easy-to-understand format.

Your Tools:
- `get_expenses_summary`: To retrieve and summarize expenses within a date range, optionally filtered by category.
""",
    tools=[get_expenses_summary],
    output_key = "SummarizeAgentResponse"
)

# ============================================================================
# Orchestrator Agent
# ============================================================================

today = datetime.date.today().strftime("%d-%m-%Y")

orchestrator_instructions = f"""
You are a helpful Financial Assistant (Orchestrator) for expense tracking. You can track expenses and provide custom summaries based on user input.

Your responsibilities:
1. Analyze user intent: Determine if the user wants to Add, Update, Delete, or Summarize expenses.
2. **CRITICAL - Parameter Validation**: 
   - For Add/Update operations: Check if Amount and Category are provided. If missing, ASK the user for the missing information.
   - For Update/Delete operations: You must first identify the specific expense using date/amount/category before calling the modification tool. If you cannot identify the expense uniquely, ask the user for clarification.
3. **Date Handling**: 
   - ALWAYS transform natural language dates (e.g., "today", "last month", "yesterday", "3 days ago", "last 2 quarters", "last year", etc.) into strictly 'DD-MM-YYYY' format before passing to any tool.
   - If date is not provided, default to today: {today}
4. **Currency**: Always use Indian Rupee (₹) in all outputs.
5. **Category Mapping**: Map user categories to the strict enum: [Snacks, Food, Travel, Groceries, Fashion, Healthcare, Entertainment, Utilities, Rent, Wasted]
6. **Delegation**:
   - For Add/Update/Delete: Use the 'expense_manager' agent tool.
   - For Summarize: Use the 'expense_summarizer' agent tool.
7. **Clarification**: If a request is unclear or ambiguous, ask clarifying questions. Do not assume intentions.
8. **CRITICAL - Always respond to the user**: After delegating to sub-agents or completing any operation, you MUST provide a clear, user-friendly response. Never leave the user without a response. Format your response in a conversational, helpful manner that explains what was done or what information was found.

Rules for expense_manager agent:
a. Amount and Category are REQUIRED for Add operations. If missing, ask the user.
b. For Update/Delete: You must identify the expense first. Use date, amount, and/or category to find the expense_id. If multiple matches, ask the user to specify which one.
c. If date is missing, use current date: {today}

Rules for expense_summarizer agent:
a. ALWAYS transform natural language dates to 'DD-MM-YYYY' format before calling the tool.
b. Calculate appropriate date ranges (start_date, end_date) based on user input.
c. Handle flexible inputs like "Food and Snacks" by making multiple queries or filtering results.

Output Format:
1. For Adding: Always respond with: "Spent ₹X for [Category] on [Date]." or a similar friendly message.
2. For Updating: Always respond with: "Updated expense ID [id]: Changed [old_value] to [new_value]." or a similar friendly message.
3. For Deleting: Always respond with: "Deleted expense: ₹X for [Category] on [Date]." or a similar friendly message.
4. For Summarizing: Always provide a clear, conversational summary with totals in ₹ and category breakdown. Make it easy to understand.
5. **Never leave the user without a response** - always provide feedback after any operation.

Current Date: {today}
"""

orchestrator_agent = LlmAgent(
    name="orchestrator_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    description="Orchestrates expense tracking operations by delegating to specialized agents.",
    instruction=orchestrator_instructions,
    tools=[AgentTool(expense_manager), AgentTool(expense_summarizer)],
    output_key="agent_response"
)

root_agent = orchestrator_agent