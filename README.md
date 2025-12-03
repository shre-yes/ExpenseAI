# Expense Tracking Agent

This project is a Python-based Expense Tracking Agent that allows users to manage their personal finances through a conversational interface. It leverages the Google ADK and Gemini models for natural language processing and agent orchestration.

## Features

*   **Add Expenses**: Record new expenses with category, amount, and date. Supports natural language dates and category normalization.
*   **Update Expenses**: Modify existing expense records using their unique ID.
*   **Delete Expenses**: Remove expense records from the database.
*   **Find Expenses**: Search for expenses based on criteria like date, category, or amount.
*   **Summarize Expenses**: Generate reports on spending within specified date ranges and by category, with totals in Indian Rupees (₹).

## Technologies Used

*   Python
*   Google ADK
*   Gemini Models
*   SQLite
*   Pydantic

## Setup and Installation

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/shre-yes/ExpenseAI
    cd ExpenseTrackingAgent
    ```

2.  **Set up a Virtual Environment**:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
    ```

3.  **Install Dependencies**:
    Create a `requirements.txt` file in the root directory with the following content:
    ```
    google-adk
    pydantic
    python-dateutil
    ```
    Then, install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure API Key**:
    Create a `.env` file in the `ExpenseAI/` directory with the following content, replacing `YOUR_API_KEY` with your actual Google API key:
    ```dotenv
    GOOGLE_GENAI_USE_VERTEXAI=FALSE
    GOOGLE_API_KEY=PASTE_YOUR_ACTUAL_API_KEY_HERE
    ```
    The agent will automatically load this key.

## Usage

**For Terminal Interaction**:
Run the agent using the ADK command:
```bash
adk run ExpenseAI
```

**For Web UI Usage**:
Run the ADK web server within your activated virtual environment:
```bash
adk web
```
This will be accessible at http://127.0.0.1:8000.

## Deployment

This project is a Python script. To deploy it on a server, you would typically:
1.  Set up a Python environment.
2.  Install dependencies.
3.  Configure the API key.
4.  Run the script, potentially using a process manager like `systemd` or `supervisor` for continuous operation.

## Database Management

The `expenses.db` SQLite database will be automatically generated and maintained by the `DatabaseHandler` class when the agent is first run. There is no need to create or manage this file separately.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.