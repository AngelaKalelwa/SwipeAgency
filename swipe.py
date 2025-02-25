import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser 
from langchain_community.utilities import SQLDatabase

load_dotenv()  # Load environment variables

def init_database(user: str, password: str, host: str, database: str, port: int = 1521) -> SQLDatabase:
    """
    Initializes a connection to an Oracle database using cx_Oracle.
    
    Returns:
    - SQLDatabase: A LangChain SQLDatabase instance
    """
    user = os.getenv("DB_USER", user)
    password = os.getenv("DB_PASSWORD", password)
    host = os.getenv("DB_HOST", host)
    port = os.getenv("DB_PORT", port)
    service_name = os.getenv("DB_SERVICE", "orcl")
    
    db_uri = f"oracle+cx_oracle://{user}:{password}@{host}:{port}/?service_name={service_name}"
    return SQLDatabase.from_uri(db_uri)


def get_sql_chain(db):
    """Generates an SQL query based on user input and database schema."""
    template = """You are a data analyst at a company. You are interacting with a user who is asking questions about the company's database.
    Based on the table schema below, write an optimized Oracle SQL query that would answer the user's question. Take the conversation history into account.

    <SCHEMA>{schema}</SCHEMA>

    Status Codes:
    NEW_STATUS = 0
    ACTIVE_STATUS = 1
    EDITED_STATUS = 2
    DEACTIVATION_STATUS = 3
    INACTIVE_STATUS = 4
    SOFT_DELETE_STATUS = 5
    LOCKED_STATUS = 6
    
    Conversation History: {chat_history}

    Guidelines:
    1. Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, comments, or backticks.
    2. Use appropriate table names and column references as per the schema.
    3. Optimize the query for performance where possible.
    4. Handle ambiguous queries by asking for clarification.
    5. Ensure full compatibility with **Oracle SQL** by following these rules:
    - Use **Oracle-specific syntax and functions**:
        - For date differences: 
        - If the columns are of type `DATE`, use:
            `(DATE2 - DATE1) * 24 * 60 * 60` to get the difference in seconds.
        - If the columns are of type `TIMESTAMP`, use `EXTRACT()` to break down the `INTERVAL` into days, hours, minutes, and seconds, then convert everything to seconds:
            ```
            EXTRACT(SECOND FROM (TIMESTAMP2 - TIMESTAMP1)) 
            + EXTRACT(MINUTE FROM (TIMESTAMP2 - TIMESTAMP1)) * 60
            + EXTRACT(HOUR FROM (TIMESTAMP2 - TIMESTAMP1)) * 3600
            + EXTRACT(DAY FROM (TIMESTAMP2 - TIMESTAMP1)) * 86400
            ```
        - Use `TO_DATE()` and `TO_CHAR()` for date conversions and formatting.
        - Use `NVL()` for null handling instead of `COALESCE()`.
        - Use `TRUNC()` for date truncation.
    - **Avoid non-Oracle functions** like `EXTRACT(EPOCH FROM ...)`.
    - Do **not** use semicolons (`;`) at the end of the query when generating queries for use in Python (e.g., with SQLAlchemy).
    - When using aliases, avoid using the `AS` keyword for column aliases (optional in Oracle).
    - If conditional logic is needed, use `CASE WHEN ... THEN ... END`.
    6. Make sure all column names and table names are exactly as they appear in the schema (case-sensitive if needed).
    7. Always sanitize inputs to prevent SQL injection.

    For example:
    Question: How many transactions are available?
    SQL Query: SELECT COUNT(*) AS total_transactions FROM SWIPE_TRANSACTIONS

    Question: What is the frequently used transaction?
    SQL Query: SELECT TRANSACTION_TYPE, COUNT(*) AS transaction_count
                FROM SWIPE_TRANSACTIONS
                GROUP BY TRANSACTION_TYPE
                ORDER BY transaction_count DESC

    Question: How many terminals are active?
    SQL Query: SELECT COUNT(*) AS active_terminals
                FROM SWIPE_TERMINALS
                WHERE STATUS = 1

    Question: What is the average time per transaction?
    SQL Query: SELECT AVG(
                    EXTRACT(SECOND FROM (TRANS_TIME - DATE_CREATED)) 
                    + EXTRACT(MINUTE FROM (TRANS_TIME - DATE_CREATED)) * 60
                    + EXTRACT(HOUR FROM (TRANS_TIME - DATE_CREATED)) * 3600
                    + EXTRACT(DAY FROM (TRANS_TIME - DATE_CREATED)) * 86400
                ) AS average_time_per_transaction 
                FROM SWIPE_TRANSACTIONS

    Your turn:
    Question: {question}
    SQL Query:
    """

    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatOpenAI(model="gpt-4o")

    def get_schema(_):
        return db.get_table_info()

    return (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm
        | StrOutputParser()
    )


def generate_report_template():
    """
    Generates a prompt template that instructs the AI model to create a report.
    """
    template = """You are a data analyst assistant. Based on the conversation history and database queries, generate a detailed report summarizing the following:
    
    1. Key findings based on SQL queries executed.
    2. Trends observed in the data.
    3. Any insights that could help the business or technical team make decisions.

    Report Format:
    1. **Introduction**: A brief overview of the analysis.
    2. **Key Findings**: Summarize key findings from the data analysis.
    3. **Trends/Patterns**: Mention any trends or patterns identified in the data.
    4. **Recommendations**: Provide any actionable recommendations for the business or technical team.

    Conversation History: {chat_history}
    """
    prompt = ChatPromptTemplate.from_template(template)
    return prompt

def get_response(user_query: str, db: SQLDatabase, chat_history: list):
    """Processes user query, generates SQL, executes it, and returns response, including report generation"""
    # Check if the user asks for a report
    if "generate report" in user_query.lower():
        # Create the report generation chain
        report_prompt = generate_report_template()
        llm = ChatOpenAI(model="gpt-4o")

        # Run the report generation
        chain = (
            RunnablePassthrough.assign(
                chat_history=chat_history
            )
            | report_prompt
            | llm
            | StrOutputParser()
        )
        return chain.invoke({
            "chat_history": chat_history
        })

    else:
        # Regular SQL generation and response
        sql_chain = get_sql_chain(db)

        template = """You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
        Based on the table schema below, question, SQL query, and SQL response, write a clear natural language response. If the query fails, explain the issue.

        <SCHEMA>{schema}</SCHEMA>
        
        Conversation History: {chat_history}
        SQL Query: <SQL>{query}</SQL>
        User Question: {question}
        SQL Response: {response}
        """
     
    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatOpenAI(model="gpt-4o")

    chain = (
        RunnablePassthrough.assign(query=sql_chain).assign(
            schema=lambda _: db.get_table_info(),
            response=lambda vars: db.run(vars["query"]),
        )
        | prompt
        | llm
        | StrOutputParser()
    )   
    return chain.invoke({
        "question": user_query,
        "chat_history": chat_history,
    })


def execute_query(query, db):
    """Execute the SQL query and handle potential errors."""
    try:
        return db.run(query)
    except Exception as e:
        return f"Query failed: {str(e)}"

st.set_page_config(page_title="Chat with Oracle", page_icon="💬")
st.title("Chat with Oracle")

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello! I'm an Oracle SQL Database assistant. Ask me anything about your database.")
    ]
 # Check if database connection already exists in session state
if "db" not in st.session_state:
    st.session_state.db = None   

# Sidebar for connection settings
#with st.sidebar:
    #st.subheader("Settings")
    #st.write("This is a simple chat application. Connect to the database and start chatting.")

    #st.text_input("Host", value=os.getenv("DB_HOST", "localhost"),key='Host')
    #st.text_input("Port", value=os.getenv("DB_PORT", "1521"), key="Port")
    #st.text_input("User", value=os.getenv("DB_USER", "root"), key="User")
    #st.text_input("Password", type="password", value=os.getenv("DB_PASSWORD", ""), key="Password")
    #st.text_input("Service_Name", value=os.getenv("DB_SERVICE", "orcl"), key="Service_Name")

    #if st.button("Connect")and not st.session_state.db:
       # try:
            #with st.spinner("Connecting to the Database...."):
                #db = init_database(
                    #st.session_state["User"],
                    #st.session_state["Password"],
                    #st.session_state["Host"],
                    #st.session_state["Port"],
                    #st.session_state["Service_Name"]
        #)
            #st.session_state["db"] = db
            #st.success("Connected to database")
        #except Exception as e:
            #st.error(f"Connection failed: {str(e)}")

    #if st.session_state.db:
        #st.success("Already connected to the database")

import streamlit as st

# Sidebar for connection settings
with st.sidebar:
    st.subheader("Settings")
    st.write("This is a simple chat application. Connect to the database and start chatting.")

    # Remove all default values for security
    host = st.text_input("Host", key='Host')  # No default value
    port = st.text_input("Port", key="Port")  # No default value
    user = st.text_input("User", key="User")  # No default value
    password = st.text_input("Password", type="password", key="Password")  # No default value
    service_name = st.text_input("Service Name", key="Service_Name")  # No default value

    # Ensure all fields are filled before connecting
    if st.button("Connect") and not st.session_state.get("db"):
        if all([host, port, user, password, service_name]):
            try:
                with st.spinner("Connecting to the Database...."):
                    db = init_database(
                        user,
                        password,
                        host,
                        port,
                        service_name
                    )
                st.session_state["db"] = db
                st.success("Connected to database")
            except Exception as e:
                st.error(f"Connection failed: {str(e)}")
        else:
            st.warning("Please fill in all fields before connecting.")

    if st.session_state.get("db"):
        st.success("Already connected to the database")


# Display chat history
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)  

# Chat input
user_query = st.chat_input("Type a message...")
if user_query is not None and user_query.strip() != "" :
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    try:
        with st.chat_message("AI"):
            response = get_response(user_query, st.session_state.db, st.session_state.chat_history)
            st.markdown(response)
            st.session_state.chat_history.append(AIMessage(content=response))
    except Exception as e:
        st.error(f"Failed to get a response:{str(e)}")