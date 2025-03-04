
# Archon: Supabase and Pydantic AI Agent Implementation Documentation

## 1. Project Architecture Overview

### 1.1 Core Components

- **Agent System**: Built on LangGraph and Pydantic AI, providing a conversational workflow
- **Document Retrieval**: Embedding-based semantic search using Supabase vector database
- **UI**: Streamlit interface with agent selection and conversation management
- **Documentation Crawlers**: Separate crawlers for different documentation sources

### 1.2 Directory Structure

```
Archon/
├── archon/
│   ├── archon_graph.py        # Core agent workflow and orchestration
│   ├── supabase_coder.py      # Supabase agent implementation
│   ├── pydantic_ai_coder.py   # Pydantic AI agent implementation
│   ├── crawl_supabase_docs.py # Crawler for Supabase documentation
│   └── crawl_pydantic_ai_docs.py # Crawler for Pydantic AI documentation
├── streamlit_ui.py            # Main UI implementation
└── utils/
    └── utils.py               # Shared utility functions
```

## 2. Agent System Implementation

### 2.1 Agent Configuration

The system supports multiple agent types (currently Supabase and Pydantic AI) with the following key files:

- **archon_graph.py**: Defines the agent state management and workflow graph
  - `AgentState`: TypedDict with fields including `agent_type`, `latest_user_message`, `messages`
  - `define_scope_with_reasoner()`: Central function that differentiates between agent types
  - `coder_agent()`: Executes the appropriate agent based on agent_type in state
  - `run_agent_with_streaming()`: Entry point for agent invocation from the UI

### 2.2 Agent Type Handling

Agent selection and state is managed in several places:

```python
# In archon_graph.py
async def define_scope_with_reasoner(state: AgentState):
    # Get the agent type from the state
    agent_type = state.get('agent_type', 'Pydantic AI Agent')
    
    # Set source filter based on agent type
    if agent_type == "Pydantic AI Agent":
        documentation_pages = await list_documentation_pages_helper(supabase)
        source_filter = "pydantic_ai_docs"
    else:  # Supabase Agent
        from archon.supabase_coder import list_documentation_pages_helper as supabase_list_docs
        documentation_pages = await supabase_list_docs(supabase)
        source_filter = "supabase_docs"
```

### 2.3 Agent Implementation Details

Both agent implementations follow a similar pattern:

1. **Supabase Agent (supabase_coder.py)**:
   - Created with `Agent()` constructor from Pydantic AI
   - Uses `match_site_pages` RPC with filter `{"source": "supabase_docs"}`
   - Exposes tools like `retrieve_relevant_documentation()` and `list_documentation_pages()`

2. **Pydantic AI Agent (pydantic_ai_coder.py)**:
   - Also built with Pydantic AI's `Agent()`
   - Uses same database but with filter `{"source": "pydantic_ai_docs"}`
   - Similar tool interfaces for consistency

## 3. Documentation Database and Retrieval

### 3.1 Database Structure

The application uses Supabase with a `site_pages` table that includes:
- `url`: URL of the documentation page
- `title`: Page title
- `content`: Page content
- `summary`: Summary of the content
- `embedding`: Vector embedding for semantic search
- `metadata`: JSON object with source information (`pydantic_ai_docs` or `supabase_docs`)

### 3.2 Retrieval Implementation

Document retrieval is performed via an RPC function `match_site_pages` with filters:

```python
# For Supabase agent (in supabase_coder.py)
result = ctx.deps.supabase.rpc(
    "match_site_pages",
    {
        "query_embedding": query_embedding,
        "match_count": 5,
        "filter": {"source": "supabase_docs"}
    }
).execute()

# For Pydantic agent (in pydantic_ai_coder.py)
result = ctx.deps.supabase.rpc(
    'match_site_pages',
    {
        'query_embedding': query_embedding,
        'match_count': 4,
        'filter': {'source': 'pydantic_ai_docs'}
    }
).execute()
```

### 3.3 Embedding Generation

Both agents use a shared approach for embedding generation:

```python
async def get_embedding(text: str, openai_client: AsyncOpenAI) -> List[float]:
    """Generate an embedding for the given text using OpenAI's API."""
    # Truncate text if too long (OpenAI's embedding model has a token limit)
    if len(text) > 8000:
        text = text[:8000]
        
    # Call the OpenAI API to generate the embedding
    response = await openai_client.embeddings.create(
        model=embedding_model,
        input=text
    )
    
    # Return the embedding as a list of floats
    return response.data[0].embedding
```

## 4. Documentation Crawling Process

### 4.1 Crawler Implementation

Two separate crawlers with similar implementations:
- `crawl_supabase_docs.py`: Crawls Supabase documentation
- `crawl_pydantic_ai_docs.py`: Crawls Pydantic AI documentation

The crawlers:
1. Fetch documentation pages
2. Process and chunk content
3. Generate embeddings
4. Store in Supabase with appropriate metadata

### 4.2 Crawler Integration with UI

The UI has a dedicated documentation tab that allows:
- Viewing current documentation status
- Triggering re-crawling of documentation
- Clearing existing records

```python
# In streamlit_ui.py (documentation_tab function)
from archon.crawl_supabase_docs import start_crawl_with_requests, clear_existing_records
from archon.crawl_pydantic_ai_docs import start_crawl_with_requests as start_pydantic_crawl
```

## 5. UI Implementation

### 5.1 Agent Selection

The UI implements agent selection in the `chat_tab()` function:

```python
# Agent selection dropdown
agent_options = ["", "Pydantic AI Agent", "Supabase Agent"]
selected_agent = st.selectbox(
    "Select Agent Type",
    options=agent_options,
    index=0,
    format_func=lambda x: "Select an agent..." if x == "" else x,
    key="agent_type_selector"
)

# Store selected agent in session state
if "selected_agent" not in st.session_state or st.session_state.selected_agent != selected_agent:
    st.session_state.selected_agent = selected_agent
    # Clear previous messages when switching agents
    st.session_state.messages = []
```

### 5.2 Chat Input Management

The chat input is conditionally enabled based on agent selection:

```python
# Initialize user_input to None
user_input = None

# Show chat input only if an agent is selected
if st.session_state.selected_agent:
    user_input = st.chat_input(f"Ask me what you want to build with {st.session_state.selected_agent}...")
else:
    st.chat_input("Please select an agent type first...", disabled=True)
```

### 5.3 Agent Invocation

The agent is invoked through `run_agent_with_streaming()`:

```python
async def run_agent_with_streaming(user_input: str, agent_type: str = ""):
    # Validate agent type
    if not agent_type:
        yield "Please select an agent type first."
        return
        
    # Enhanced logging for agent initialization
    print(f"\n====================================================")
    print(f"[AGENT INITIALIZATION] Starting agent with type: {agent_type}")
    print(f"[AGENT INITIALIZATION] User input: {user_input[:50]}...")
    print(f"====================================================\n")
    
    config = {
        "configurable": {
            "thread_id": thread_id,
            "agent_type": agent_type  # Pass agent type in config
        }
    }
```

## 6. Logging Implementation

### 6.1 Enhanced Logging

The application implements detailed logging to trace agent operation:

```python
# In streamlit_ui.py
import logging
# Set up HTTP request logging to show database queries
logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.INFO)
```

### 6.2 Agent-Specific Logging

Both agents implement detailed operational logging:

```python
# In supabase_coder.py
print(f"[SUPABASE QUERY] Searching for documentation with filter: {{\"source\": \"supabase_docs\"}}")
print(f"[SUPABASE RESULT] Found {len(result.data if result.data else [])} matching documents")

# In pydantic_ai_coder.py
print(f"[PYDANTIC QUERY] Getting document URLs with filter: metadata->>source=pydantic_ai_docs")
print(f"[PYDANTIC RESULT] Found {len(urls)} unique documentation URLs")
```

## 7. Adding a New Agent Type

To add a new agent type (e.g., "Example Agent"), follow these steps:

### 7.1 Create Agent Implementation

1. Create a new file `archon/example_coder.py` following the pattern of existing agents:
   ```python
   from pydantic_ai import Agent, RunContext
   from dataclasses import dataclass
   from supabase import Client
   from openai import AsyncOpenAI
   
   @dataclass
   class ExampleDeps:
       supabase: Client
       openai_client: AsyncOpenAI
       reasoner_output: str
   
   # Initialize the agent
   example_coder = Agent[ExampleDeps]()
   
   # Define tools and functions
   @example_coder.tool
   async def retrieve_relevant_documentation(ctx: RunContext[ExampleDeps], user_query: str) -> str:
       # Implementation similar to other agents but with "example_docs" filter
   ```

2. Implement a crawler `archon/crawl_example_docs.py` for the new documentation source

### 7.2 Update ArchonGraph

Modify `archon_graph.py` to include the new agent type:

```python
# Import the new agent
from archon.example_coder import example_coder, ExampleDeps

# In define_scope_with_reasoner function:
if agent_type == "Example Agent":
    # Get Example documentation pages
    from archon.example_coder import list_documentation_pages_helper as example_list_docs
    documentation_pages = await example_list_docs(supabase)
    source_filter = "example_docs"
```

### 7.3 Update UI

Modify the agent selection in `streamlit_ui.py`:

```python
# Update agent options
agent_options = ["", "Pydantic AI Agent", "Supabase Agent", "Example Agent"]
```

### 7.4 Add Documentation Tab Support

Update the documentation tab to support crawling example documentation:

```python
# Import crawler
from archon.crawl_example_docs import start_crawl_with_requests as start_example_crawl

# Add UI components for the new agent's docs
if docs_tab == "Example Docs":
    # Add crawl button and status for Example docs
```

## 8. Key SQL Functions

The application relies on a Supabase SQL function `match_site_pages` for semantic search:

```sql
CREATE FUNCTION match_site_pages(
  query_embedding vector(1536),
  match_threshold float DEFAULT 0.5,
  match_count int DEFAULT 5,
  filter jsonb DEFAULT '{}'
) RETURNS TABLE (
  id bigint,
  url text,
  title text,
  content text,
  embedding vector(1536),
  metadata jsonb,
  similarity float
) LANGUAGE plpgsql AS $$
BEGIN
  RETURN QUERY
  SELECT
    site_pages.id,
    site_pages.url,
    site_pages.title,
    site_pages.content,
    site_pages.embedding,
    site_pages.metadata,
    1 - (site_pages.embedding <=> query_embedding) AS similarity
  FROM
    site_pages
  WHERE
    site_pages.metadata @> filter AND
    1 - (site_pages.embedding <=> query_embedding) > match_threshold
  ORDER BY
    site_pages.embedding <=> query_embedding
  LIMIT
    match_count;
END;
$$;
```

## 9. Environment Configuration

Required environment variables include:

- `SUPABASE_URL`: URL of Supabase instance
- `SUPABASE_SERVICE_KEY`: Service key for Supabase
- `PRIMARY_MODEL`: Primary LLM (e.g., "gpt-4o-mini")
- `EMBEDDING_MODEL`: Embedding model (default: "text-embedding-3-small")
- `OPENAI_API_KEY`: OpenAI API key

These can be configured in `.env` file or set as environment variables.

---

This detailed documentation provides a comprehensive overview of the implementation of the Supabase and Pydantic AI agent system, focusing on the key components, workflows, and extension points for creating new agents.
