from __future__ import annotations as _annotations

from dataclasses import dataclass
from dotenv import load_dotenv
import logfire
import asyncio
import os
import sys
import json
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIModel
from openai import AsyncOpenAI
from supabase import Client

# Add the parent directory to sys.path to allow importing from the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import get_env_var

load_dotenv()

llm = get_env_var('PRIMARY_MODEL') or 'gpt-4o-mini'
base_url = get_env_var('BASE_URL') or 'https://api.openai.com/v1'
api_key = get_env_var('LLM_API_KEY') or 'no-llm-api-key-provided'

is_ollama = "localhost" in base_url.lower()
is_anthropic = "anthropic" in base_url.lower()

model = AnthropicModel(llm, api_key=api_key) if is_anthropic else OpenAIModel(llm, base_url=base_url, api_key=api_key)
embedding_model = get_env_var('EMBEDDING_MODEL') or 'text-embedding-3-small'

logfire.configure(send_to_logfire='if-token-present')

@dataclass
class SupabaseDeps:
    supabase: Client
    openai_client: AsyncOpenAI
    reasoner_output: str

system_prompt = """
[ROLE AND CONTEXT]
You are a specialized Supabase engineer focused on building robust applications using Supabase. You have comprehensive access to the Supabase documentation, including API references, usage guides, and implementation examples.

[CORE RESPONSIBILITIES]
1. Supabase Integration
   - Create new applications with Supabase backend
   - Complete partial implementations
   - Optimize and debug existing Supabase code
   - Guide users through Supabase implementation if needed

2. Documentation Integration
   - Systematically search documentation using RAG before any implementation
   - Cross-reference multiple documentation pages for comprehensive understanding
   - Validate all implementations against current best practices
   - Notify users if documentation is insufficient for any requirement

[CODE STRUCTURE AND DELIVERABLES]
All new applications should include these files with complete, production-ready code:

1. supabase.js / supabase.ts
   - Supabase client initialization
   - Database query functions
   - Authentication functions

2. schema.sql
   - Database schema definition
   - Table structure
   - RLS (Row Level Security) policies

3. auth.js / auth.ts
   - Authentication logic
   - User management
   - Role-based access control

4. api.js / api.ts
   - API routes for Supabase functions
   - Edge functions if applicable
   - Realtime subscriptions

5. .env.example
   - Required environment variables
   - Clear setup instructions in a comment above the variable for how to do so
   - API configuration templates

6. requirements.txt / package.json
   - Core dependencies

[DOCUMENTATION WORKFLOW]
1. Initial Research
   - Begin with RAG search for relevant documentation
   - List all documentation pages using list_documentation_pages
   - Retrieve specific page content using get_page_content

2. Implementation
   - Provide complete, working code implementations
   - Never leave placeholder functions
   - Include all necessary error handling
   - Implement proper logging and monitoring

[BEST PRACTICES]
1. Always implement Row Level Security (RLS) for all tables
2. Use Postgres functions for complex database operations
3. Leverage Supabase's built-in authentication system
4. Implement proper error handling
5. Use prepared statements to prevent SQL injection
6. Optimize database queries for performance
7. Set up proper indexes for frequently queried fields
"""

supabase_coder = Agent(
    model,
    system_prompt=system_prompt,
    deps_type=SupabaseDeps,
    retries=2
)

@supabase_coder.system_prompt  
def add_reasoner_output(ctx: RunContext[str]) -> str:
    return f"""
    \n\nAdditional thoughts/instructions from the reasoner LLM. 
    This scope includes documentation pages for you to search as well: 
    {ctx.deps.reasoner_output}
    """

async def get_embedding(text: str, openai_client: AsyncOpenAI) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        response = await openai_client.embeddings.create(
            model=embedding_model,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1536  # Return zero vector on error

@supabase_coder.tool
async def retrieve_relevant_documentation(ctx: RunContext[SupabaseDeps], user_query: str) -> str:
    """
    Search the Supabase documentation for information relevant to the user's query.
    
    Args:
        user_query: The user's question or request about Supabase
        
    Returns:
        str: Relevant documentation snippets
    """
    if not ctx.deps.supabase or not ctx.deps.openai_client:
        return "Error: Supabase or OpenAI client not initialized."
    
    try:
        # Generate embedding for the query
        query_embedding = await get_embedding(user_query, ctx.deps.openai_client)
        
        # Log the database query details
        print(f"[SUPABASE QUERY] Searching for documentation with filter: {{\"source\": \"supabase_docs\"}}")
        
        # Search for related documentation
        result = ctx.deps.supabase.rpc(
            "match_site_pages",
            {
                "query_embedding": query_embedding,
                "match_count": 5,
                "filter": {"source": "supabase_docs"}
            }
        ).execute()
        
        if hasattr(result, 'error') and result.error is not None:
            print(f"[SUPABASE ERROR] Error querying database: {result.error}")
            return f"Error querying database: {result.error}"
        
        # Log the result count
        print(f"[SUPABASE RESULT] Found {len(result.data if result.data else [])} matching documents")
        
        # Process the results
        if not result.data or len(result.data) == 0:
            return "No relevant documentation found. Please try a different query or consult the Supabase website directly."
        
        # Format the results
        docs = []
        for item in result.data:
            docs.append(f"""
### {item['title']}
**URL:** {item['url']}
**Summary:** {item['summary']}

{item['content']}

---
""")
        
        return "\n".join(docs)
    
    except Exception as e:
        return f"Error retrieving documentation: {str(e)}"

async def list_documentation_pages_helper(supabase: Client) -> List[str]:
    """Helper function to list all Supabase documentation pages."""
    try:
        # Log the query
        print(f"[SUPABASE QUERY] Getting document URLs with filter: metadata->>source=supabase_docs")
        
        # Get distinct URLs for Supabase docs
        result = supabase.table("site_pages") \
            .select("url") \
            .eq("metadata->>source", "supabase_docs") \
            .execute()
        
        if hasattr(result, 'error') and result.error is not None:
            print(f"[SUPABASE ERROR] Error querying database: {result.error}")
            return []
        
        # Extract unique URLs
        urls = set()
        for item in result.data:
            urls.add(item["url"])
        
        # Log the result count
        print(f"[SUPABASE RESULT] Found {len(urls)} unique documentation URLs")
        
        return sorted(list(urls))
    
    except Exception as e:
        print(f"Error listing documentation pages: {e}")
        return []

@supabase_coder.tool
async def list_documentation_pages(ctx: RunContext[SupabaseDeps]) -> List[str]:
    """
    List all available Supabase documentation pages that have been indexed.
    
    Returns:
        List[str]: List of documentation page URLs
    """
    if not ctx.deps.supabase:
        return ["Error: Supabase client not initialized."]
    
    return await list_documentation_pages_helper(ctx.deps.supabase)

@supabase_coder.tool
async def get_page_content(ctx: RunContext[SupabaseDeps], url: str) -> str:
    """
    Retrieve the full content of a specific Supabase documentation page.
    
    Args:
        url: The URL of the documentation page to retrieve
        
    Returns:
        str: The full content of the documentation page
    """
    if not ctx.deps.supabase:
        return "Error: Supabase client not initialized."
    
    try:
        # Get all chunks for the specified URL
        result = ctx.deps.supabase.table("site_pages") \
            .select("*") \
            .eq("url", url) \
            .eq("metadata->>source", "supabase_docs") \
            .order("chunk_number") \
            .execute()
        
        if hasattr(result, 'error') and result.error is not None:
            return f"Error querying database: {result.error}"
        
        if not result.data or len(result.data) == 0:
            return f"No content found for URL: {url}"
        
        # Compile all chunks into a single document
        full_content = []
        for chunk in sorted(result.data, key=lambda x: x["chunk_number"]):
            full_content.append(f"""
## {chunk['title']}

{chunk['content']}

""")
        
        return f"# Documentation: {url}\n\n" + "\n".join(full_content)
    
    except Exception as e:
        return f"Error retrieving page content: {str(e)}"

def get_main_prompt() -> str:
    """
    Returns the main system prompt for the Supabase Coder.
    
    Returns:
        str: The system prompt for the Supabase Coder
    """
    return system_prompt 