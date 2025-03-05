from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai import Agent, RunContext
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated, List, Any, Dict, Optional, Tuple
from langgraph.config import get_stream_writer
from langgraph.types import interrupt
from dotenv import load_dotenv
from openai import AsyncOpenAI
from supabase import Client
import logfire
import os
import sys
from enum import Enum
import asyncio
import json
import logging
import time
from archon.pydantic_ai_coder import get_main_prompt
from archon.supabase_coder import get_main_prompt as get_supabase_prompt
from supabase import create_client
from datetime import datetime
from utils.utils import get_env_var
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    UserPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    RetryPromptPart,
    ModelMessagesTypeAdapter
)

# Import the crawler registry for document retrieval
from archon.crawler_registry import get_enabled_crawlers, get_crawler

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from archon.pydantic_ai_coder import pydantic_ai_coder, PydanticAIDeps, list_documentation_pages_helper
from archon.supabase_coder import supabase_coder, SupabaseDeps

# Load environment variables
load_dotenv()

# Configure logfire to suppress warnings (optional)
logfire.configure(send_to_logfire='never')

base_url = get_env_var('BASE_URL') or 'https://api.openai.com/v1'
api_key = get_env_var('LLM_API_KEY') or 'no-llm-api-key-provided'

is_ollama = "localhost" in base_url.lower()
is_anthropic = "anthropic" in base_url.lower()
is_openai = "openai" in base_url.lower()

reasoner_llm_model_name = get_env_var('REASONER_MODEL') or 'o3-mini'
reasoner_llm_model = AnthropicModel(reasoner_llm_model_name, api_key=api_key) if is_anthropic else OpenAIModel(reasoner_llm_model_name, base_url=base_url, api_key=api_key)

pydantic_reasoner = Agent(  
    reasoner_llm_model,
    system_prompt='You are an expert at coding AI agents with Pydantic AI and defining the scope for doing so.',  
)

supabase_reasoner = Agent(  
    reasoner_llm_model,
    system_prompt='You are an expert at building applications with Supabase and defining the scope for doing so.',  
)

primary_llm_model_name = get_env_var('PRIMARY_MODEL') or 'gpt-4o-mini'
primary_llm_model = AnthropicModel(primary_llm_model_name, api_key=api_key) if is_anthropic else OpenAIModel(primary_llm_model_name, base_url=base_url, api_key=api_key)

router_agent = Agent(  
    primary_llm_model,
    system_prompt='Your job is to route the user message either to the end of the conversation or to continue coding the application or agent.',  
)

end_conversation_agent = Agent(  
    primary_llm_model,
    system_prompt='Your job is to end a conversation for creating an application or agent by giving instructions for how to execute it and then saying a nice goodbye to the user.',  
)

openai_client=None

if is_ollama:
    openai_client = AsyncOpenAI(base_url=base_url,api_key=api_key)
elif get_env_var("OPENAI_API_KEY"):
    openai_client = AsyncOpenAI(api_key=get_env_var("OPENAI_API_KEY"))
else:
    openai_client = None

if get_env_var("SUPABASE_URL"):
    supabase: Client = Client(
        get_env_var("SUPABASE_URL"),
        get_env_var("SUPABASE_SERVICE_KEY")
    )
else:
    supabase = None

# Define state schema
class AgentState(TypedDict):
    latest_user_message: str
    messages: Annotated[List[bytes], lambda x, y: x + y]
    scope: str
    agent_type: str = "Pydantic AI Agent"  # Default to Pydantic AI Agent if not specified

# Use the registry to dynamically generate the DocumentType enum
from archon.crawler_registry import generate_document_type_enum

# Use the generated enum code
exec(generate_document_type_enum())

# Scope Definition Node with Reasoner LLM
async def define_scope_with_reasoner(state: AgentState):
    # Get the agent type from the state
    agent_type = state.get("agent_type", "Pydantic AI Agent")
    
    # Get the user message
    message = state["latest_user_message"]
    
    # Initialize OpenAI client
    client = AsyncOpenAI(
        api_key=get_env_var("OPENAI_API_KEY"),
    )
    
    # Get relevant documentation based on agent type
    doc_pages = []
    
    # Use the registry to get the correct crawler
    if "Pydantic" in agent_type:
        crawler = get_crawler("pydantic")
        if crawler and crawler.list_documentation_pages_helper:
            doc_pages = await crawler.list_documentation_pages_helper()
            source_filter = crawler.source_name
    elif "Supabase" in agent_type:
        crawler = get_crawler("supabase")
        if crawler and crawler.list_documentation_pages_helper:
            doc_pages = await crawler.list_documentation_pages_helper()
            source_filter = crawler.source_name
    elif "FastAPI" in agent_type:
        crawler = get_crawler("fastapi")
        if crawler and crawler.list_documentation_pages_helper:
            doc_pages = await crawler.list_documentation_pages_helper()
            source_filter = crawler.source_name
    else:
        # For any other agent type, try to extract the name and look up in registry
        agent_name = agent_type.lower().split()[0]  # e.g., "FastAPI" from "FastAPI Agent"
        crawler = get_crawler(agent_name)
        if crawler and crawler.list_documentation_pages_helper:
            doc_pages = await crawler.list_documentation_pages_helper()
            source_filter = crawler.source_name
        else:
            # Fallback to no documentation
            doc_pages = []
            source_filter = None
    
    # Log the documentation pages
    logging.info(f"Retrieved {len(doc_pages)} documentation pages for {agent_type}")
    
    # Construct prompt for the reasoner
    if "Pydantic" in agent_type:
        prompt = f"""
        I will give you a task or question. Your job is to create a detailed scope document for how to build a solution using Pydantic.
        
        For the task, create a scope document that includes:
        1. Feature requirements
        2. Pydantic models with all fields and validators needed
        3. Functions that would process these models
        4. Any utility code needed
        5. API design if applicable
        
        Available documentation pages: {len(doc_pages)} pages from the Pydantic documentation.
        
        User question:
        {message}
        """
    elif "Supabase" in agent_type:
        prompt = f"""
        I will give you a task or question. Your job is to create a detailed scope document for how to build a solution using Supabase.
        
        For the task, create a scope document that includes:
        1. Database tables with all columns and relationships
        2. API endpoints using Supabase that should be created
        3. Authentication and authorization approach
        4. Any frontend component requirements
        5. Integration plan with other services if applicable
        
        Available documentation pages: {len(doc_pages)} pages from the Supabase documentation.
        
        User question:
        {message}
        """
    elif "FastAPI" in agent_type:
        prompt = f"""
        I will give you a task or question. Your job is to create a detailed scope document for how to build a solution using FastAPI.
        
        For the task, create a scope document that includes:
        1. API architecture overview
        2. Detailed API endpoints with paths, methods, request/response models
        3. Dependency injection needs
        4. Database integration approach
        5. Authentication and middleware requirements
        6. Testing strategy
        
        Available documentation pages: {len(doc_pages)} pages from the FastAPI documentation.
        
        User question:
        {message}
        """
    else:
        # For any other agent type
        prompt = f"""
        I will give you a task or question. Your job is to create a detailed scope document for how to build a solution.
        
        For the task, create a scope document that includes:
        1. Feature requirements
        2. Architecture overview
        3. Component breakdown
        4. Implementation strategy
        5. Testing approach
        
        Available documentation pages: {len(doc_pages)} pages from the documentation.
        
        User question:
        {message}
        """
    
    # Get response from OpenAI
    response = await client.chat.completions.create(
        model="gpt-4-turbo",
        temperature=0,
        messages=[
            {"role": "system", "content": "You are a technical architect who specializes in creating detailed scope documents."},
            {"role": "user", "content": prompt}
        ]
    )
    
    # Extract the response
    scope = response.choices[0].message.content
    
    # Update state
    state["scope"] = scope
    
    return state

# Coding Node with Feedback Handling
async def coder_agent(state: AgentState, writer):    
    # Get the agent type from the state
    agent_type = state.get('agent_type', 'Pydantic AI Agent')
    
    # Print for debugging
    print(f"Coder using agent type: {agent_type}")
    
    # Get the message history into the format for Pydantic AI
    message_history: list[ModelMessage] = []
    for message_row in state['messages']:
        message_history.extend(ModelMessagesTypeAdapter.validate_json(message_row))

    if agent_type == "Pydantic AI Agent":
        # Prepare dependencies for Pydantic AI coder
        deps = PydanticAIDeps(
            supabase=supabase,
            openai_client=openai_client,
            reasoner_output=state['scope']
        )
        
        # Run the Pydantic AI coder agent
        if not is_openai:
            writer = get_stream_writer()
            result = await pydantic_ai_coder.run(state['latest_user_message'], deps=deps, message_history=message_history)
            writer(result.data)
        else:
            async with pydantic_ai_coder.run_stream(
                state['latest_user_message'],
                deps=deps,
                message_history=message_history
            ) as result:
                # Stream partial text as it arrives
                async for chunk in result.stream_text(delta=True):
                    writer(chunk)
    else:  # Supabase Agent
        # Prepare dependencies for Supabase coder
        deps = SupabaseDeps(
            supabase=supabase,
            openai_client=openai_client,
            reasoner_output=state['scope']
        )
        
        # Run the Supabase coder agent
        if not is_openai:
            writer = get_stream_writer()
            result = await supabase_coder.run(state['latest_user_message'], deps=deps, message_history=message_history)
            writer(result.data)
        else:
            async with supabase_coder.run_stream(
                state['latest_user_message'],
                deps=deps,
                message_history=message_history
            ) as result:
                # Stream partial text as it arrives
                async for chunk in result.stream_text(delta=True):
                    writer(chunk)

    return {"messages": [result.new_messages_json()]}

# Interrupt the graph to get the user's next message
def get_next_user_message(state: AgentState):
    value = interrupt({})

    # Set the user's latest message for the LLM to continue the conversation
    return {
        "latest_user_message": value
    }

# Determine if the user is finished creating their AI agent or not
async def route_user_message(state: AgentState):
    # Get the agent type from the state
    agent_type = state.get("agent_type", "Pydantic AI Agent")
    
    # Get the user message
    message = state["latest_user_message"]
    
    # Initialize OpenAI client
    client = AsyncOpenAI(
        api_key=get_env_var("OPENAI_API_KEY"),
    )
    
    # Construct prompt for the router
    if "Pydantic" in agent_type:
        prompt = f"""
        I've been asked to help with a Pydantic task. Decide whether I should:
        1. End the conversation and provide information
        2. Continue the conversation and help code a solution
        
        User message: {message}
        
        Only respond with either "FINISH" or "CONTINUE_CODING".
        """
    elif "Supabase" in agent_type:
        prompt = f"""
        I've been asked to help with a Supabase task. Decide whether I should:
        1. End the conversation and provide information
        2. Continue the conversation and help code a solution
        
        User message: {message}
        
        Only respond with either "FINISH" or "CONTINUE_CODING".
        """
    elif "FastAPI" in agent_type:
        prompt = f"""
        I've been asked to help with a FastAPI task. Decide whether I should:
        1. End the conversation and provide information
        2. Continue the conversation and help code a FastAPI application
        
        User message: {message}
        
        Only respond with either "FINISH" or "CONTINUE_CODING".
        """
    else:
        # Generic prompt for any other agent type
        agent_name = agent_type.split()[0]  # e.g., "FastAPI" from "FastAPI Agent"
        prompt = f"""
        I've been asked to help with a {agent_name} task. Decide whether I should:
        1. End the conversation and provide information
        2. Continue the conversation and help code a solution
        
        User message: {message}
        
        Only respond with either "FINISH" or "CONTINUE_CODING".
        """
    
    # Get response from OpenAI
    router_agent = await client.chat.completions.create(
        model="gpt-4-turbo",
        temperature=0,
        messages=[
            {"role": "system", "content": "You are a router that decides whether to end a conversation or continue coding."},
            {"role": "user", "content": prompt}
        ]
    )
    
    # Extract the response
    response = router_agent.choices[0].message.content.strip()
    
    if "FINISH" in response:
        return "finish"
    else:
        return "continue_coding"

# End of conversation agent to give instructions for executing the agent
async def finish_conversation(state: AgentState, writer):    
    # Get the message history into the format for Pydantic AI
    message_history: list[ModelMessage] = []
    for message_row in state['messages']:
        message_history.extend(ModelMessagesTypeAdapter.validate_json(message_row))

    # Run the agent in a stream
    if not is_openai:
        writer = get_stream_writer()
        result = await end_conversation_agent.run(state['latest_user_message'], message_history=message_history)
        writer(result.data)   
    else: 
        async with end_conversation_agent.run_stream(
            state['latest_user_message'],
            message_history=message_history
        ) as result:
            # Stream partial text as it arrives
            async for chunk in result.stream_text(delta=True):
                writer(chunk)

    return {"messages": [result.new_messages_json()]}        

# Build workflow
builder = StateGraph(AgentState)

# Add nodes
builder.add_node("define_scope_with_reasoner", define_scope_with_reasoner)
builder.add_node("coder_agent", coder_agent)
builder.add_node("get_next_user_message", get_next_user_message)
builder.add_node("finish_conversation", finish_conversation)

# Set edges
builder.add_edge(START, "define_scope_with_reasoner")
builder.add_edge("define_scope_with_reasoner", "coder_agent")
builder.add_edge("coder_agent", "get_next_user_message")
builder.add_conditional_edges(
    "get_next_user_message",
    route_user_message,
    {"coder_agent": "coder_agent", "finish_conversation": "finish_conversation"}
)
builder.add_edge("finish_conversation", END)

# Configure persistence
memory = MemorySaver()
agentic_flow = builder.compile(checkpointer=memory)