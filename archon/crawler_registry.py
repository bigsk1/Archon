"""
Crawler Registry System

This module provides a centralized registry for documentation crawlers.
It works with the existing crawler implementations to provide a consistent way
to discover and use them in the UI and agent system.

===== HOW TO ADD A NEW CRAWLER =====

1. Create a new crawler file by copying crawler_template.py to a new file
   (e.g., archon/crawl_your_source_docs.py)

2. Implement the required functions in your new crawler file as described
   in the template comments
   
   IMPORTANT NAMING CONVENTION:
   - The filename must follow the pattern: crawl_xxx_docs.py
   - The SOURCE_NAME in your file must be: xxx_docs
   
   Example: For a file named crawl_supabase_docs.py, set SOURCE_NAME = "supabase_docs"
   
   This convention is required for auto-discovery to work correctly.

3. Add your crawler to the register_default_crawlers() function in this file
   (optional, as auto-discovery should find your crawler):

   ```python
   # Add your crawler here
   register_crawler(
       name="your_source_docs",  # Must match SOURCE_NAME in your file
       module_path="archon.crawl_your_source_docs",  # Python import path
       display_name="Your Source Docs",  # Optional, for display in UI
       keywords=["your", "source", "docs"],  # Optional, for search
       description="Documentation for Your Source",  # Optional
       enabled=True  # Set to False to disable temporarily
   )
   ```

4. Restart the application to see your new crawler in the UI

===== END HOW TO ADD A NEW CRAWLER =====
"""

import importlib
import inspect
import os
import sys
import logging
import glob
from typing import Dict, List, Callable, Any, Optional, Union
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("crawler_registry")

# Crawler registry dictionary
# Maps crawler names to metadata and functions
_crawler_registry = {}

def register_crawler(
    name: str,
    module_path: str,
    display_name: Optional[str] = None,
    keywords: Optional[List[str]] = None,
    description: str = "",
    enabled: bool = True
) -> bool:
    """
    Register a crawler with the registry.
    
    Args:
        name: Short name for the crawler (e.g., "supabase")
        module_path: Import path for the module (e.g., "archon.crawl_supabase_docs")
        display_name: Display name for the UI (defaults to capitalized name + " Docs")
        keywords: List of keywords for query matching
        description: Short description of the crawler
        enabled: Whether the crawler is enabled
        
    Returns:
        bool: True if registration was successful, False otherwise
    """
    if not display_name:
        display_name = f"{name.capitalize()} Docs"
        
    try:
        # Import the module
        module = importlib.import_module(module_path)
        
        # Required crawler functions
        required_functions = [
            "start_crawl_with_crawl4ai", 
            "start_crawl_with_requests", 
            "sync_clear_records"
        ]
        
        # Check if required functions exist
        missing_functions = []
        for func_name in required_functions:
            if not hasattr(module, func_name):
                missing_functions.append(func_name)
        
        if missing_functions:
            logger.error(f"Crawler {name} is missing required functions: {', '.join(missing_functions)}")
            return False
        
        # Try to find the source name in the module
        source_name = None
        if hasattr(module, "SOURCE_NAME"):
            source_name = getattr(module, "SOURCE_NAME")
        else:
            # Default source name is name_docs
            source_name = f"{name}_docs"
            logger.warning(f"SOURCE_NAME not found in {module_path}, using default: {source_name}")
        
        # Try to find get_urls_to_crawl function
        get_urls_func = None
        if hasattr(module, "get_urls_to_crawl"):
            get_urls_func = getattr(module, "get_urls_to_crawl")
        elif hasattr(module, "get_supabase_docs_urls") and name == "supabase":
            # Special case for supabase which uses a different function name
            get_urls_func = getattr(module, "get_supabase_docs_urls")
        else:
            logger.error(f"Crawler {name} is missing get_urls_to_crawl function")
            return False
            
        # Try to find list_documentation_pages_helper function
        list_pages_func = None
        if hasattr(module, "list_documentation_pages_helper"):
            list_pages_func = getattr(module, "list_documentation_pages_helper")
        
        # Store crawler info in registry
        _crawler_registry[name] = {
            "name": name,
            "source_name": source_name,
            "display_name": display_name,
            "module_path": module_path,
            "module": module,
            "start_crawl_with_crawl4ai": getattr(module, "start_crawl_with_crawl4ai"),
            "start_crawl_with_requests": getattr(module, "start_crawl_with_requests"),
            "sync_clear_records": getattr(module, "sync_clear_records"),
            "get_urls_to_crawl": get_urls_func,
            "list_documentation_pages_helper": list_pages_func,
            "keywords": keywords or [],
            "description": description,
            "enabled": enabled,
            "tracker": None  # Initialize tracker to None
        }
        
        logger.info(f"Successfully registered crawler: {name}")
        return True
        
    except ImportError as e:
        logger.error(f"Could not import module {module_path}: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Error registering crawler {name}: {str(e)}")
        return False

def get_crawler(name: str) -> Optional[Dict]:
    """
    Get a crawler by name.
    
    Args:
        name: Name of the crawler
    
    Returns:
        Dict or None: Crawler info if found, None otherwise
    """
    return _crawler_registry.get(name)

def get_all_crawlers() -> Dict[str, Dict]:
    """
    Get all registered crawlers.
    
    Returns:
        Dict: Dictionary of crawler name to crawler info
    """
    return _crawler_registry

def get_enabled_crawlers() -> Dict[str, Dict]:
    """
    Get all enabled crawlers.
    
    Returns:
        Dict: Dictionary of crawler name to crawler info
    """
    return {name: info for name, info in _crawler_registry.items() if info["enabled"]}

def get_crawler_ui_config(name: str) -> Dict[str, Any]:
    """
    Get UI configuration for a crawler.
    
    Args:
        name: Name of the crawler
    
    Returns:
        Dict: UI configuration
    """
    crawler = get_crawler(name)
    if not crawler:
        return {}
    
    return {
        "tab_name": crawler["display_name"],
        "crawler_name": crawler["name"],
        "start_crawl_with_crawl4ai_func": crawler["start_crawl_with_crawl4ai"],
        "start_crawl_with_requests_func": crawler["start_crawl_with_requests"],
        "sync_clear_records_func": crawler["sync_clear_records"],
        "get_urls_to_crawl_func": crawler["get_urls_to_crawl"],
        "fetch_doc_pages_func": crawler.get("list_documentation_pages_helper")
    }

def get_all_ui_configs() -> List[Dict[str, Any]]:
    """
    Get UI configurations for all enabled crawlers.
    
    Returns:
        List[Dict]: List of UI configurations
    """
    return [get_crawler_ui_config(name) for name, info in get_enabled_crawlers().items()]

# Generate document type enum for agent system
def generate_document_type_enum() -> str:
    """
    Generate DocumentType Enum code for the agent system.
    
    Returns:
        String containing Python code for DocumentType enum
    """
    enum_code = "class DocumentType(Enum):\n"
    enum_code += '    """Type of document to retrieve."""\n'
    
    # Add enum values for each crawler
    for name, info in get_enabled_crawlers().items():
        enum_code += f'    {name.upper()} = "{name}"\n'
    
    # Add NONE type
    enum_code += '    NONE = "none"\n'
    
    return enum_code

def generate_categorize_doc_type_method() -> str:
    """
    Generate _categorize_doc_type method code for the agent system.
    
    Returns:
        String containing Python code for _categorize_doc_type method
    """
    method_code = "def _categorize_doc_type(self, query: str) -> DocumentType:\n"
    method_code += '    """Determine the type of documentation the query is about."""\n'
    method_code += "    query = query.lower()\n\n"
    
    # Add keyword checks for each crawler
    for name, info in get_enabled_crawlers().items():
        if info["keywords"]:
            method_code += f"    # Check for {info['display_name']} keywords\n"
            method_code += f"    {name}_keywords = {repr(info['keywords'])}\n"
            method_code += f"    if any(keyword in query for keyword in {name}_keywords):\n"
            method_code += f"        return DocumentType.{name.upper()}\n\n"
    
    # Add default case
    method_code += "    # Default case - will check all sources\n"
    method_code += "    return DocumentType.NONE\n"
    
    return method_code

def auto_discover_crawlers():
    """
    Automatically discover crawlers in the archon directory.
    
    Looks for files matching the pattern 'crawl_*_docs.py'
    """
    # Get the directory where this module is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Crawlers pattern
    pattern = os.path.join(current_dir, "crawl_*_docs.py")
    
    # Find all matching files
    crawler_files = glob.glob(pattern)
    
    for file_path in crawler_files:
        # Get the filename without extension
        filename = os.path.basename(file_path)[:-3]
        
        # Extract crawler name from filename (e.g., "supabase" from "crawl_supabase_docs.py")
        # Also append "_docs" to match expected SOURCE_NAME format
        crawler_name = filename.replace("crawl_", "")
        
        # Check for special case of pydantic vs pydantic_ai to avoid duplicates
        if crawler_name == "pydantic_ai" and "pydantic" in _crawler_registry:
            # If we already have pydantic registered with the same module, skip this one
            if _crawler_registry["pydantic"]["module_path"] == f"archon.{filename}":
                continue
        
        # Skip if already registered with the exact same name
        if crawler_name in _crawler_registry:
            continue
        
        # Register the crawler
        register_crawler(
            name=crawler_name,
            module_path=f"archon.{filename}"
        )
    
    # Return the registry after discovery
    return _crawler_registry

def register_default_crawlers():
    """Register known default crawlers."""
    # These are the crawlers we know exist from the codebase
    # ================================================
    # HOW TO ADD A NEW CRAWLER: 
    # Add a new dictionary entry to this list with your crawler information.
    # Make sure your crawler module is properly implemented first by copying
    # the crawler_template.py file and implementing the required functions.
    # Update keywords and descrition based on your docs
    # ================================================
    defaults = [
        {
            "name": "supabase_docs",
            "module_path": "archon.crawl_supabase_docs",
            "display_name": "Supabase Docs",
            "keywords": ["supabase", "postgres", "postgresql", "rpc", "edge function", 
                         "storage", "auth", "realtime", "subscription"],
            "description": "Supabase documentation for building applications with Supabase"
        },
        {
            "name": "pydantic_docs",
            "module_path": "archon.crawl_pydantic_docs",
            "display_name": "Pydantic AI Docs",
            "keywords": ["pydantic", "basemodel", "schema", "validation", "validator", 
                        "dataclass", "field", "alias", "parse", "serialize", "pydantic v2", "root_validator",
                        "pydantic-ai", "agent", "tool", "system_prompt", "runcontext", "retry", "anthropic", "openai"],
            "description": "Pydantic AI documentation for building AI agents with Pydantic"
        },
        # ADD YOUR NEW CRAWLER HERE
    ]
    
    for crawler_info in defaults:
        register_crawler(**crawler_info)

# Initialize the registry
register_default_crawlers()
auto_discover_crawlers()

def list_available_crawlers():
    """Print a list of all available crawlers."""
    print("\n=== Available Documentation Crawlers ===\n")
    
    for name, info in get_all_crawlers().items():
        enabled = "✅ ENABLED" if info["enabled"] else "❌ DISABLED"
        print(f"{info['display_name']} ({name}) - {enabled}")
        print(f"  Source: {info['source_name']}")
        print(f"  Module: {info['module_path']}")
        if info["description"]:
            print(f"  Description: {info['description']}")
        print()

# Reset and initialize the registry to ensure no duplicates
def reset_and_initialize_registry():
    """Reset and initialize the crawler registry."""
    global _crawler_registry
    _crawler_registry = {}
    register_default_crawlers()
    auto_discover_crawlers()
    return _crawler_registry

# Initialize the registry
reset_and_initialize_registry()

def update_crawler_tracker(name: str, tracker) -> bool:
    """
    Update the tracker instance for a crawler.
    
    Args:
        name: Name of the crawler
        tracker: The tracker instance
    
    Returns:
        bool: True if update was successful, False otherwise
    """
    if name not in _crawler_registry:
        logger.error(f"Crawler {name} not found in registry")
        return False
    
    _crawler_registry[name]["tracker"] = tracker
    return True

if __name__ == "__main__":
    # Print registered crawlers when run directly
    list_available_crawlers() 