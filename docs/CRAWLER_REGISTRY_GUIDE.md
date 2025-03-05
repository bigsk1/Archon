# Crawler Registry Guide

This guide explains how to use the crawler registry system to add new documentation sources to Archon.

## Quick Start

The simplest way to create a new crawler is:

1. Copy the template file:
   ```bash
   cp archon/crawler_template.py archon/crawl_your_source_docs.py
   ```

2. Edit your new file:
   - Change SOURCE_NAME to "your_source_docs" (must match filename pattern)
   - Change BASE_URL to your documentation's URL
   - Customize the fallback_urls in get_urls_to_crawl function

3. Restart the application - your crawler will be auto-discovered

**Important:** The filename and SOURCE_NAME must follow the convention:
- Filename: `crawl_xxx_docs.py` 
- SOURCE_NAME: `xxx_docs`

For example, if your file is named `crawl_fastapi_docs.py`, your SOURCE_NAME must be `fastapi_docs`.

## Common Issues and Solutions

- **"Crawler not found in registry" error**: Make sure your SOURCE_NAME matches your filename pattern
- **"name 're' is not defined" error**: This means there's an issue with function calls in your crawler file. Make sure your main_with_crawl4ai and main_with_requests functions call `await clear_records()` and not `await clear_existing_records(SOURCE_NAME)`
- **UI slider defaulting to wrong value**: The default URL limit is set to 50, which can be changed in ui_helpers.py

## Overview

The Crawler Registry is a centralized system that manages documentation crawlers in the Archon application. It provides:

1. Automatic discovery and registration of crawler modules
2. UI tab generation for each registered crawler
3. Integration with the agent system for documentation retrieval
4. Consistent interface for crawler functions

## Core Components

The registry system consists of:

- **`crawler_registry.py`**: Central registry that manages crawler registration and discovery
- **Documentation Crawler Modules**: Individual crawler implementations (e.g., `crawl_fastapi_docs.py`)
- **UI Integration**: Dynamic tab generation in the Streamlit UI

## How to Add a New Documentation Source

### Step 1: Create a new crawler module

Create a new file in the `archon` directory with the naming convention `crawl_[name]_docs.py`.

Example: `crawl_langchain_docs.py` for LangChain documentation.

### Step 2: Implement required functions

Your crawler module must implement these required functions:

```python
# Required constants
# IMPORTANT: This SOURCE_NAME must match your file name without the "crawl_" prefix
# For file "crawl_langchain_docs.py", use SOURCE_NAME = "langchain_docs"
SOURCE_NAME = "your_source_docs"  # Used for identifying documents in the database

# Required functions
def get_urls_to_crawl(url_limit: int = 50) -> List[str]:
    """Return a list of URLs to crawl."""
    # Implementation here
    return urls

async def clear_records():
    """Clear existing records for this documentation source."""
    return await clear_existing_records(SOURCE_NAME)

async def main_with_crawl4ai(tracker=None, url_limit: int = 50):
    """Start crawling with Crawl4AI implementation."""
    # Be sure to call await clear_records() not clear_existing_records directly
    
async def main_with_requests(tracker=None, url_limit: int = 50):
    """Start crawling with requests implementation."""
    # Be sure to call await clear_records() not clear_existing_records directly
```

### Step 3: Register your crawler (Optional)

The system will auto-discover your crawler, but you can also explicitly register it:

```python
from archon.crawler_registry import register_crawler

register_crawler(
    name="your_source_docs",  # Must match SOURCE_NAME in your file
    module_path="archon.crawl_your_source_docs",
    display_name="Your Source Docs",
    keywords=["your", "source", "keywords"],
    description="Description of your documentation source",
    enabled=True
)
```

## How the Registry Works

1. On startup, the registry:
   - Registers default crawlers (Supabase, Pydantic, FastAPI)
   - Auto-discovers any additional crawler modules in the `archon` directory
   - Makes crawlers available to the UI and agent system

2. For the UI:
   - The `create_documentation_tabs()` function uses the registry to create tabs
   - Each tab has controls for crawling, viewing, and managing documentation

3. For the agent system:
   - The `generate_document_type_enum()` function creates a DocumentType enum
   - The agent uses this enum to determine which documentation to retrieve

## Functions of crawler_registry.py

- **Registration functions**:
  - `register_crawler()` - Register a new crawler
  - `register_default_crawlers()` - Register known crawlers
  - `auto_discover_crawlers()` - Discover crawlers from file patterns

- **Retrieval functions**:
  - `get_crawler()` - Get a crawler by name
  - `get_all_crawlers()` - Get all registered crawlers
  - `get_enabled_crawlers()` - Get only enabled crawlers

- **UI helper functions**:
  - `get_crawler_ui_config()` - Get UI configuration for a specific crawler
  - `get_all_ui_configs()` - Get UI configurations for all enabled crawlers

- **Agent helper functions**:
  - `generate_document_type_enum()` - Generate enum code
  - `generate_categorize_doc_type_method()` - Generate method code

## Best Practices

1. **Follow the naming convention** - Use `crawl_[name]_docs.py` for your crawler modules and set SOURCE_NAME to "[name]_docs"
2. **Implement all required functions** - Ensure your crawler has all the required functions
3. **Use the base crawler helpers** - Reuse functions from `base_crawler.py` when possible
4. **Add keywords** - Include relevant keywords for your documentation to help the agent
5. **Test crawling** - Test your crawler before deploying to production

## Example: Implementing a New Crawler

Here's a simplified example of a crawler for Django documentation:

```python
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Any
from archon.base_crawler import process_and_store_document, clear_existing_records

# Define source name - Must match filename pattern
SOURCE_NAME = "django_docs"
BASE_URL = "https://docs.djangoproject.com/en/stable/"

def get_urls_to_crawl(url_limit: int = 50) -> List[str]:
    """Get all URLs from the Django documentation."""
    response = requests.get(f"{BASE_URL}contents/")
    soup = BeautifulSoup(response.text, "html.parser")
    
    urls = []
    for link in soup.select("a.reference.internal"):
        href = link.get("href")
        if href and not href.startswith("#"):
            urls.append(f"{BASE_URL}{href}")
    
    return urls[:url_limit] if url_limit > 0 else urls

async def clear_records():
    """Clear existing Django documentation records."""
    return await clear_existing_records(SOURCE_NAME)

async def main_with_crawl4ai(tracker=None, url_limit: int = 50):
    """Start crawling Django docs with Crawl4AI."""
    # Create tracker if needed
    if not tracker:
        from archon.base_crawler import CrawlProgressTracker
        tracker = CrawlProgressTracker()
    
    try:
        # Clear existing records
        await clear_records()  # Use our function, not clear_existing_records directly
        
        # Get URLs and process them
        urls = get_urls_to_crawl(url_limit)
        # Rest of implementation...
    except Exception as e:
        print(f"Error: {str(e)}") 