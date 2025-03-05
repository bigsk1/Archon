"""
Template for creating a new documentation crawler.

====== INSTRUCTIONS FOR CREATING A NEW CRAWLER ======

1. Copy this file to a new file named for your crawler (e.g., `fastapi_docs.py`)
2. Make the following REQUIRED changes:
   - Update SOURCE_NAME (e.g., "fastapi_docs")
   - Update BASE_URL to your documentation site URL
   - Implement get_urls_to_crawl() to fetch URLs from your documentation
   
3. Make the following OPTIONAL changes if needed:
   - Custom content cleanup in main_with_crawl4ai/main_with_requests
   - Custom URL filtering logic

4. DO NOT change these parts as they ensure consistent behavior:
   - The helper function creation at the bottom of the file
   - The imports from base_crawler.py
   - Function signatures for the standard functions

====== END INSTRUCTIONS ======

IMPORTANT: This template is designed to work with the base_crawler.py functions.
DO NOT implement your own versions of crawl_with_crawl4ai or crawl_with_requests.
Instead, use the versions provided in base_crawler.py which handle all URL tracking,
error handling, and progress reporting consistently.
"""

import os
import sys
import asyncio
import threading
import requests
from typing import List, Dict, Any, Optional, Callable
import re
from urllib.parse import urlparse
from dotenv import load_dotenv

# Import the base crawler functionality directly
from archon.base_crawler import (
    CrawlProgressTracker, 
    crawl_with_crawl4ai,
    crawl_parallel_with_crawl4ai,
    crawl_with_requests,
    crawl_parallel_with_requests,
    process_and_store_document,
    clear_existing_records,
    create_crawler_ui_helpers,
    process_html_to_markdown,
    generate_embeddings,
    chunk_text
)

# IMPORTANT NOTE:
# When using AsyncWebCrawler directly, always initialize it with the `config` parameter,
# NOT `browser_config`. Example: `AsyncWebCrawler(config=browser_config)`.
# Using the wrong parameter name can cause "multiple values for keyword argument" errors.

# Define the source name for this crawler
# IMPORTANT: This name MUST match your file name without the "crawl_" prefix and "_docs" suffix
# Example: For file "crawl_fastapi_docs.py", use SOURCE_NAME = "fastapi_docs"
# This ensures auto-discovery works correctly and avoids registry issues
SOURCE_NAME = "name_docs"

# CHANGE THIS: Replace with the base URL for the documentation
BASE_URL = "https://REPLACE_WITH_YOUR_DOCS_DOMAIN.com"

def get_urls_to_crawl(url_limit: int = 50) -> List[str]:
    """Get all documentation URLs to crawl.
    
    Args:
        url_limit: Maximum number of URLs to return. Set to 0 or negative for no limit.
        
    Returns:
        List of URLs to crawl
    """
    # Ensure url_limit is properly handled - if it's zero or negative from UI, set a very high number as "no limit"
    effective_limit = 999999 if url_limit <= 0 else url_limit
    
    print(f"Starting to fetch URLs from {BASE_URL} with limit: {url_limit} (effective: {effective_limit})")
    
    # CUSTOMIZE: Define site-specific fallback URLs to use if sitemap fails
    # These should point to important landing pages for the documentation
    fallback_urls = [
        f"{BASE_URL}/docs/introduction",
        f"{BASE_URL}/docs/getting-started",
        f"{BASE_URL}/docs/guides/basic-usage",
        # Add more URLs as needed
    ]
    
    # Function to extract URLs from a sitemap - this handles both direct sitemaps and sitemap indexes
    def extract_urls_from_sitemap(sitemap_url: str, namespace: str = '{http://www.sitemaps.org/schemas/sitemap/0.9}', 
                                  depth: int = 0, max_depth: int = 3) -> List[str]:
        """Extract URLs from a sitemap, handling recursive sitemaps.
        
        Args:
            sitemap_url: URL of the sitemap to process
            namespace: XML namespace for the sitemap format
            depth: Current recursion depth (for internal use)
            max_depth: Maximum recursion depth to prevent infinite loops
            
        Returns:
            List of extracted URLs
        """
        # Prevent excessive recursion
        if depth > max_depth:
            print(f"Maximum sitemap recursion depth ({max_depth}) reached for {sitemap_url}")
            return []
            
        try:
            print(f"Fetching sitemap from {sitemap_url}")
            response = requests.get(sitemap_url, timeout=10)
            if response.status_code != 200:
                print(f"Failed to fetch sitemap {sitemap_url}: HTTP {response.status_code}")
                return []
                
            try:
                import xml.etree.ElementTree as ET
                root = ET.fromstring(response.content)
            except Exception as e:
                print(f"Failed to parse sitemap XML from {sitemap_url}: {e}")
                return []
            
            # First, check if this is a sitemap index (contains other sitemaps)
            sitemap_urls = []
            for sitemap in root.findall(f".//{namespace}sitemap"):
                loc_element = sitemap.find(f"{namespace}loc")
                if loc_element is not None and loc_element.text:
                    sitemap_urls.append(loc_element.text)
            
            # If this is a sitemap index, recursively fetch all sitemaps
            all_page_urls = []
            if sitemap_urls:
                print(f"Found sitemap index with {len(sitemap_urls)} child sitemaps")
                for child_sitemap_url in sitemap_urls:
                    # CUSTOMIZE: Filter which nested sitemaps to process based on your site's structure
                    # Example: Skip certain sitemaps that aren't relevant to documentation
                    # if "blog" in child_sitemap_url or "news" in child_sitemap_url:
                    #     continue
                    
                    child_urls = extract_urls_from_sitemap(child_sitemap_url, namespace, depth + 1, max_depth)
                    all_page_urls.extend(child_urls)
                return all_page_urls
            
            # If this is a regular sitemap (not an index), extract page URLs
            page_urls = []
            for url_element in root.findall(f".//{namespace}url"):
                loc_element = url_element.find(f"{namespace}loc")
                if loc_element is not None and loc_element.text:
                    page_urls.append(loc_element.text)
            
            if page_urls:
                print(f"Found {len(page_urls)} direct page URLs in {sitemap_url}")
            
            return page_urls
            
        except Exception as e:
            print(f"Error processing sitemap {sitemap_url}: {e}")
            return []
    
    # Try multiple sitemap URLs - these are common patterns across many sites
    sitemap_paths = [
        "/sitemap.xml",
        "/docs/sitemap.xml",
        "/sitemap-0.xml",
        "/sitemap_docs.xml",
        "/sitemap_index.xml",
        "/latest/sitemap.xml",  # For sites with versioned docs like Pydantic
        "",  # For when the base URL itself is the sitemap
    ]
    
    # CUSTOMIZE: Add or remove sitemap paths specific to your documentation site
    
    # Collect all URLs from the various sitemaps
    all_urls = []
    for path in sitemap_paths:
        try:
            sitemap_url = f"{BASE_URL}{path}"
            urls = extract_urls_from_sitemap(sitemap_url)
            if urls:
                print(f"Successfully extracted {len(urls)} URLs from {sitemap_url}")
                all_urls.extend(urls)
        except Exception as e:
            print(f"Error processing sitemap at {path}: {e}")
            continue
    
    # Use fallback URLs if we couldn't extract any from sitemaps
    if not all_urls:
        print(f"No URLs found from sitemaps. Using {len(fallback_urls)} fallback URLs.")
        all_urls = fallback_urls
    
    # CUSTOMIZE: Filter URLs to include only documentation pages
    # Modify these filters based on your site's URL structure
    filtered_urls = []
    for url in all_urls:
        # This is a more flexible approach to filtering documentation URLs
        # It handles various common documentation URL patterns
        
        # Skip URLs that are clearly not documentation
        if any(exclude in url.lower() for exclude in [
            '/blog/', '/news/', '/archive/', '/legacy/', '/draft/', 
            '/community/', '/forum/', '/download/', '/releases/'
        ]):
            continue
            
        # Include URLs that match common documentation patterns
        base_domain = BASE_URL.split('//')[1].split('/')[0]  # Extract domain without protocol and path
        if (
            # Standard docs path
            '/docs/' in url or 
            # API reference paths
            '/api/' in url or '/reference/' in url or
            # Guide/tutorial paths
            '/guide/' in url or '/tutorial/' in url or
            # For sites with versioned docs (like Pydantic)
            '/latest/' in url or '/stable/' in url or '/current/' in url or
            # For sites with docs directly at the root (check if it's from our domain)
            base_domain in url
        ):
            filtered_urls.append(url)
    
    # Deduplicate URLs while preserving order
    seen = set()
    unique_urls = []
    for url in filtered_urls:
        if url not in seen:
            seen.add(url)
            unique_urls.append(url)
    
    print(f"After filtering and deduplication: {len(unique_urls)} unique documentation URLs")
    
    # Apply URL limit if specified
    if url_limit > 0 and len(unique_urls) > url_limit:
        print(f"Limiting to {url_limit} URLs (from {len(unique_urls)} total)")
        return unique_urls[:url_limit]
    
    return unique_urls

async def process_url_with_requests(url: str, tracker: Optional[CrawlProgressTracker] = None) -> Optional[str]:
    """
    Fetch content from a URL using requests (alternative to Crawl4AI).
    Implement this if you need a simpler crawler alternative.
    """
    if tracker:
        tracker.log(f"Fetching with requests: {url}")
    
    try:
        response = requests.get(url, timeout=30)
        
        if response.status_code == 200:
            # Convert HTML to markdown
            import html2text
            h = html2text.HTML2Text()
            h.ignore_links = False
            h.ignore_images = False
            h.ignore_tables = False
            h.body_width = 0  # No wrapping
            content = h.handle(response.text)
            
            if tracker:
                tracker.log(f"Successfully fetched content from {url} - {len(content)} characters")
            
            return content
        else:
            if tracker:
                tracker.log(f"Failed to fetch {url} - Status code: {response.status_code}")
            return None
    except Exception as e:
        if tracker:
            tracker.log(f"Error fetching {url}: {str(e)}")
        return None

async def crawl_parallel_with_requests(urls: List[str], tracker: Optional[CrawlProgressTracker] = None, max_concurrent: int = 5, source: str = "unknown"):
    """
    Crawl multiple URLs in parallel using requests.
    This is an alternative to the Crawl4AI method.
    
    Args:
        urls: List of URLs to crawl
        tracker: Optional tracker for progress updates
        max_concurrent: Maximum number of concurrent requests
        source: Source name for logging and tracking
    """
    # Get URL limit from tracker if available
    url_limit = tracker._status.get("url_limit", 0) if tracker else 0
    
    # If we have a limit and it's less than the number of URLs, trim the list
    # Use a deterministic, non-configurable approach
    if url_limit > 0:
        if len(urls) > url_limit:
            if tracker:
                tracker.log(f"[crawl_parallel_with_requests] Strictly enforcing URL limit: {url_limit}")
            # Trim to exactly the URL limit
            urls = urls[:url_limit]
        if tracker:
            tracker.log(f"[crawl_parallel_with_requests] Will process EXACTLY {len(urls)} URLs")
    else:
        if tracker:
            tracker.log(f"[crawl_parallel_with_requests] No URL limit set, will process all {len(urls)} URLs")
    
    # Create a semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent)
    processed_urls = set()
    # Add a safety counter to track processed URLs
    processed_count = 0
    max_to_process = url_limit if url_limit > 0 else float('inf')
    
    async def process_url(url: str, url_index: int):
        nonlocal processed_count
        
        # Hard safety check - never process more than the URL limit
        if processed_count >= max_to_process:
            if tracker:
                tracker.log(f"[Safety] Already processed {processed_count} URLs, at limit {max_to_process}")
            return
            
        # Skip if we're stopping
        if tracker and tracker.stop_requested:
            return
            
        # Skip if already processed
        if url in processed_urls:
            if tracker:
                tracker.log(f"Skipping already processed URL: {url}")
            return
            
        # Add to processed URLs
        processed_urls.add(url)
        
        # Acquire semaphore
        async with semaphore:
            # Another check to make sure we don't exceed the limit
            if processed_count >= max_to_process:
                return
                
            processed_count += 1
                
            # Check again if we should stop
            if tracker and tracker.stop_requested:
                return
                
            if tracker:
                tracker.log(f"Processing URL {processed_count}/{max_to_process}: {url}")
            
            if tracker:
                tracker._status["current_url"] = url
                tracker.processed_urls.add(url)
            
            # Get content
            content = None
            try:
                content = await process_url_with_requests(url, tracker)
            except Exception as e:
                if tracker:
                    tracker.log(f"Error fetching {url}: {str(e)}")
                    tracker.urls_failed += 1
                    tracker.urls_processed += 1
                return
            
            # Process the content
            if content:
                try:
                    # Store the document and get number of chunks stored
                    chunks_stored = await process_and_store_document(url, content, tracker, source)
                    
                    # Update tracker
                    if tracker:
                        if chunks_stored > 0:
                            tracker.urls_succeeded += 1
                        else:
                            tracker.urls_failed += 1
                        tracker.urls_processed += 1
                        tracker.log(f"Processed: {url} - Stored {chunks_stored} chunks")
                except Exception as e:
                    # Update tracker for failed URLs
                    if tracker:
                        tracker.log(f"Error processing {url}: {str(e)}")
                        tracker.urls_failed += 1
                        tracker.urls_processed += 1
            else:
                # Update tracker for failed URLs
                if tracker:
                    tracker.log(f"Failed to process: {url} - No content retrieved")
                    tracker.urls_failed += 1
                    tracker.urls_processed += 1
            
            # Update tracker activity
            if tracker:
                tracker.update_activity()
                if tracker.progress_callback:
                    tracker.progress_callback(tracker.get_status())
    
    # Create exactly url_limit tasks or fewer
    tasks = []
    if tracker:
        tracker.log(f"Creating tasks with strict limit of {max_to_process}")
        
    for i, url in enumerate(urls):
        # Don't even create tasks beyond the limit
        if i >= max_to_process:
            if tracker:
                tracker.log(f"[Task creation] Hard stop at {i} tasks due to URL limit of {max_to_process}")
            break
            
        # Check if we should stop before creating more tasks
        if tracker and tracker.stop_requested:
            tracker.log("Stopping before processing remaining URLs")
            break
            
        # Pass the index to the task
        tasks.append(process_url(url, i))
    
    if tracker:
        tracker.log(f"Created EXACTLY {len(tasks)} tasks to process URLs (limit: {max_to_process})")
    
    # Wait for all tasks to complete
    try:
        await asyncio.gather(*tasks)
    except Exception as e:
        if tracker:
            tracker.log(f"Error in crawl_parallel_with_requests: {str(e)}")
            
    if tracker:
        tracker.log(f"Completed processing {processed_count} URLs")

async def main_with_crawl4ai(tracker: Optional[CrawlProgressTracker] = None, url_limit: int = 50):
    """
    Main function to crawl documentation using Crawl4AI.
    
    Args:
        tracker: Optional CrawlProgressTracker to track progress
        url_limit: Maximum number of URLs to crawl
    """
    # Create a tracker if none is provided
    if tracker is None:
        tracker = CrawlProgressTracker()
    
    # Start tracking
    tracker.start()
    tracker.log(f"Starting {SOURCE_NAME} docs crawl with Crawl4AI")
    tracker.log(f"URL limit set to: {url_limit}")
    
    try:
        # Clear existing records if needed
        tracker.log("Clearing existing records...")
        await clear_records()
        tracker.log("Existing records cleared")
        
        # Fetch URLs to crawl
        tracker.log("Fetching URLs to crawl...")
        all_urls = get_urls_to_crawl(url_limit)
        tracker.log(f"Initial URL count from get_urls_to_crawl: {len(all_urls)}")
        
        # Apply URL limit if specified and needed (enforcement layer)
        if url_limit > 0 and len(all_urls) > url_limit:
            tracker.log(f"Limiting URL processing from {len(all_urls)} to {url_limit}")
            all_urls = all_urls[:url_limit]
            tracker.log(f"URLs after limiting: {len(all_urls)}")
        
        # Store URL limit in tracker for progress calculation
        tracker._status["url_limit"] = url_limit if url_limit > 0 else len(all_urls)
        tracker.urls_found = len(all_urls)
        tracker._status["urls_found"] = len(all_urls)
        tracker.log(f"Found {len(all_urls)} URLs to crawl")
        
        # Update progress
        if tracker.progress_callback:
            tracker.progress_callback(tracker.get_status())
        
        # Check if we have any URLs to process
        if not all_urls:
            tracker.log("No URLs found to crawl")
            tracker.complete(is_successful=False)
            return
        
        # Check if crawl was stopped before processing started
        if tracker.stop_requested:
            tracker.log("Crawl stopped before processing started")
            tracker.complete(is_successful=False)
            return
        
        # Process URLs in parallel using base_crawler function
        tracker.log(f"Starting parallel crawl with Crawl4AI on {len(all_urls)} URLs...")
        await crawl_parallel_with_crawl4ai(all_urls, tracker, source=SOURCE_NAME)
        
        # Complete tracking
        tracker.log(f"Crawl completed. Processed {tracker.urls_processed} URLs: {tracker.urls_succeeded} succeeded, {tracker.urls_failed} failed")
        tracker.complete(is_successful=tracker.urls_succeeded > 0)
        
    except Exception as e:
        error_msg = f"Error in main_with_crawl4ai: {str(e)}"
        print(error_msg)
        tracker.log(error_msg)
        tracker.complete(is_successful=False)
        # Update progress one last time
        if tracker.progress_callback:
            tracker.progress_callback(tracker.get_status())

async def main_with_requests(tracker: Optional[CrawlProgressTracker] = None, url_limit: int = 50):
    """
    Main function to crawl documentation using requests.
    
    Args:
        tracker: Optional CrawlProgressTracker to track progress
        url_limit: Maximum number of URLs to crawl
    """
    # Create a tracker if none is provided
    if tracker is None:
        tracker = CrawlProgressTracker()
    
    # Start tracking
    tracker.start()
    tracker.log(f"Starting {SOURCE_NAME} docs crawl with requests")
    tracker.log(f"URL limit set to: {url_limit}")
    
    try:
        # Clear existing records if needed
        tracker.log("Clearing existing records...")
        await clear_records()
        tracker.log("Existing records cleared")
        
        # Fetch URLs to crawl
        tracker.log("Fetching URLs to crawl...")
        all_urls = get_urls_to_crawl(url_limit)
        tracker.log(f"Initial URL count from get_urls_to_crawl: {len(all_urls)}")
        
        # Apply URL limit if specified and needed (enforcement layer)
        if url_limit > 0 and len(all_urls) > url_limit:
            tracker.log(f"Limiting URL processing from {len(all_urls)} to {url_limit}")
            all_urls = all_urls[:url_limit]
            tracker.log(f"URLs after limiting: {len(all_urls)}")
        
        # Store URL limit in tracker for progress calculation
        tracker._status["url_limit"] = url_limit if url_limit > 0 else len(all_urls)
        tracker.urls_found = len(all_urls)
        tracker._status["urls_found"] = len(all_urls)
        tracker.log(f"Found {len(all_urls)} URLs to crawl")
        
        # Update progress
        if tracker.progress_callback:
            tracker.progress_callback(tracker.get_status())
        
        # Check if we have any URLs to process
        if not all_urls:
            tracker.log("No URLs found to crawl")
            tracker.complete(is_successful=False)
            return
        
        # Check if crawl was stopped before processing started
        if tracker.stop_requested:
            tracker.log("Crawl stopped before processing started")
            tracker.complete(is_successful=False)
            return
        
        # Process URLs in parallel using base_crawler function
        tracker.log(f"Starting parallel crawl with requests on {len(all_urls)} URLs...")
        await crawl_parallel_with_requests(all_urls, tracker, source=SOURCE_NAME)
        
        # Complete tracking
        tracker.log(f"Crawl completed. Processed {tracker.urls_processed} URLs: {tracker.urls_succeeded} succeeded, {tracker.urls_failed} failed")
        tracker.complete(is_successful=tracker.urls_succeeded > 0)
        
    except Exception as e:
        error_msg = f"Error in main_with_requests: {str(e)}"
        print(error_msg)
        tracker.log(error_msg)
        tracker.complete(is_successful=False)
        # Update progress one last time
        if tracker.progress_callback:
            tracker.progress_callback(tracker.get_status())

async def clear_records():
    """Clear existing records for this crawler."""
    return await clear_existing_records(SOURCE_NAME)

# Create the UI helper functions
helpers = create_crawler_ui_helpers(
    source_name=SOURCE_NAME,
    main_with_crawl4ai_func=main_with_crawl4ai,
    main_with_requests_func=main_with_requests,
    get_urls_to_crawl_func=get_urls_to_crawl,
    clear_existing_records_func=clear_records
)

# Export the helper functions for use in the UI
start_crawl_with_crawl4ai = helpers["start_crawl_with_crawl4ai"]
start_crawl_with_requests = helpers["start_crawl_with_requests"]
sync_clear_records = helpers["sync_clear_records"] 