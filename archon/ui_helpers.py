"""
UI Helpers for Streamlit App

This module provides helper functions for the Streamlit UI, specifically for
creating documentation crawler tabs and other UI components.
"""

import os
import sys
import time
import asyncio
import threading
import uuid
import streamlit as st
import logging
from typing import Dict, List, Any, Optional, Callable, Tuple, Union

# Use the crawler registry for UI configuration
from archon.crawler_registry import get_all_ui_configs, get_crawler, get_crawler_ui_config

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_crawler_tab(
    tab_name: str,
    crawler_name: str,
    start_crawl_with_crawl4ai_func: Callable,
    start_crawl_with_requests_func: Callable,
    sync_clear_records_func: Callable,
    get_urls_to_crawl_func: Callable,
    fetch_doc_pages_func: Optional[Callable] = None
):
    """
    Create a standardized documentation crawler tab for the Streamlit UI.
    
    Args:
        tab_name: The display name for the tab
        crawler_name: The internal name of the crawler
        start_crawl_with_crawl4ai_func: Function to start crawl with Crawl4AI
        start_crawl_with_requests_func: Function to start crawl with requests
        sync_clear_records_func: Function to clear records
        get_urls_to_crawl_func: Function to get URLs to crawl
        fetch_doc_pages_func: Optional function to fetch documentation pages
    """
    # Initialize session state for this crawler
    if f"{crawler_name}_crawl_in_progress" not in st.session_state:
        st.session_state[f"{crawler_name}_crawl_in_progress"] = False
    
    if f"{crawler_name}_status" not in st.session_state:
        st.session_state[f"{crawler_name}_status"] = {
            "urls_found": 0,
            "urls_processed": 0,
            "urls_succeeded": 0,
            "urls_failed": 0,
            "elapsed_time": 0,
            "messages": [],
            "logs": []
        }
    
    # Function to update crawler status in session state
    def update_status(status):
        st.session_state[f"{crawler_name}_status"] = status
        
        # Check if crawl is complete
        if status.get("is_complete", False):
            st.session_state[f"{crawler_name}_crawl_in_progress"] = False
    
    # Display information about this crawler
    st.markdown(f"## {tab_name} Documentation")
    st.markdown(f"This section allows you to crawl and index the {tab_name} documentation. The crawler will:")
    
    st.markdown("""
    1. Fetch URLs from the {tab_name} sitemap
    2. Crawl each page and extract content
    3. Split content into chunks
    4. Generate embeddings for each chunk
    5. Store the chunks in the {tab_name} database
    """.format(tab_name=tab_name))
    
    st.markdown("This process may take several minutes depending on the number of pages.")
    
    # Create two columns for the main buttons
    col1, col2, col3 = st.columns(3)
    
    # URL limit slider (10-500)
    st.markdown("### Maximum URLs to crawl")
    url_limit = st.slider(
        "Maximum URLs to crawl",
        min_value=10,
        max_value=500,
        value=50,
        step=10,
        key=f"{crawler_name}_url_limit"
    )
    
    # Create crawl buttons
    with col1:
        # Button to start Crawl4AI crawler
        if st.button(f"Crawl {tab_name} (Crawl4AI)", key=f"{crawler_name}_crawl4ai"):
            if not st.session_state[f"{crawler_name}_crawl_in_progress"]:
                st.session_state[f"{crawler_name}_crawl_in_progress"] = True
                st.session_state[f"{crawler_name}_status"]["logs"] = []
                
                def run_async_crawl4ai():
                    # Create a new event loop for this thread
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        # Call the function directly - it's not a coroutine
                        start_crawl_with_crawl4ai_func(
                            progress_callback=update_status,
                            url_limit=url_limit
                        )
                    except Exception as e:
                        error_msg = f"Error starting Crawl4AI crawler: {str(e)}"
                        print(error_msg)
                        st.error(error_msg)
                        
                        # Update status to reflect the error
                        status = st.session_state[f"{crawler_name}_status"]
                        status["messages"].append(error_msg)
                        status["logs"].append(f"[ERROR] {error_msg}")
                        update_status(status)
                        
                        # Mark as not in progress
                        st.session_state[f"{crawler_name}_crawl_in_progress"] = False
                    finally:
                        loop.close()
                
                # Start the crawler in a separate thread
                crawler_thread = threading.Thread(target=run_async_crawl4ai)
                crawler_thread.start()
                
                st.success(f"Started {tab_name} Crawl4AI crawler (max URLs: {url_limit})")
            else:
                st.warning(f"{tab_name} crawl already in progress")
    
    with col2:
        # Button to start Requests crawler            
        if st.button(f"Crawl {tab_name} (Requests)", key=f"{crawler_name}_requests"):
            if not st.session_state[f"{crawler_name}_crawl_in_progress"]:
                st.session_state[f"{crawler_name}_crawl_in_progress"] = True
                st.session_state[f"{crawler_name}_status"]["logs"] = []
                
                def run_async_requests():
                    # Create a new event loop for this thread
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        # Call the function directly - it's not a coroutine
                        start_crawl_with_requests_func(
                            progress_callback=update_status,
                            url_limit=url_limit
                        )
                    except Exception as e:
                        error_msg = f"Error starting Requests crawler: {str(e)}"
                        print(error_msg)
                        st.error(error_msg)
                        
                        # Update status to reflect the error
                        status = st.session_state[f"{crawler_name}_status"]
                        status["messages"].append(error_msg)
                        status["logs"].append(f"[ERROR] {error_msg}")
                        update_status(status)
                        
                        # Mark as not in progress
                        st.session_state[f"{crawler_name}_crawl_in_progress"] = False
                    finally:
                        loop.close()
                
                # Start the crawler in a separate thread
                crawler_thread = threading.Thread(target=run_async_requests)
                crawler_thread.start()
                
                st.success(f"Started {tab_name} Requests crawler (max URLs: {url_limit})")
            else:
                st.warning(f"{tab_name} crawl already in progress")
    
    with col3:
        # Button to clear records
        if st.button(f"Clear {tab_name} Records", key=f"{crawler_name}_clear_button"):
            if not st.session_state[f"{crawler_name}_crawl_in_progress"]:
                # Run in a separate thread to avoid event loop conflicts
                def run_clear_records():
                    try:
                        sync_clear_records_func()
                        # Can't use st.success here as it's in a different thread
                        print(f"{tab_name} records cleared successfully")
                    except Exception as e:
                        print(f"Error clearing {tab_name} records: {str(e)}")
                
                clear_thread = threading.Thread(target=run_clear_records)
                clear_thread.start()
                st.success(f"Starting to clear {tab_name} records...")
            else:
                st.error("Cannot clear records while a crawl is in progress")
    
    # Create status metrics section
    st.markdown("### Status")
    status = st.session_state[f"{crawler_name}_status"]
    
    # Create columns for the metrics
    metrics_col1, metrics_col2, metrics_col3, metrics_col4, metrics_col5 = st.columns(5)
    
    with metrics_col1:
        st.metric("URLs Found", status.get("urls_found", 0))
    
    with metrics_col2:
        st.metric("URLs Processed", status.get("urls_processed", 0))
    
    with metrics_col3:
        st.metric("Successes", status.get("urls_succeeded", 0))
    
    with metrics_col4:
        st.metric("Failures", status.get("urls_failed", 0))
    
    with metrics_col5:
        st.metric("Elapsed Time", f"{status.get('elapsed_time', 0):.1f}s")
    
    # Progress indicator
    if st.session_state[f"{crawler_name}_crawl_in_progress"]:
        urls_found = status.get("urls_found", 0)
        urls_processed = status.get("urls_processed", 0)
        url_limit = status.get("url_limit", 0)
        
        # Determine what to display in the denominator
        if url_limit > 0 and url_limit < urls_found:
            progress_text = f"⏳ Crawling in progress... ({urls_processed}/{url_limit} URLs processed)"
            progress_value = urls_processed / url_limit if url_limit > 0 else 0
        else:
            progress_text = f"⏳ Crawling in progress... ({urls_processed}/{urls_found} URLs processed)"
            progress_value = urls_processed / max(urls_found, 1) if urls_found > 0 else 0
            
        st.progress(progress_value, text=progress_text)
    
    # Control buttons
    button_col1, button_col2 = st.columns(2)
    
    with button_col1:
        if st.button("🔄 Check Status", key=f"{crawler_name}_check_status_button"):
            # Explicitly get the current crawler status from the registry if available
            crawler = get_crawler(crawler_name)
            if crawler and "tracker" in crawler and crawler["tracker"]:
                # Update the session state with the latest status
                latest_status = crawler["tracker"].get_status()
                st.session_state[f"{crawler_name}_status"] = latest_status
                st.info(f"Status updated for {tab_name}")
            
            # Force a refresh of the UI
            st.rerun()
    
    with button_col2:
        if st.button("🛑 Stop Crawling", key=f"{crawler_name}_stop_button"):
            if st.session_state[f"{crawler_name}_crawl_in_progress"]:
                # Get the crawler from the registry to access the tracker
                crawler = get_crawler(crawler_name)
                if crawler and crawler.get("tracker"):
                    # Call the stop method on the tracker
                    crawler["tracker"].stop()
                    
                    # Immediately mark as not in progress for UI responsiveness
                    st.session_state[f"{crawler_name}_crawl_in_progress"] = False
                    
                    # Update the status
                    status = st.session_state[f"{crawler_name}_status"]
                    status["stop_requested"] = True
                    status["is_complete"] = True  # Force to complete for UI
                    status["messages"].append("Stop requested by user")
                    update_status(status)
                    
                    st.success("Crawl stopped successfully")
                    st.rerun()  # Force a UI refresh
                else:
                    # Fallback to just updating the session state
                    st.session_state[f"{crawler_name}_crawl_in_progress"] = False
                    status = st.session_state[f"{crawler_name}_status"]
                    status["messages"].append("Crawl stopped by user (tracker not found)")
                    status["stop_requested"] = True
                    status["is_complete"] = True  # Force to complete for UI
                    update_status(status)
                    st.success("Crawl stopped successfully")
                    st.rerun()  # Force a UI refresh
            else:
                st.warning("No crawl in progress")
    
    # Clear status button
    if st.button("🧹 Clear Status", key=f"{crawler_name}_clear_status_button"):
        st.session_state[f"{crawler_name}_status"] = {
            "urls_found": 0,
            "urls_processed": 0,
            "urls_succeeded": 0,
            "urls_failed": 0,
            "elapsed_time": 0,
            "messages": [],
            "logs": []
        }
        st.success(f"{tab_name} status cleared")
    
    # Display info about URLs to crawl
    st.markdown("### Available URLs to Crawl")
    
    # Only fetch URLs when explicitly requested
    if st.button("Fetch Available URLs", key=f"{crawler_name}_fetch_urls"):
        try:
            urls = get_urls_to_crawl_func()
            st.session_state[f"{crawler_name}_urls"] = urls
            st.success(f"Found {len(urls)} URLs to crawl")
            
            # Show a sample of URLs
            if len(urls) > 0:
                with st.expander("Show Sample URLs"):
                    st.write(urls[:5])
        except Exception as e:
            error_msg = f"Error getting URLs: {str(e)}"
            st.error(error_msg)
            
            # Show traceback for debugging
            import traceback
            st.expander("Error Details").code(traceback.format_exc())
    elif f"{crawler_name}_urls" in st.session_state:
        urls = st.session_state[f"{crawler_name}_urls"]
        st.info(f"Found {len(urls)} URLs to crawl")
        
        # Show a sample of URLs
        if len(urls) > 0:
            with st.expander("Show Sample URLs"):
                st.write(urls[:5])
    else:
        st.info("Click 'Fetch Available URLs' to see what URLs will be crawled")
    
    # Display info about documentation pages
    if fetch_doc_pages_func:
        st.markdown(f"### {tab_name} Pages")
        
        # Only fetch pages when explicitly requested
        if st.button("Fetch Documentation Pages", key=f"{crawler_name}_fetch_pages"):
            try:
                # Create a new thread and run the async function there to avoid event loop issues
                doc_pages = []
                
                def run_async_fetch():
                    nonlocal doc_pages
                    # Create a new event loop for this thread
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        # Run the async function in this thread's event loop
                        doc_pages = loop.run_until_complete(fetch_doc_pages_func())
                    finally:
                        loop.close()
                
                # Start a thread to run the async function
                thread = threading.Thread(target=run_async_fetch)
                thread.start()
                thread.join()  # Wait for the thread to complete
                
                # Store in session state
                st.session_state[f"{crawler_name}_doc_pages"] = doc_pages
                st.success(f"Found {len(doc_pages)} documentation pages")
                
                # Show a sample of documentation pages
                if len(doc_pages) > 0:
                    with st.expander("Show Sample Pages"):
                        st.write(doc_pages[:5])
            except Exception as e:
                st.error(f"Error getting documentation pages: {str(e)}")
        elif f"{crawler_name}_doc_pages" in st.session_state:
            doc_pages = st.session_state[f"{crawler_name}_doc_pages"]
            st.info(f"Found {len(doc_pages)} documentation pages")
            
            # Show a sample of documentation pages
            if len(doc_pages) > 0:
                with st.expander("Show Sample Pages"):
                    st.write(doc_pages[:5])
        else:
            st.info("Click 'Fetch Documentation Pages' to see what pages are available")
    
    # Display logs in an expander
    if len(status.get("logs", [])) > 0 or len(status.get("messages", [])) > 0:
        with st.expander("📋 Show Crawl Logs", expanded=True):
            # Create a container for logs with fixed height and auto-scroll
            log_container = st.container()
            
            # Use a scrollable container with fixed height
            with log_container:
                # Combine logs and messages
                all_logs = status.get("logs", [])
                all_messages = status.get("messages", [])
                
                # Display logs in a simpler way to ensure they're visible
                if all_logs:
                    for log in all_logs:
                        st.text(log)
                elif all_messages:
                    for msg in all_messages:
                        st.text(msg)
                else:
                    st.info("No logs available")

def update_crawler_status(crawler_name: str, status: Dict[str, Any]):
    """
    Update the crawler status in the session state.
    
    Args:
        crawler_name: The name of the crawler
        status: The status dictionary with updated values
    """
    st.session_state[f"{crawler_name}_status"] = status
    
    # Check if crawl is complete
    if status.get("is_complete", False):
        st.session_state[f"{crawler_name}_crawl_in_progress"] = False

def create_documentation_tabs():
    """
    Create documentation tabs based on registered crawlers.
    This function uses the crawler registry to dynamically create tabs.
    """
    # Get all UI configurations from the registry
    crawler_configs = get_all_ui_configs()
    
    if not crawler_configs:
        st.warning("No documentation crawlers registered")
        return
    
    # Create tabs
    tab_names = [config["tab_name"] for config in crawler_configs]
    tabs = st.tabs(tab_names)
    
    # Create content for each tab
    for i, (tab, config) in enumerate(zip(tabs, crawler_configs)):
        with tab:
            create_crawler_tab(
                tab_name=config["tab_name"],
                crawler_name=config["crawler_name"],
                start_crawl_with_crawl4ai_func=config["start_crawl_with_crawl4ai_func"],
                start_crawl_with_requests_func=config["start_crawl_with_requests_func"],
                sync_clear_records_func=config["sync_clear_records_func"],
                get_urls_to_crawl_func=config["get_urls_to_crawl_func"],
                fetch_doc_pages_func=config.get("fetch_doc_pages_func")
            ) 