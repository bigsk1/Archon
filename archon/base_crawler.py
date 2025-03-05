import os
import sys
import asyncio
import threading
import requests
import json
import html2text
import re
from typing import List, Dict, Any, Optional, Callable, Set
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse
from dotenv import load_dotenv

# Add the parent directory to sys.path to allow importing from the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import get_env_var

# Import Crawl4AI components
try:
    from crawl4ai import AsyncWebCrawler, BrowserConfig
except ImportError:
    print("WARNING: crawl4ai package not found. Crawl4AI functionality will not be available.")
    
from openai import AsyncOpenAI
from supabase import create_client, Client

load_dotenv()

# Initialize OpenAI and Supabase clients
base_url = get_env_var('BASE_URL') or 'https://api.openai.com/v1'
api_key = get_env_var('LLM_API_KEY') or 'no-llm-api-key-provided'
is_ollama = "localhost" in base_url.lower()

embedding_model = get_env_var('EMBEDDING_MODEL') or 'text-embedding-3-small'

openai_client = None

if is_ollama:
    openai_client = AsyncOpenAI(base_url=base_url, api_key=api_key)
else:
    openai_client = AsyncOpenAI(api_key=get_env_var("OPENAI_API_KEY"))

supabase: Client = create_client(
    get_env_var("SUPABASE_URL"),
    get_env_var("SUPABASE_SERVICE_KEY")
)

@dataclass
class ProcessedChunk:
    """Data structure for a processed document chunk ready for storage."""
    url: str
    chunk_number: int
    title: str
    summary: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]

class CrawlProgressTracker:
    """Tracks the progress of a crawl operation."""
    
    def __init__(self, progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None):
        """
        Initialize a new progress tracker.
        
        Args:
            progress_callback: Optional callback function to report progress
        """
        self.start_time = datetime.now(timezone.utc)
        self.end_time = None
        self.is_completed = False
        self.is_successful = False
        self.stop_requested = False
        self.progress_callback = progress_callback
        
        # URLs tracking
        self.urls_found = 0
        self.urls_processed = 0
        self.urls_succeeded = 0
        self.urls_failed = 0
        self.processed_urls = set()
        
        # Chunks tracking
        self.chunks_processed = 0
        self.chunks_stored = 0
        
        # Last activity time
        self.last_activity = datetime.now(timezone.utc)
        
        # Status dictionary
        self._status = {
            "start_time": self.start_time.isoformat(),
            "elapsed_time": 0,
            "is_complete": False,
            "is_successful": False,
            "stop_requested": False,
            "urls_found": 0,
            "urls_processed": 0,
            "urls_succeeded": 0,
            "urls_failed": 0,
            "chunks_processed": 0,
            "chunks_stored": 0,
            "progress_percentage": 0,
            "current_url": "",
            "url_limit": 0,
            "logs": [],
            "messages": [],
            "last_activity": self.last_activity.isoformat()
        }
    
    def log(self, message: str):
        """
        Log a message and update the status.
        
        Args:
            message: The message to log
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        print(log_entry)
        
        self._status["logs"].append(log_entry)
        self._status["messages"].append(message)
        
        # Update tracking counts in status
        self._status["urls_found"] = self.urls_found
        self._status["urls_processed"] = self.urls_processed
        self._status["urls_succeeded"] = self.urls_succeeded
        self._status["urls_failed"] = self.urls_failed
        self._status["chunks_processed"] = self.chunks_processed
        self._status["chunks_stored"] = self.chunks_stored
        
        # Calculate elapsed time
        if not self.end_time:
            self._status["elapsed_time"] = (datetime.now(timezone.utc) - self.start_time).total_seconds()
        
        # Update last activity
        self.update_activity()
        
        # Call the progress callback if provided
        if self.progress_callback:
            self.progress_callback(self.get_status())
    
    def start(self):
        """Mark the tracker as started."""
        self.start_time = datetime.now(timezone.utc)
        self.log("Crawl started")
    
    def complete(self, is_successful: bool = True):
        """
        Mark the tracker as completed.
        
        Args:
            is_successful: Whether the crawl was successful
        """
        if not self.is_completed:
            self.is_completed = True
            self.is_successful = is_successful
            self.end_time = datetime.now(timezone.utc)
            
            self._status["is_complete"] = True
            self._status["is_successful"] = is_successful
            
            elapsed = self.end_time - self.start_time
            self._status["elapsed_time"] = elapsed.total_seconds()
            
            status = "successfully" if is_successful else "with errors"
            self.log(f"Crawl completed {status} in {elapsed.total_seconds():.2f} seconds")
            
            # Final callback
            if self.progress_callback:
                self.progress_callback(self.get_status())
    
    def stop(self):
        """Request the crawl to stop."""
        self.stop_requested = True
        self._status["stop_requested"] = True
        self.log("Stop requested - crawler will stop after current operation")
        
        # Call the progress callback if provided
        if self.progress_callback:
            self.progress_callback(self.get_status())
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the crawl.
        
        Returns:
            Dict with status information
        """
        # Update elapsed time if still running
        if not self.is_completed and not self.end_time:
            self._status["elapsed_time"] = (datetime.now(timezone.utc) - self.start_time).total_seconds()
        
        # Calculate progress percentage
        urls_found = self._status.get("urls_found", 0)
        urls_processed = self._status.get("urls_processed", 0)
        url_limit = self._status.get("url_limit", 0)
        
        # Calculate progress based on either the limit or total urls found
        if url_limit > 0 and url_limit < urls_found:
            progress = (urls_processed / url_limit) * 100 if url_limit > 0 else 0
        else:
            progress = (urls_processed / urls_found) * 100 if urls_found > 0 else 0
            
        self._status["progress_percentage"] = min(100, progress)
        self._status["is_complete"] = self.is_completed
        self._status["is_successful"] = self.is_successful
        self._status["stop_requested"] = self.stop_requested
        
        return self._status

    def update_activity(self):
        """Update the last activity timestamp."""
        self.last_activity = datetime.now(timezone.utc)
        self._status["last_activity"] = self.last_activity.isoformat()

def clean_markdown_content(markdown: str) -> str:
    """Clean and normalize markdown content."""
    if not markdown:
        return ""
    
    # Remove multiple newlines
    cleaned = re.sub(r'\n{3,}', '\n\n', markdown)
    
    # Remove HTML comments
    cleaned = re.sub(r'<!--.*?-->', '', cleaned, flags=re.DOTALL)
    
    # Replace tabs with spaces
    cleaned = cleaned.replace('\t', '    ')
    
    # Remove any remaining HTML tags that weren't converted
    cleaned = re.sub(r'<[^>]*>', '', cleaned)
    
    return cleaned.strip()

def chunk_text(text: str, chunk_size: int = 5000) -> List[str]:
    """Split text into chunks of approximately equal size, preserving paragraph boundaries where possible."""
    if not text or len(text) <= chunk_size:
        return [text] if text else []
    
    # Split by double newlines (paragraphs)
    paragraphs = re.split(r'\n\s*\n', text)
    
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        # If adding this paragraph exceeds the chunk size and we already have content,
        # save the current chunk and start a new one
        if len(current_chunk) + len(paragraph) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = paragraph
        # If the paragraph itself exceeds chunk size, split it further
        elif len(paragraph) > chunk_size:
            # If we have current content, save it first
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            
            # Split long paragraph by sentences or fail-safe by characters
            sentences = re.split(r'(?<=[.!?])\s+', paragraph)
            temp_chunk = ""
            
            for sentence in sentences:
                if len(temp_chunk) + len(sentence) > chunk_size and temp_chunk:
                    chunks.append(temp_chunk.strip())
                    temp_chunk = sentence
                else:
                    temp_chunk += " " + sentence if temp_chunk else sentence
            
            if temp_chunk:
                current_chunk = temp_chunk
        else:
            # Add paragraph to current chunk
            current_chunk += "\n\n" + paragraph if current_chunk else paragraph
    
    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

async def get_title_and_summary(chunk: str, url: str) -> Dict[str, str]:
    """Extract title and summary using an LLM."""
    system_prompt = """You are an AI that extracts titles and summaries from documentation chunks.
    Return a JSON object with 'title' and 'summary' keys.
    For the title: If this seems like the start of a document, extract its title. If it's a middle chunk, derive a descriptive title.
    For the summary: Create a concise summary of the main points in this chunk.
    Keep both title and summary concise but informative."""
    
    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",  # Always use gpt-4o-mini for this task
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"URL: {url}\n\nContent:\n{chunk[:1000]}..."}  # Send first 1000 chars for context
            ],
            response_format={ "type": "json_object" }
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error getting title and summary: {e}")
        # Return placeholder title and summary when there's an error
        return {
            "title": f"Documentation: {url.split('/')[-1]}",
            "summary": "This is a documentation page."
        }

async def get_embedding(text: str) -> List[float]:
    """Generate an embedding for the given text using OpenAI's API."""
    try:
        # Truncate text if too long
        if len(text) > 8000:
            text = text[:8000]
            
        # Call the OpenAI API to generate the embedding
        response = await openai_client.embeddings.create(
            model=embedding_model,
            input=text
        )
        
        # Return the embedding as a list of floats
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        raise

async def process_chunk(chunk: str, chunk_number: int, url: str, source: str) -> ProcessedChunk:
    """
    Process a single chunk of text and prepare it for storage.
    
    Args:
        chunk: The text chunk to process
        chunk_number: The chunk number within the document
        url: The source URL
        source: The source name (e.g., "supabase_docs")
        
    Returns:
        ProcessedChunk: The processed chunk ready for storage
    """
    try:
        # Extract title and summary
        try:
            extracted = await get_title_and_summary(chunk, url)
            title = extracted.get("title", f"Documentation: {url.split('/')[-1]}")
            summary = extracted.get("summary", "Documentation chunk")
        except Exception as e:
            print(f"Error extracting title and summary for chunk {chunk_number} from {url}: {e}")
            # Fallback values
            title = f"Documentation: {url.split('/')[-1]}"
            summary = "Documentation chunk"
        
        # Generate embedding
        try:
            embedding = await get_embedding(chunk)
        except Exception as e:
            print(f"Error generating embedding for chunk {chunk_number} from {url}: {e}")
            # Empty embedding as fallback
            embedding = []
        
        # Prepare metadata
        metadata = {
            "source": source,
            "url": url,
            "chunk_number": chunk_number,
            "processed_at": datetime.now().isoformat()
        }
        
        return ProcessedChunk(
            url=url,
            chunk_number=chunk_number,
            title=title,
            summary=summary, 
            content=chunk,
            metadata=metadata,
            embedding=embedding
        )
    except Exception as e:
        print(f"Error processing chunk {chunk_number} from {url}: {e}")
        # Return a minimal valid chunk to avoid downstream errors
        return ProcessedChunk(
            url=url,
            chunk_number=chunk_number,
            title=f"Error processing: {url.split('/')[-1]}",
            summary="Error during processing", 
            content=chunk[:1000] if chunk else "Error: No content",
            metadata={"source": source, "url": url, "error": str(e)},
            embedding=[]
        )

async def insert_chunk(chunk: ProcessedChunk) -> bool:
    """
    Insert a processed chunk into the Supabase database.
    
    Args:
        chunk: The processed chunk to insert
        
    Returns:
        bool: True if insertion was successful, False otherwise
    """
    try:
        # Check if Supabase client is initialized
        if not supabase:
            print("Supabase client not initialized")
            return False
            
        # Check if chunk has required fields
        if not chunk.url or not chunk.content:
            print(f"Chunk missing required fields: url={bool(chunk.url)}, content={bool(chunk.content)}")
            return False
        
        # Insert the chunk
        result = supabase.table("site_pages").insert({
            "url": chunk.url,
            "chunk_number": chunk.chunk_number,
            "title": chunk.title,
            "summary": chunk.summary,
            "content": chunk.content,
            "embedding": chunk.embedding,
            "metadata": chunk.metadata
        }).execute()
        
        if hasattr(result, 'error') and result.error:
            print(f"Error inserting chunk: {result.error}")
            return False
        
        return True
    except Exception as e:
        print(f"Exception inserting chunk: {e}")
        return False

async def process_and_store_document(url: str, markdown: str, tracker: Optional[CrawlProgressTracker] = None, source: str = "unknown") -> int:
    """Process a document and store its chunks in the database.
    
    Args:
        url: The URL of the document
        markdown: The markdown content to process
        tracker: Optional tracker for progress updates
        source: The source name for the document
        
    Returns:
        int: Number of chunks successfully stored
    """
    # Clean the markdown content
    cleaned_content = clean_markdown_content(markdown)
    
    # Check if content is empty after cleaning
    if not cleaned_content:
        if tracker:
            tracker.log(f"No content to process for {url} after cleaning")
        else:
            print(f"No content to process for {url} after cleaning")
        return 0
    
    # Split into chunks
    chunks = chunk_text(cleaned_content)
    
    # Check if any chunks were created
    if not chunks:
        if tracker:
            tracker.log(f"No chunks created for {url}")
        else:
            print(f"No chunks created for {url}")
        return 0
    
    # Process each chunk
    processed_chunks = []
    for i, chunk in enumerate(chunks):
        if tracker and tracker.stop_requested:
            if tracker:
                tracker.log(f"Stopping chunk processing for {url}")
            break
            
        if tracker:
            tracker.log(f"Processing chunk {i+1}/{len(chunks)} for {url}")
        
        try:
            processed_chunk = await process_chunk(chunk, i+1, url, source)
            processed_chunks.append(processed_chunk)
            
            if tracker:
                tracker.chunks_processed += 1
                # Update progress
                if tracker.progress_callback:
                    tracker.progress_callback(tracker.get_status())
        except Exception as e:
            error_msg = f"Error processing chunk {i+1}/{len(chunks)} for {url}: {str(e)}"
            if tracker:
                tracker.log(error_msg)
            else:
                print(error_msg)
    
    # Store chunks in database
    chunks_stored = 0
    for chunk in processed_chunks:
        if tracker and tracker.stop_requested:
            if tracker:
                tracker.log(f"Stopping chunk storage for {url}")
            break
        
        try:
            success = await insert_chunk(chunk)
            if success:
                chunks_stored += 1
        except Exception as e:
            error_msg = f"Error storing chunk for {url}: {str(e)}"
            if tracker:
                tracker.log(error_msg)
            else:
                print(error_msg)
    
    if tracker:
        tracker.chunks_stored += chunks_stored
        tracker.log(f"Stored {chunks_stored}/{len(processed_chunks)} chunks for {url}")
        # Ensure UI gets updated
        if tracker.progress_callback:
            tracker.progress_callback(tracker.get_status())
    else:
        print(f"Stored {chunks_stored}/{len(processed_chunks)} chunks for {url}")
    
    return chunks_stored

async def crawl_with_crawl4ai(url: str, tracker: Optional[CrawlProgressTracker] = None, max_retries: int = 3) -> Optional[str]:
    """
    Crawl a single URL using Crawl4AI.
    
    Args:
        url: URL to crawl
        tracker: Optional CrawlProgressTracker to track progress
        max_retries: Maximum number of retry attempts
        
    Returns:
        Optional[str]: The cleaned content as markdown, or None if failed
        
    Note:
        IMPORTANT: When using AsyncWebCrawler directly, always initialize it with the `config` parameter,
        NOT `browser_config`. Example: `AsyncWebCrawler(config=browser_config)`. Using the wrong parameter
        name can cause "multiple values for keyword argument" errors.
    """
    if tracker:
        tracker.log(f"Crawling with Crawl4AI: {url}")
    
    # Configure the browser
    browser_config = BrowserConfig(
        headless=True,
        ignore_https_errors=True,
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        viewport_width=1280,
        viewport_height=800
    )
    
    # Implement retry logic
    for attempt in range(1, max_retries + 1):
        try:
            if attempt > 1 and tracker:
                tracker.log(f"Retry attempt {attempt}/{max_retries} for {url}")
            
            # Use async with context manager pattern to ensure proper initialization and cleanup
            async with AsyncWebCrawler(config=browser_config) as crawler:
                # Use arun method which is the correct method in v0.5.0
                result = await crawler.arun(url)
                
                # Extract content based on what's available in the result
                content = ""
                
                # In v0.5.0, the result has a markdown property
                if hasattr(result, 'markdown') and result.markdown:
                    content = result.markdown
                    if tracker:
                        tracker.log(f"Got markdown content from {url} - {len(content)} characters")
                elif hasattr(result, 'html') and result.html:
                    if tracker:
                        tracker.log(f"Converting HTML to markdown for {url}")
                    h = html2text.HTML2Text()
                    h.ignore_links = False
                    h.ignore_images = False
                    h.ignore_tables = False
                    h.body_width = 0  # No wrapping
                    content = h.handle(result.html)
                elif hasattr(result, 'text') and result.text:
                    if tracker:
                        tracker.log(f"Using plain text content for {url}")
                    content = result.text
                
                if content:
                    return content
                else:
                    if tracker:
                        tracker.log(f"No content found for {url}")
                    return None
                    
        except Exception as e:
            if tracker:
                tracker.log(f"Error crawling {url} with Crawl4AI: {str(e)}")
            
            # If this was the last retry, return None
            if attempt == max_retries:
                if tracker:
                    tracker.log(f"Failed to crawl {url} after {max_retries} attempts")
                return None
            
            # Wait before retrying
            await asyncio.sleep(2)
    
    return None

async def crawl_parallel_with_crawl4ai(urls: List[str], tracker: Optional[CrawlProgressTracker] = None, max_concurrent: int = 5, source: str = "unknown"):
    """Crawl multiple URLs in parallel using Crawl4AI."""
    # Create a semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Keep track of processed URLs to avoid duplicates
    processed_urls = set()
    
    async def process_url(url: str):
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
            # Check again if we should stop
            if tracker and tracker.stop_requested:
                return
                
            if tracker:
                tracker.log(f"Processing URL: {url}")
            
            if tracker:
                tracker.processed_urls.add(url)
            
            # Get content
            content = await crawl_with_crawl4ai(url, tracker)
            
            # Process the content
            if content:
                # Store the document
                await process_and_store_document(url, content, tracker, source)
                
                # Update tracker
                if tracker:
                    tracker.urls_succeeded += 1
                    tracker.urls_processed += 1
                    tracker.log(f"Successfully processed: {url}")
            else:
                # Update tracker for failed URLs
                if tracker:
                    tracker.log(f"Failed to process: {url} - No content retrieved")
                    tracker.urls_failed += 1
                    tracker.urls_processed += 1
            
            # Update tracker activity
            if tracker:
                tracker.update_activity()
                # Explicitly update progress
                if tracker.progress_callback:
                    tracker.progress_callback(tracker.get_status())
    
    # Create tasks for all URLs
    tasks = [process_url(url) for url in urls]
    
    # Wait for all tasks to complete
    await asyncio.gather(*tasks)

async def clear_existing_records(source: str):
    """Clear existing documentation records with the specified source."""
    try:
        # Verify Supabase client is initialized
        if not supabase:
            error_msg = "Supabase client is not initialized"
            print(error_msg)
            return {"error": error_msg}
            
        # Verify Supabase URL and key are present
        if not get_env_var("SUPABASE_URL") or not get_env_var("SUPABASE_SERVICE_KEY"):
            error_msg = "Supabase URL or service key is missing"
            print(error_msg)
            return {"error": error_msg}
            
        print(f"Clearing existing records with source: {source}")
        result = supabase.table("site_pages").delete().filter("metadata->>source", "eq", source).execute()
        
        print(f"Clear result: {result}")
        
        if hasattr(result, 'error') and result.error:
            error_msg = f"Error clearing records: {result.error}"
            print(error_msg)
            return {"error": error_msg}
            
        return {"success": True, "message": f"Successfully cleared all {source} records"}
    except Exception as e:
        import traceback
        error_msg = f"Exception clearing records: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        return {"error": error_msg}

def create_crawler_ui_helpers(
    source_name: str, 
    main_with_crawl4ai_func, 
    main_with_requests_func, 
    get_urls_to_crawl_func, 
    clear_existing_records_func
):
    """
    Create standard UI helper functions for a crawler.
    
    Args:
        source_name: Name of the documentation source (e.g., "pydantic_ai_docs")
        main_with_crawl4ai_func: The main function to run crawl with Crawl4AI
        main_with_requests_func: The main function to run crawl with Requests
        get_urls_to_crawl_func: Function to get URLs to crawl
        clear_existing_records_func: Function to clear existing records
        
    Returns:
        Dict containing start_crawl_with_crawl4ai, start_crawl_with_requests, sync_clear_records functions
    """
    
    def start_crawl_with_crawl4ai(progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None, url_limit: int = 50) -> CrawlProgressTracker:
        """Start the crawling process using Crawl4AI in a separate thread and return the tracker."""
        tracker = CrawlProgressTracker(progress_callback)
        
        # Store the tracker in the registry
        from archon.crawler_registry import update_crawler_tracker
        update_crawler_tracker(source_name, tracker)
        
        # Store the URL limit in the tracker status
        tracker._status["url_limit"] = url_limit
        
        def run_crawl():
            try:
                # Pass the URL limit to main_with_crawl4ai
                asyncio.run(main_with_crawl4ai_func(tracker, url_limit=url_limit))
            except Exception as e:
                print(f"Error in crawl thread: {e}")
                tracker.log(f"Thread error: {str(e)}")
                tracker.complete(is_successful=False)
                if progress_callback:
                    progress_callback(tracker.get_status())
        
        # Start the crawling process in a separate thread
        thread = threading.Thread(target=run_crawl)
        thread.daemon = True
        thread.start()
        
        return tracker
    
    def start_crawl_with_requests(progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None, url_limit: int = 50) -> CrawlProgressTracker:
        """Start the crawling process using Requests in a separate thread and return the tracker."""
        tracker = CrawlProgressTracker(progress_callback)
        
        # Store the tracker in the registry
        from archon.crawler_registry import update_crawler_tracker
        update_crawler_tracker(source_name, tracker)
        
        # Store the URL limit in the tracker status
        tracker._status["url_limit"] = url_limit
        
        def run_crawl():
            try:
                # Pass the URL limit to main_with_requests
                asyncio.run(main_with_requests_func(tracker, url_limit=url_limit))
            except Exception as e:
                print(f"Error in crawl thread: {e}")
                tracker.log(f"Thread error: {str(e)}")
                tracker.complete(is_successful=False)
                if progress_callback:
                    progress_callback(tracker.get_status())
        
        # Start the crawling process in a separate thread
        thread = threading.Thread(target=run_crawl)
        thread.daemon = True
        thread.start()
        
        return tracker
    
    def sync_clear_records():
        """Clear existing records synchronously using a thread to avoid event loop conflicts."""
        print(f"Starting to clear {source_name} records...")
        
        def run_clear():
            try:
                # Create a new event loop in this thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                # Run the clear function
                result = loop.run_until_complete(clear_existing_records_func())
                loop.close()
                
                print(f"Successfully cleared {source_name} records")
                return result
            except Exception as e:
                print(f"Error clearing {source_name} records: {e}")
                import traceback
                traceback.print_exc()
                return {"error": str(e)}
        
        # Run in a separate thread to avoid event loop conflicts
        clear_thread = threading.Thread(target=run_clear)
        clear_thread.start()
        
        # Return immediately - the clearing happens in background
        return {"status": "clearing_in_progress"}
        
    return {
        "start_crawl_with_crawl4ai": start_crawl_with_crawl4ai,
        "start_crawl_with_requests": start_crawl_with_requests,
        "sync_clear_records": sync_clear_records,
        "get_urls_to_crawl": get_urls_to_crawl_func
    }

# HTML to Markdown converter
html_to_markdown = html2text.HTML2Text()
html_to_markdown.ignore_links = False
html_to_markdown.ignore_images = False
html_to_markdown.ignore_tables = False
html_to_markdown.body_width = 0  # No wrapping

def get_env_var(var_name: str, default: str = None) -> str:
    """
    Get an environment variable value or return the default.
    
    Args:
        var_name: Name of the environment variable
        default: Default value to return if variable is not set
        
    Returns:
        str: Value of the environment variable or default
    """
    return os.environ.get(var_name, default)

async def generate_embeddings(text: str, client=None, model: str = None) -> List[float]:
    """
    Generate embedding vector for text using the configured embedding model.
    
    Args:
        text: The text to embed
        client: Optional OpenAI client to use (if not provided, a new one will be created)
        model: Optional model name to override the default embedding model
        
    Returns:
        List[float]: The embedding vector
    """
    from openai import AsyncOpenAI
    
    if not text:
        return []
    
    # Get embedding model from env var if not specified
    if not model:
        model = get_env_var('EMBEDDING_MODEL') or 'text-embedding-3-small'
    
    # Initialize client if not provided
    if not client:
        base_url = get_env_var('BASE_URL') or 'https://api.openai.com/v1'
        api_key = get_env_var('LLM_API_KEY') or get_env_var('OPENAI_API_KEY')
        is_ollama = "localhost" in base_url.lower()
        
        if is_ollama:
            client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        else:
            client = AsyncOpenAI(api_key=api_key)
    
    try:
        response = await client.embeddings.create(
            input=text,
            model=model
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        # Return empty embedding on error
        return []

def process_html_to_markdown(html_content: str, clean_content: bool = True) -> str:
    """
    Convert HTML content to markdown.
    
    Args:
        html_content: HTML content to convert
        clean_content: Whether to clean the markdown content after conversion
        
    Returns:
        str: Converted markdown content
    """
    if not html_content:
        return ""
    
    # Convert HTML to markdown
    markdown_content = html_to_markdown.handle(html_content)
    
    # Clean the content if requested
    if clean_content:
        markdown_content = clean_markdown_content(markdown_content)
    
    return markdown_content

def clean_markdown_content(content: str) -> str:
    """
    Clean markdown content by removing artifacts that might confuse an AI model.
    Focus on preserving semantic meaning rather than visual formatting.
    
    Args:
        content: The markdown content to clean
        
    Returns:
        str: The cleaned markdown content
    """
    if not content:
        return content
    
    # Remove _NUMBER patterns (e.g., _10, _23) that appear as artifacts
    cleaned = re.sub(r'_\d+\s+', ' ', content)
    cleaned = re.sub(r'\s+_\d+', ' ', cleaned)
    
    # Remove line numbers at the beginning of lines that might confuse the AI
    # This handles patterns like "1 -- Schema" or "34 --"
    cleaned = re.sub(r'^\s*\d+\s+', '', cleaned, flags=re.MULTILINE)
    
    # Remove adjacent line numbers that might confuse the model (like "15 return secret_value; 16 end; 17")
    cleaned = re.sub(r';\s*\d+\s+', '; ', cleaned)
    cleaned = re.sub(r'\)\s*\d+\s+', ') ', cleaned)
    
    # Fix common code formatting issues that might confuse the model
    cleaned = re.sub(r'(\d+)(\s+)(select|create|insert|update|delete|alter|drop)', r'\2\3', cleaned)
    
    # Ensure headers are properly formatted for the AI to recognize them
    cleaned = re.sub(r'(#+)(\w)', r'\1 \2', cleaned)
    
    # Preserve important semantic markers like code blocks
    # Make sure SQL code blocks are properly delimited
    if '$$' in cleaned and not '```sql' in cleaned:
        cleaned = re.sub(r'\$\$', r'```sql', cleaned, count=1)
        cleaned = re.sub(r'\$\$', r'```', cleaned, count=1)
        # Handle any remaining $$ pairs
        cleaned = re.sub(r'\$\$', r'```sql', cleaned, count=1)
        cleaned = re.sub(r'\$\$', r'```', cleaned, count=1)
    
    # Ensure proper spacing for list items so the AI recognizes them
    cleaned = re.sub(r'(\*|\d+\.)\s*(\w)', r'\1 \2', cleaned)
    
    return cleaned.strip()

async def process_text_with_embedding(url: str, title: str, content: str, metadata: Dict[str, Any]) -> List[ProcessedChunk]:
    """
    Process text content, split into chunks, and generate embeddings.
    
    Args:
        url: Source URL for the content
        title: Title of the document
        content: Markdown content to process
        metadata: Additional metadata to store
        
    Returns:
        List[ProcessedChunk]: List of processed chunks with embeddings
    """
    if not content:
        return []
    
    # Chunk the content
    chunks = chunk_text(content)
    processed_chunks = []
    
    # Process each chunk
    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            continue
            
        # Generate embedding
        embedding = await generate_embeddings(chunk)
        
        # Create a processed chunk
        processed_chunk = ProcessedChunk(
            url=url,
            chunk_number=i,
            title=title,
            summary="",  # Summary can be generated separately if needed
            content=chunk,
            metadata=metadata,
            embedding=embedding
        )
        
        processed_chunks.append(processed_chunk)
    
    return processed_chunks

async def store_processed_chunks_in_supabase(processed_chunks: List[ProcessedChunk], source_name: str) -> int:
    """
    Store processed chunks in Supabase.
    
    Args:
        processed_chunks: List of processed chunks to store
        source_name: Name of the documentation source
        
    Returns:
        int: Number of chunks stored
    """
    import os
    from supabase import create_client, Client
    
    # Get the Supabase URL and key from environment variables
    supabase_url = get_env_var("SUPABASE_URL")
    supabase_key = get_env_var("SUPABASE_ANON_KEY")
    
    if not supabase_url or not supabase_key:
        raise ValueError("Supabase URL and key must be provided")
    
    # Initialize the Supabase client
    supabase = create_client(supabase_url, supabase_key)
    
    stored_count = 0
    
    # Store each chunk
    for chunk in processed_chunks:
        try:
            # Prepare the data for insertion
            chunk_data = {
                "url": chunk.url,
                "chunk_number": chunk.chunk_number,
                "title": chunk.title,
                "summary": chunk.summary,
                "content": chunk.content,
                "metadata": chunk.metadata,
                "embedding": chunk.embedding,
                "source": source_name
            }
            
            # Insert the chunk into Supabase
            response = supabase.table("docs").insert(chunk_data).execute()
            
            if hasattr(response, 'error') and response.error:
                print(f"Error storing chunk: {response.error}")
            else:
                stored_count += 1
                
        except Exception as e:
            print(f"Error storing chunk: {e}")
    
    return stored_count

async def crawl_with_requests(url: str, tracker: Optional[CrawlProgressTracker] = None, max_retries: int = 3) -> Optional[str]:
    """
    Crawl a single URL using requests library.
    
    Args:
        url: URL to crawl
        tracker: Optional CrawlProgressTracker to track progress
        max_retries: Maximum number of retry attempts
        
    Returns:
        Optional[str]: The cleaned content as markdown, or None if failed
    """
    if tracker:
        tracker.log(f"Crawling with Requests: {url}")
    
    # Implement retry logic
    for attempt in range(1, max_retries + 1):
        try:
            if attempt > 1 and tracker:
                tracker.log(f"Retry attempt {attempt}/{max_retries} for {url}")
            
            # Use requests to fetch the content
            response = requests.get(
                url, 
                timeout=30,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                }
            )
            
            # Check if successful
            if response.status_code != 200:
                if tracker:
                    tracker.log(f"HTTP error {response.status_code} for {url}")
                
                # If this is the last attempt, return None
                if attempt == max_retries:
                    return None
                
                # Wait before retrying
                await asyncio.sleep(2)
                continue
            
            # Convert HTML to markdown
            content = process_html_to_markdown(response.text)
            
            if tracker:
                tracker.log(f"Successfully fetched and converted {url} - {len(content)} characters")
            
            return content
            
        except Exception as e:
            if tracker:
                tracker.log(f"Error crawling {url} with Requests: {str(e)}")
            
            # If this is the last attempt, return None
            if attempt == max_retries:
                return None
            
            # Wait before retrying
            await asyncio.sleep(2)
    
    # This should never be reached due to the return in the loop
    return None

async def crawl_parallel_with_requests(urls: List[str], tracker: Optional[CrawlProgressTracker] = None, max_concurrent: int = 5, source: str = "unknown"):
    """Crawl multiple URLs in parallel using the requests library."""
    # Initialize semaphore for concurrency control
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Process a single URL
    async def process_url(url: str):
        # Skip if we're stopping
        if tracker and tracker.stop_requested:
            return
            
        # Skip if already processed
        if tracker and url in tracker.processed_urls:
            tracker.log(f"[{source}] Skipping already processed URL: {url}")
            return
            
        # Add to processed URLs
        if tracker:
            tracker.processed_urls.add(url)
            
        async with semaphore:
            # Skip if we're stopping
            if tracker and tracker.stop_requested:
                return
                
            try:
                # Crawl the URL
                content = await crawl_with_requests(url, tracker)
                
                if content:
                    # Process and store the document
                    chunks_stored = await process_and_store_document(url, content, tracker, source=source)
                    
                    if tracker:
                        if chunks_stored > 0:
                            tracker.urls_succeeded += 1
                            tracker._status["urls_succeeded"] += 1
                        else:
                            tracker.urls_failed += 1
                            tracker._status["urls_failed"] += 1
                        tracker.urls_processed += 1
                        tracker._status["urls_processed"] += 1
                else:
                    if tracker:
                        tracker.log(f"[{source}] No content retrieved from {url}")
                        tracker.urls_failed += 1
                        tracker._status["urls_failed"] += 1
                        tracker.urls_processed += 1
                        tracker._status["urls_processed"] += 1
            except Exception as e:
                if tracker:
                    tracker.log(f"[{source}] Error processing {url}: {str(e)}")
                    tracker.urls_failed += 1
                    tracker._status["urls_failed"] += 1
                    tracker.urls_processed += 1
                    tracker._status["urls_processed"] += 1
            
            # Update tracker activity
            if tracker:
                tracker.update_activity()
                # Explicitly update progress
                if tracker.progress_callback:
                    tracker.progress_callback(tracker.get_status())
    
    # Create tasks for all URLs
    tasks = []
    for url in urls:
        # Check if we should stop before creating more tasks
        if tracker and tracker.stop_requested:
            tracker.log("Stopping before processing remaining URLs")
            break
        tasks.append(process_url(url))
    
    # Wait for all tasks to complete
    await asyncio.gather(*tasks) 