import os
import logging
import json
import requests
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from environment.agents.base import BaseTool


class WebSearcher(BaseTool):
    """
    Web search agent that retrieves current information from the web using You.com search API.
    Useful for finding current trends, reference materials, and contextual information for video content creation.
    """

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        
        # Configure You.com API settings
        self.api_key = os.getenv('YOUCOM_API_KEY')
        self.base_url = os.getenv('YOUCOM_BASE_URL', 'https://api.you.com')
        self.search_enabled = os.getenv('YOUCOM_SEARCH_ENABLED', 'false').lower() == 'true'
        
        # Validate configuration
        if self.search_enabled and not self.api_key:
            self.logger.warning("YOUCOM_SEARCH_ENABLED is true but YOUCOM_API_KEY not set. Web search will be disabled.")
            self.search_enabled = False

    class InputSchema(BaseTool.BaseInputSchema):
        query: str = Field(
            ...,
            description="Search query for finding relevant web information"
        )
        search_type: str = Field(
            default="web",
            description="Type of search: 'web' for general web search, 'news' for news search"
        )
        max_results: int = Field(
            default=5,
            description="Maximum number of search results to return (1-10)"
        )

    class OutputSchema(BaseModel):
        status: str = Field(
            ...,
            description="Execution status (success/error/disabled)"
        )
        results: List[Dict[str, Any]] = Field(
            default=[],
            description="List of search results with title, url, and snippet"
        )
        query_used: str = Field(
            default="",
            description="The search query that was executed"
        )
        error_message: Optional[str] = Field(
            default=None,
            description="Error message if search failed"
        )

    def _validate_inputs(self, query: str, search_type: str, max_results: int) -> bool:
        """Validate input parameters"""
        if not query or not query.strip():
            self.logger.error("Search query cannot be empty")
            return False
            
        if search_type not in ['web', 'news']:
            self.logger.error(f"Invalid search_type: {search_type}. Must be 'web' or 'news'")
            return False
            
        if not (1 <= max_results <= 10):
            self.logger.error(f"max_results must be between 1 and 10, got {max_results}")
            return False
            
        return True

    def _make_search_request(self, query: str, search_type: str, max_results: int) -> Dict[str, Any]:
        """Make the actual API request to You.com"""
        try:
            # Choose endpoint based on search type
            endpoint = '/search' if search_type == 'web' else '/news'
            url = f"{self.base_url}{endpoint}"
            
            headers = {
                'X-API-Key': self.api_key,
                'Content-Type': 'application/json'
            }
            
            payload = {
                'query': query,
                'count': max_results
            }
            
            self.logger.info(f"Making {search_type} search request: {query}")
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 401:
                raise Exception("Authentication failed. Please check your YOUCOM_API_KEY.")
            elif response.status_code == 429:
                raise Exception("Rate limit exceeded. Please try again later.")
            else:
                raise Exception(f"API request failed with status {response.status_code}: {response.text}")
                
        except requests.exceptions.Timeout:
            raise Exception("Search request timed out. Please try again.")
        except requests.exceptions.ConnectionError:
            raise Exception("Unable to connect to You.com API. Please check your internet connection.")
        except Exception as e:
            raise Exception(f"Search request failed: {str(e)}")

    def _format_results(self, api_response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Format API response into standardized result format"""
        formatted_results = []
        
        # Handle different possible response formats
        hits = api_response.get('hits', [])
        if not hits and 'results' in api_response:
            hits = api_response['results']
        
        for hit in hits:
            result = {
                'title': hit.get('title', 'No title'),
                'url': hit.get('url', ''),
                'snippet': hit.get('snippet', hit.get('description', 'No description available'))
            }
            formatted_results.append(result)
            
        return formatted_results

    def run(self, query: str, search_type: str = "web", max_results: int = 5) -> OutputSchema:
        """
        Execute web search and return formatted results
        
        Args:
            query: Search query string
            search_type: Type of search ('web' or 'news')  
            max_results: Maximum number of results to return (1-10)
            
        Returns:
            OutputSchema with search results or error information
        """
        # Check if web search is enabled
        if not self.search_enabled:
            self.logger.info("Web search is disabled. Set YOUCOM_SEARCH_ENABLED=true to enable.")
            return self.OutputSchema(
                status="disabled",
                results=[],
                query_used=query,
                error_message="Web search is disabled. Configure YOUCOM_API_KEY and set YOUCOM_SEARCH_ENABLED=true to enable."
            )
        
        # Validate inputs
        if not self._validate_inputs(query, search_type, max_results):
            return self.OutputSchema(
                status="error",
                results=[],
                query_used=query,
                error_message="Invalid input parameters"
            )
        
        try:
            # Make search request
            api_response = self._make_search_request(query, search_type, max_results)
            
            # Format results
            formatted_results = self._format_results(api_response)
            
            self.logger.info(f"Successfully retrieved {len(formatted_results)} search results")
            
            return self.OutputSchema(
                status="success",
                results=formatted_results,
                query_used=query
            )
            
        except Exception as e:
            self.logger.error(f"Web search failed: {str(e)}")
            return self.OutputSchema(
                status="error",
                results=[],
                query_used=query,
                error_message=str(e)
            )