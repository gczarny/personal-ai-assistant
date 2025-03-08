# src/clients/tavily_search.py
import asyncio
from typing import Optional

from loguru import logger
from tavily import TavilyClient
from tavily.errors import (
    MissingAPIKeyError,
    InvalidAPIKeyError,
    UsageLimitExceededError,
    BadRequestError,
)

from core.exceptions import APIAuthenticationError, APIRateLimitError, APIError
from core.result import SearchResult


class TavilySearchManager:
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.client = None

        if api_key:
            try:
                self.client = TavilyClient(api_key=api_key)
                logger.info("Tavily Search client initialized")
            except MissingAPIKeyError as e:
                logger.error(f"Failed to initialize Tavily client: {e}")
                self.client = None

    async def search(
        self,
        query: str,
        search_depth: str = "basic",
        max_results: int = 5,
        include_domains: Optional[list] = None,
        exclude_domains: Optional[list] = None,
        include_raw_content: bool = False,
        include_images: bool = False,
        include_answer: bool = False,
    ) -> SearchResult:
        """
        Perform a web search using Tavily API.
        Args:
            query: The search query
            search_depth: "basic" or "advanced" (more thorough but slower)
            max_results: max num of results to return
            include_domains: List of domains to include in the search
            exclude_domains: List of domains to exclude from the search
            include_answer: if to include a generated answer
            include_raw_content: if to include the raw content of the results
            include_images: if to include images in the results
        Returns:
            SearchResult object with the results or error details
        """
        if not self.client:
            error = APIError("Tavily Search client is not initialized")
            logger.error(str(error))
            return SearchResult.fail(error=error)

        try:
            logger.info(f"Performing Tavily search: '{query}'")

            search_params = {
                "query": query,
                "search_depth": search_depth,
                "max_results": max_results,
                "include_answer": include_answer,
                "include_raw_content": include_raw_content,
                "include_images": include_images,
            }

            # Add optional parameters if provided
            if include_domains:
                search_params["include_domains"] = include_domains
            if exclude_domains:
                search_params["exclude_domains"] = exclude_domains

            result = await asyncio.to_thread(self.client.search, **search_params)

            metadata = {
                "search_id": result.get("search_id", ""),
                "search_depth": search_depth,
                "max_results": max_results,
                "result_count": len(result.get("results", [])),
                "response_time": result.get("response_time", 0),
            }

            logger.info(f"Received {metadata['result_count']} results from Tavily API")

            return SearchResult.ok(result, metadata=metadata)

        except InvalidAPIKeyError as e:
            error = APIAuthenticationError(
                f"Authentication failed with Tavily API: {str(e)}"
            )
            logger.error(f"Authentication error: {str(e)}")
            return SearchResult.fail(error=error)

        except UsageLimitExceededError as e:
            error = APIRateLimitError(f"Rate limit exceeded with Tavily API: {str(e)}")
            logger.error(f"Rate limit error: {str(e)}")
            return SearchResult.fail(error=error)

        except BadRequestError as e:
            error = APIError(f"Bad request error with Tavily API: {str(e)}")
            logger.error(f"Bad request error: {str(e)}")
            return SearchResult.fail(error=error)

        except Exception as e:
            error = APIError(f"Unexpected error during search: {str(e)}")
            logger.error(f"Unexpected error: {str(e)}")
            return SearchResult.fail(error=error)
