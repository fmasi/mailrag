"""
Query engine module for the Email RAG system.

This module provides the interface for querying the indexed emails.
It handles: 
1. Creating query engines from indexes
2. Performing various types of queries (similarity search, RAG, etc.)
3. Formatting and presenting results

This separation keeps query logic decoupled from indexing and storage.
"""

from typing import List, Optional

from llama_index.core import VectorStoreIndex
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.core.schema import NodeWithScore


class EmailQueryEngine:
    """
    Provides query capabilities for the email index.
    
    This class wraps LlamaIndex query functionality and provides
    domain-specific query methods for the email RAG system.
    """
    
    def __init__(self, index: VectorStoreIndex):
        """
        Initialize the query engine with an index.
        
        Args:
            index: VectorStoreIndex to query against
        """
        self.index = index
        
        # Create a standard query engine
        # This uses the configured LLM and embedding model from Settings
        self.query_engine: BaseQueryEngine = index.as_query_engine(
            similarity_top_k=5,  # Return top 5 most relevant documents
        )
    
    def query(self, query_text: str) -> str:
        """
        Perform a RAG query on the indexed emails.
        
        Args:
            query_text: The question/query to ask
        
        Returns:
            The LLM's response, which includes context from relevant emails
            
        Why this approach:
            - RAG (Retrieval Augmented Generation) retrieves relevant emails first
            - Then passes them as context to the LLM
            - The LLM generates an answer based on the context
            - This is better than pure LLM because it uses your actual data
        """
        response = self.query_engine.query(query_text)
        return str(response)
    
    def retrieval_query(
        self, query_text: str, top_k: int = 5
    ) -> List[dict]:
        """
        Perform a pure retrieval query (no LLM, just similarity search).
        
        Args:
            query_text: The query to search for
            top_k: Number of top results to return
        
        Returns:
            List of dictionaries containing:
            - text: The email content
            - metadata: Email metadata (sender, subject, date)
            - score: Similarity score (0-1, higher is more relevant)
            
        Why this method:
            - Sometimes you want just the most relevant emails without LLM generation
            - This is faster and cheaper (no LLM call)
            - Useful for understanding what documents are being retrieved
            - Good for debugging the retrieval part of RAG
        """
        # Use the index retriever to get similar documents
        retriever = self.index.as_retriever(similarity_top_k=top_k)
        results = retriever.retrieve(query_text)
        
        # Format results into a more readable structure
        formatted_results = []
        
        for result in results:
            formatted_results.append({
                "text": result.text[:500],  # First 500 chars of email
                "metadata": result.metadata,
                "score": result.score if hasattr(result, 'score') else None,
            })
        
        return formatted_results
    
    def query_with_metadata_filter(
        self, query_text: str, sender: Optional[str] = None, top_k: int = 5
    ) -> List[dict]:
        """
        Query emails with optional metadata filtering.
        
        Args:
            query_text: The query/question
            sender: Optional: filter results by sender email address
            top_k: Number of results to return
        
        Returns:
            List of relevant emails matching the query and filters
            
        Why this method:
            - Metadata (sender, subject, date) enables powerful filtering
            - You might want "emails from john@example.com about vacation policies"
            - This combines retrieval with business logic
            - Shows the value of extracting metadata during document creation
            
        Note:
            Currently implements basic sender filtering.
            In the future, could add date range filtering, subject keywords, etc.
        """
        # Get retrieval results
        results = self.retrieval_query(query_text, top_k=top_k * 2)  # Get more to filter
        
        # Apply sender filter if specified
        if sender:
            results = [
                r for r in results
                if sender.lower() in r["metadata"].get("sender", "").lower()
            ]
        
        # Return top k after filtering
        return results[:top_k]
    
    def query_by_sender(self, sender: str, query_text: str = None) -> List[dict]:
        """
        Find emails from a specific sender.
        
        Args:
            sender: Email address or name to search for
            query_text: Optional additional query filter
        
        Returns:
            List of emails from the specified sender
            
        Why this method:
            - Shows how metadata makes domain-specific queries possible
            - Useful for finding conversations with specific people
            - Demonstrates metadata value from Task 3
        """
        if query_text:
            return self.query_with_metadata_filter(query_text, sender=sender)
        else:
            # If no query, return emails from sender (need retripper that supports filtering)
            # For now, return empty with note about limitation
            print(f"Note: Basic sender-only query not yet supported.")
            print(f"Use query_with_metadata_filter(query_text, sender='{sender}') instead.")
            return []
    
    def print_query_results(self, results: List[dict], title: str = "Query Results") -> None:
        """
        Pretty-print query results.
        
        Args:
            results: List of result dictionaries from retrieval_query
            title: Title for the output
            
        Why this method:
            - Makes results human-readable
            - Shows how to access and display metadata
            - Useful for debugging and exploration
        """
        print(f"\n{'=' * 60}")
        print(f"{title}")
        print(f"{'=' * 60}")
        
        for i, result in enumerate(results, 1):
            print(f"\n[Result {i}]")
            print(f"  From: {result['metadata'].get('sender', 'Unknown')}")
            print(f"  Subject: {result['metadata'].get('subject', 'No subject')}")
            print(f"  Date: {result['metadata'].get('date', 'Unknown')}")
            
            if result.get('score'):
                print(f"  Relevance Score: {result['score']:.3f}")
            
            print(f"  Content (preview):")
            print(f"    {result['text'][:300]}...")
        
        print(f"\n{'=' * 60}\n")
