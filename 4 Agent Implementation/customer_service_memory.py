"""
E-Commerce Customer Service Agent with Memory Strategies
This agent demonstrates different approaches to managing conversation
memory in agentic systems.
"""

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationSummaryBufferMemory,
    VectorStoreRetrieverMemory
)
from langchain.chains import ConversationChain
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
import os

# =======================================================================
# MEMORY MECHANISMS IN AGENT ARCHITECTURES
# Implementing persistent, contextual memory using various strategies
# =======================================================================

class CustomerServiceAgent:
    """
    A customer service agent demonstrating different memory strategies.
    
    This class shows three different approaches to handling memory:
    1. Buffer Window Memory: Short-term with fixed window
    2. Summary Buffer Memory: Pruning via summarization
    3. Vector Store Memory: Long-term semantic retrieval
    """
    
    def __init__(
        self, 
        model_name: str = "gpt-4", 
        temperature: float = 0.3
    ):
        """
        Initialize the customer service agent with multiple memory options.
        
        Args:
            model_name: The LLM model to use
            temperature: Controls randomness (lower = more deterministic)
        """
        self.llm = ChatOpenAI(
            model=model_name, 
            temperature=temperature,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # System prompt defines the agent's role and behavior
        self.system_prompt = """You are a helpful customer service agent 
for an e-commerce company. Your role is to assist customers with their 
inquiries about orders, products, returns, and general support. 

Always be professional, empathetic, and solution-oriented. Use any 
relevant conversation history to provide personalized responses."""
        
        # Initialize different memory strategies
        self._setup_memory_strategies()
    
    def _setup_memory_strategies(self):
        """
        Set up three different memory strategies to demonstrate trade-offs
        between short-term and long-term memory approaches.
        """
        
        # ---------------------------------------------------------------
        # STRATEGY 1: BUFFER WINDOW MEMORY (Short-term, Fixed Window)
        # Keeps only the last N conversation turns
        # Trade-off: Simple but loses older context
        # ---------------------------------------------------------------
        self.window_memory = ConversationBufferWindowMemory(
            k=3,  # Keep last 3 exchanges
            memory_key="history",
            return_messages=True
        )
        
        # ---------------------------------------------------------------
        # STRATEGY 2: SUMMARY BUFFER MEMORY (Automatic Pruning)
        # Keeps recent messages and summarizes older ones
        # Trade-off: Preserves context but summaries lose detail
        # ---------------------------------------------------------------
        self.summary_memory = ConversationSummaryBufferMemory(
            llm=self.llm,
            max_token_limit=500,  # Summarize when exceeding this limit
            memory_key="history",
            return_messages=True
        )
        
        # ---------------------------------------------------------------
        # STRATEGY 3: VECTOR STORE MEMORY (Long-term Semantic Retrieval)
        # Stores all conversations and retrieves relevant ones by meaning
        # Trade-off: Best for long-term recall but requires vector DB
        # ---------------------------------------------------------------
        
        # Create embeddings model for semantic search
        embeddings = OpenAIEmbedings(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Initialize FAISS vector store (in-memory vector database)
        # FAISS is a library for efficient similarity search of vectors
        # Note: In production, you would persist this to disk or use
        # a cloud-based vector store like Pinecone or Chroma
        initial_docs = [
            Document(
                page_content="Customer service conversation history",
                metadata={"type": "initialization"}
            )
        ]
        vector_store = FAISS.from_documents(initial_docs, embeddings)
        
        # Create retriever that finds relevant past conversations
        retriever = vector_store.as_retriever(
            search_kwargs={"k": 2}  # Retrieve 2 most relevant memories
        )
        
        self.vector_memory = VectorStoreRetrieverMemory(
            retriever=retriever,
            memory_key="history",
            return_messages=False  # Returns strings, not message objects
        )
        
    def create_chain_with_memory(self, memory_type: str):
        """
        Create a conversation chain with specified memory strategy.
        
        Args:
            memory_type: One of "window", "summary", "vector"
            
        Returns:
            ConversationChain configured with the chosen memory strategy
        """
        # Select the appropriate memory based on strategy
        memory_map = {
            "window": self.window_memory,
            "summary": self.summary_memory,
            "vector": self.vector_memory
        }
        
        memory = memory_map.get(memory_type)
        if not memory:
            raise ValueError(
                f"Unknown memory type: {memory_type}. "
                f"Choose from: {list(memory_map.keys())}"
            )
        
        # Create prompt template that includes memory
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])
        
        # Build the conversation chain
        chain = ConversationChain(
            llm=self.llm,
            memory=memory,
            prompt=prompt,
            verbose=False
        )
        
        return chain
    
    def demonstrate_memory_strategy(
        self, 
        memory_type: str, 
        queries: list
    ):
        """
        Demonstrate a specific memory strategy with a series of queries.
        
        Args:
            memory_type: The memory strategy to demonstrate
            queries: List of customer queries to process
        """
        print(f"\n{'=' * 70}")
        print(f"DEMONSTRATING: {memory_type.upper()} MEMORY STRATEGY")
        print(f"{'=' * 70}\n")
        
        # Create chain with chosen memory strategy
        chain = self.create_chain_with_memory(memory_type)
        
        # Process each query and show the response
        for i, query in enumerate(queries, 1):
            print(f"[Query {i}] Customer: {query}")
            response = chain.predict(input=query)
            print(f"[Response {i}] Agent: {response}\n")
            print("-" * 70 + "\n")
        
        # Show what's in memory after all queries
        self._display_memory_contents(memory_type)
    
    def _display_memory_contents(self, memory_type: str):
        """
        Display the current contents of the specified memory.
        This helps students understand what each strategy retains.
        
        Args:
            memory_type: The memory strategy to inspect
        """
        print(f"[MEMORY CONTENTS - {memory_type.upper()}]\n")
        
        memory_map = {
            "window": self.window_memory,
            "summary": self.summary_memory,
            "vector": self.vector_memory
        }
        
        memory = memory_map[memory_type]
        
        try:
            memory_vars = memory.load_memory_variables({})
            print(memory_vars)
        except Exception as e:
            print(f"Unable to display memory: {e}")
        
        print("\n" + "=" * 70 + "\n")


# =======================================================================
# DEMONSTRATION CODE
# =======================================================================

def main():
    """
    Demonstrate all three memory strategies with the same conversation.
    This shows how each strategy handles memory differently.
    """
    print("=" * 70)
    print("CUSTOMER SERVICE AGENT: MEMORY STRATEGIES COMPARISON")
    print("Implementing Memory Mechanisms in Agent Architectures")
    print("=" * 70)
    
    # Initialize the agent
    agent = CustomerServiceAgent(
        model_name="gpt-4",
        temperature=0.3
    )
    
    # Create a conversation scenario that tests memory capabilities
    # This sequence will reveal differences in memory strategies
    conversation_queries = [
        "Hi, I'm John Smith and I ordered a laptop last week.",
        "The order number is #12345. Can you check its status?",
        "Great! Also, what's your return policy?",
        "One more thing, do you offer extended warranties?",
        "Going back to my laptop order, when will it arrive?",
        "Thanks! By the way, what was my order number again?"
    ]
    
    print("\nThis demonstration will show the same conversation using")
    print("three different memory strategies. Notice how each strategy")
    print("handles context differently.\n")
    
    # ===================================================================
    # DEMONSTRATION 1: Window Memory
    # Short-term memory with fixed window
    # ===================================================================
    print("\n" + "=" * 70)
    print("STRATEGY 1: BUFFER WINDOW MEMORY")
    print("Keeps only the last 3 conversation exchanges")
    print("Best for: Simple conversations, limited context needs")
    print("Limitation: Forgets older information (like order number)")
    print("=" * 70)
    input("\nPress Enter to see this strategy in action...")
    
    agent.demonstrate_memory_strategy("window", conversation_queries)
    
    # ===================================================================
    # DEMONSTRATION 2: Summary Memory
    # Memory pruning through summarization
    # ===================================================================
    print("\n" + "=" * 70)
    print("STRATEGY 2: SUMMARY BUFFER MEMORY")
    print("Keeps recent messages and summarizes older ones")
    print("Best for: Longer conversations with budget constraints")
    print("Limitation: Summaries may lose specific details")
    print("=" * 70)
    input("\nPress Enter to see this strategy in action...")
    
    # Reset the agent to clear previous memory
    agent = CustomerServiceAgent(model_name="gpt-4", temperature=0.3)
    agent.demonstrate_memory_strategy("summary", conversation_queries)
    
    # ===================================================================
    # DEMONSTRATION 3: Vector Store Memory
    # Long-term memory with semantic retrieval
    # ===================================================================
    print("\n" + "=" * 70)
    print("STRATEGY 3: VECTOR STORE MEMORY")
    print("Stores all conversations and retrieves by semantic similarity")
    print("Best for: Long-term context, finding relevant past information")
    print("Limitation: Requires vector database infrastructure")
    print("=" * 70)
    input("\nPress Enter to see this strategy in action...")
    
    # Reset the agent
    agent = CustomerServiceAgent(model_name="gpt-4", temperature=0.3)
    agent.demonstrate_memory_strategy("vector", conversation_queries)
    
    # ===================================================================
    # SUMMARY
    # ===================================================================
    print("\n" + "=" * 70)
    print("SUMMARY: CHOOSING THE RIGHT MEMORY STRATEGY")
    print("=" * 70)
    print("""
When selecting a memory strategy, consider:

1. BUFFER WINDOW MEMORY
   Use when: Simple, short conversations
   Token cost: Low (fixed window size)
   Context retention: Limited to recent exchanges
   
2. SUMMARY BUFFER MEMORY
   Use when: Longer conversations with budget limits
   Token cost: Medium (summarization has cost)
   Context retention: Good for general context, may lose specifics
   
3. VECTOR STORE MEMORY
   Use when: Need to recall specific past information
   Token cost: Medium (retrieval queries)
   Context retention: Excellent for long-term semantic recall

Key Takeaway: Different pruning and retention strategies have different 
trade-offs between short-term and long-term memory. Choose based on your 
use case requirements for memory persistence and relevance.
    """)


if __name__ == "__main__":
    main()
