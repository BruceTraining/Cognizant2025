"""
E-Commerce Customer Service Agent with ReACT Pattern and MCP Integration
This agent demonstrates the ReACT (Reasoning + Acting) pattern where
the agent reasons about what action to take, executes tools, observes
the results, and iterates until the task is complete.

This implementation uses the Model Context Protocol (MCP) to connect
to external services like Stripe for payment processing.
"""

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import HumanMessage, AIMessage
from typing import Dict, List, Optional, Any
import os
import json
import asyncio

# ============================================================================
# REACT PATTERN IMPLEMENTATION
# The agent uses Reasoning, Action, Observation, and Iteration
# ============================================================================

# ============================================================================
# TOOL DEFINITIONS - Traditional Tools (Non-MCP)
# These are simple tools that don't require external MCP servers
# ============================================================================

@tool
def check_order_status(order_id: str) -> str:
    """
    Check the status of a customer's order.
    
    Args:
        order_id: The unique identifier for the order
        
    Returns:
        Current status of the order
    """
    # In a real implementation, this would call your order management
    # system API. For demonstration, we return simulated data.
    
    mock_orders = {
        "ORD123": {
            "status": "shipped",
            "tracking": "1Z999AA10123456784",
            "expected_delivery": "2025-10-22"
        },
        "ORD456": {
            "status": "processing",
            "tracking": None,
            "expected_delivery": "2025-10-25"
        },
        "ORD789": {
            "status": "delivered",
            "tracking": "1Z999AA10123456785",
            "delivered_date": "2025-10-18"
        }
    }
    
    if order_id in mock_orders:
        order = mock_orders[order_id]
        return json.dumps(order, indent=2)
    else:
        return f"Order {order_id} not found in our system."


@tool
def check_return_policy(product_category: str) -> str:
    """
    Get the return policy for a specific product category.
    
    Args:
        product_category: The category of the product (e.g., electronics,
                         clothing, furniture)
        
    Returns:
        Return policy information for that category
    """
    policies = {
        "electronics": """Electronics can be returned within 30 days of 
purchase. Items must be in original packaging with all accessories. 
A 15% restocking fee applies to opened items.""",
        
        "clothing": """Clothing can be returned within 60 days of purchase. 
Items must be unworn with tags attached. No restocking fee.""",
        
        "furniture": """Furniture can be returned within 14 days of delivery. 
Items must be unassembled and in original packaging. Customer pays 
return shipping.""",
        
        "default": """Standard return policy: 30 days from purchase date. 
Items must be in original condition. Contact customer service for 
specific category policies."""
    }
    
    return policies.get(product_category.lower(), policies["default"])


# ============================================================================
# MCP CONFIGURATION
# This section configures the connection to MCP servers
# ============================================================================

async def setup_mcp_client() -> MultiServerMCPClient:
    """
    Set up the MCP client to connect to external services.
    
    This demonstrates how to use the Model Context Protocol to connect
    to standardized tool servers. In this case, we're connecting to
    a Stripe MCP server for payment processing.
    
    Returns:
        Configured MultiServerMCPClient instance
    """
    # Configure the MCP servers we want to connect to
    # Note: In production, you would have actual MCP servers running
    mcp_config = {
        "stripe": {
            # The command to start the Stripe MCP server
            # In production, this would be something like:
            # "command": "npx",
            # "args": ["-y", "@stripe/mcp-server"],
            
            # For demonstration, we'll use a mock server
            "command": "python",
            "args": ["mock_stripe_mcp_server.py"],
            "transport": "stdio"
        }
    }
    
    # Create the MCP client
    # This client handles all the Model Context Protocol communication
    # so we don't have to build our own MCP client
    client = MultiServerMCPClient(mcp_config)
    
    return client


# ============================================================================
# CUSTOMER SERVICE AGENT WITH REACT PATTERN AND MCP
# ============================================================================

class CustomerServiceAgent:
    """
    An autonomous agent using the ReACT pattern with MCP integration.
    
    ReACT stands for:
    - Reasoning: The agent thinks about what to do
    - Action: The agent executes a tool or provides a response
    - Observation: The agent observes the result
    - Iteration: The agent repeats until the task is complete
    
    This agent uses the Model Context Protocol to connect to external
    services like Stripe for payment processing, demonstrating how
    MCP enables standardized tool integration at the orchestration layer.
    """
    
    def __init__(
        self, 
        model_name: str = "gpt-4", 
        temperature: float = 0.3,
        use_mcp: bool = True
    ):
        """
        Initialize the customer service agent with ReACT and MCP support.
        
        Args:
            model_name: The LLM model to use
            temperature: Controls randomness (lower = more deterministic)
            use_mcp: Whether to use MCP for external tool integration
        """
        # Configure the LLM
        self.llm = ChatOpenAI(
            model=model_name, 
            temperature=temperature,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Store configuration
        self.use_mcp = use_mcp
        
        # Traditional tools that don't require MCP
        self.local_tools = [
            check_order_status,
            check_return_policy
        ]
        
        # STUB: Memory implementation will go here
        self.conversation_history: List = []
        
        # MCP client will be initialized asynchronously
        self.mcp_client: Optional[MultiServerMCPClient] = None
        self.mcp_tools: List = []
        
        # Create the system prompt that explains the ReACT pattern
        self.system_prompt = """You are a helpful customer service agent 
for an e-commerce company. You have access to tools that let you help 
customers with their requests.

Your goal is to assist customers by:
1. REASONING about what information or action is needed
2. TAKING ACTION by using the appropriate tools
3. OBSERVING the results from those tools
4. ITERATING until you have fully addressed the customer's needs

When a customer needs to make a payment (for upgrades, expedited 
shipping, additional items, etc.), you should:
1. Clearly explain what they're paying for
2. Confirm the amount
3. Use the payment processing tool to handle the transaction
4. Provide confirmation of the transaction

Always be professional, clear, and helpful. If you need to use a tool, 
explain what you're doing and why."""
    
    async def initialize(self):
        """
        Async initialization to set up MCP connections.
        
        This must be called after creating the agent instance because
        MCP client setup requires async operations.
        """
        if self.use_mcp:
            print("\n[Initializing MCP client...]")
            
            # PRODUCTION NOTE: In a real implementation, you would
            # connect to actual MCP servers here. For example:
            # - Stripe MCP server for payments
            # - Shipping API MCP server for logistics
            # - Inventory MCP server for stock checks
            
            # For demonstration, we'll simulate MCP tools
            # In production, this would be:
            # self.mcp_client = await setup_mcp_client()
            # self.mcp_tools = await self.mcp_client.get_tools()
            
            print("[MCP client ready]")
            print("[Note: Using simulated MCP tools for demonstration]")
            
            # Create a simulated MCP tool that mimics what a real
            # Stripe MCP server would provide
            self.mcp_tools = [self._create_simulated_stripe_tool()]
        
        # Combine local tools with MCP tools
        all_tools = self.local_tools + self.mcp_tools
        
        # Create the agent using LangChain's tool calling pattern
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])
        
        # Build the agent with tool calling capabilities
        self.agent = create_tool_calling_agent(
            self.llm, 
            all_tools, 
            self.prompt
        )
        
        # Create an executor that will run the ReACT loop
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=all_tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5
        )
    
    def _create_simulated_stripe_tool(self):
        """
        Create a simulated Stripe MCP tool for demonstration.
        
        In production, this tool would be provided by the Stripe MCP
        server and loaded via self.mcp_client.get_tools().
        
        This demonstrates what a tool looks like when it comes from
        an MCP server rather than being defined locally.
        """
        @tool
        def process_payment(
            amount: float, 
            currency: str, 
            customer_email: str,
            description: str
        ) -> str:
            """
            Process a payment using Stripe via MCP.
            This tool is provided by the Stripe MCP server and handles
            payment processing through the standardized MCP protocol.
            
            Args:
                amount: Amount to charge in smallest currency unit 
                        (e.g., cents for USD)
                currency: Three-letter currency code (e.g., 'usd')
                customer_email: Customer's email address
                description: Description of what the payment is for
                
            Returns:
                Payment confirmation with transaction details
            """
            # PRODUCTION NOTE: In a real MCP setup, this function body
            # would not exist here. Instead, when the agent calls this
            # tool, the following would happen:
            #
            # 1. LangChain's agent executor invokes the tool
            # 2. The MCP client (MultiServerMCPClient) intercepts
            # 3. MCP client formats the request per MCP protocol
            # 4. Request sent to Stripe MCP server
            # 5. Stripe MCP server calls actual Stripe API
            # 6. Result flows back through MCP protocol
            # 7. MCP client returns structured response to agent
            #
            # This eliminates the need to build custom Stripe API
            # integration code. The MCP server handles all Stripe
            # API specifics, authentication, error handling, etc.
            
            # For demonstration, simulate the payment process
            import random
            transaction_id = (
                f"pi_{''.join(random.choices('0123456789ABCDEF', k=24))}"
            )
            
            result = {
                "status": "success",
                "transaction_id": transaction_id,
                "amount": amount,
                "currency": currency.upper(),
                "customer_email": customer_email,
                "description": description,
                "message": (
                    f"Payment of {amount/100:.2f} {currency.upper()} "
                    f"processed successfully via Stripe MCP"
                ),
                "mcp_server": "stripe",
                "protocol": "Model Context Protocol (MCP)"
            }
            
            return json.dumps(result, indent=2)
        
        return process_payment
    
    async def process_query(
        self, 
        customer_query: str,
        show_reasoning: bool = True
    ) -> str:
        """
        Process a customer query using the ReACT pattern.
        
        The agent will:
        1. Reason about what needs to be done
        2. Act by calling appropriate tools (including MCP tools)
        3. Observe the results
        4. Iterate until the task is complete
        
        Args:
            customer_query: The customer's question or request
            show_reasoning: If True, display the reasoning process
            
        Returns:
            Final response to the customer
        """
        print("\n" + "=" * 70)
        print("PROCESSING CUSTOMER QUERY WITH REACT PATTERN")
        print("=" * 70)
        print(f"\nCustomer Query: {customer_query}\n")
        
        if show_reasoning:
            print("--- Agent Reasoning and Actions ---\n")
        
        # This single invoke() call executes the entire ReACT loop
        # The agent will reason, act, observe, and iterate automatically
        # until it has a final answer or reaches max_iterations
        result = await self.agent_executor.ainvoke({
            "input": customer_query
        })
        
        # Extract the final output
        final_response = result.get("output", "")
        
        # STUB: Store in memory (to be implemented)
        # self._store_in_memory(customer_query, final_response)
        
        return final_response
    
    # ========================================================================
    # STUBS FOR FUTURE IMPLEMENTATION
    # ========================================================================
    
    # STUB: Memory Implementation
    def _store_in_memory(
        self, 
        query: str, 
        response: str
    ) -> None:
        """
        STUB: Will implement persistent memory using vector stores.
        This will allow the agent to remember past conversations and
        provide continuity across interactions.
        """
        pass
    
    # STUB: Feedback Loops
    def _get_human_feedback(self, response: str) -> Dict:
        """
        STUB: Will implement human-in-the-loop feedback.
        This allows humans to review and approve certain actions,
        especially high-stakes decisions like refunds or escalations.
        """
        pass


# ============================================================================
# DEMONSTRATION CODE
# ============================================================================

async def main():
    """
    Demonstrate the ReACT pattern with MCP integration.
    """
    print("=" * 70)
    print("E-COMMERCE CUSTOMER SERVICE AGENT")
    print("WITH REACT PATTERN AND MCP INTEGRATION")
    print("=" * 70)
    
    # Initialize the agent
    agent = CustomerServiceAgent(
        model_name="gpt-4",
        temperature=0.3,
        use_mcp=True
    )
    
    # Important: Must call initialize() because MCP setup is async
    await agent.initialize()
    
    # Example scenarios that demonstrate different aspects of ReACT
    
    # Scenario 1: Simple information lookup (Traditional tool)
    print("\n\n" + "=" * 70)
    print("SCENARIO 1: Order Status Check")
    print("=" * 70)
    
    query1 = "Can you check the status of my order ORD123?"
    response1 = await agent.process_query(query1)
    
    print("\n" + "=" * 70)
    print("FINAL RESPONSE:")
    print("=" * 70)
    print(f"\n{response1}\n")
    
    
    # Scenario 2: Payment processing (MCP tool - Stripe)
    print("\n\n" + "=" * 70)
    print("SCENARIO 2: Payment for Expedited Shipping via MCP")
    print("=" * 70)
    
    query2 = """I need my order ORD456 to arrive faster. Can you upgrade 
me to expedited shipping? I'm willing to pay the $15 extra charge. 
My email is customer@example.com"""
    
    response2 = await agent.process_query(query2)
    
    print("\n" + "=" * 70)
    print("FINAL RESPONSE:")
    print("=" * 70)
    print(f"\n{response2}\n")
    
    
    # Scenario 3: Policy inquiry (Traditional tool)
    print("\n\n" + "=" * 70)
    print("SCENARIO 3: Return Policy Question")
    print("=" * 70)
    
    query3 = "What's your return policy for electronics?"
    response3 = await agent.process_query(query3)
    
    print("\n" + "=" * 70)
    print("FINAL RESPONSE:")
    print("=" * 70)
    print(f"\n{response3}\n")
    
    print("\n" + "=" * 70)
    print("KEY TAKEAWAY:")
    print("=" * 70)
    print("""
Notice how the payment processing used MCP in Scenario 2. The agent
didn't need custom Stripe API code. Instead, it used the standardized
Model Context Protocol to communicate with the Stripe MCP server.

This demonstrates the power of MCP at the orchestration layer:
- The agent doesn't care HOW to call Stripe's API
- The MCP server handles all Stripe-specific details
- The agent just uses a standardized tool interface
- Adding new services (shipping, inventory, etc.) just means
  connecting to more MCP servers - no custom integration code needed

This is why MCP shines when orchestrated by frameworks like LangChain,
rather than building MCP clients inside individual functions.
""")


if __name__ == "__main__":
    asyncio.run(main())
