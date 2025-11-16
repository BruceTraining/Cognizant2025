"""
Simple LangChain Agent with Official Stripe MCP Server

This example demonstrates how to connect to Stripe's official MCP server
using LangChain's built-in MCP client capabilities.

Prerequisites:
1. Install required packages:
   pip install langchain-openai langchain-mcp-adapters langgraph

2. Set environment variables:
   export OPENAI_API_KEY="your-openai-key"
   export STRIPE_SECRET_KEY="your-stripe-secret-key"

3. Ensure you have npx installed (comes with Node.js)

Key Learning Points:
- LangChain provides MultiServerMCPClient (no need to build one)
- Stripe hosts both remote (https://mcp.stripe.com) and local MCP servers
- Tools are automatically discovered from the MCP server
- The agent can use these tools seamlessly in its reasoning
"""

import asyncio
import os
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent


async def main():
    """
    Create an agent that uses the official Stripe MCP server.
    """
    
    print("=" * 70)
    print("LANGCHAIN AGENT WITH OFFICIAL STRIPE MCP SERVER")
    print("=" * 70)
    
    # ====================================================================
    # STEP 1: Configure the LLM
    # ====================================================================
    print("\n[Step 1] Initializing language model...")
    
    llm = ChatOpenAI(
        model="gpt-4",
        temperature=0.3,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    print("✓ Language model ready")
    
    
    # ====================================================================
    # STEP 2: Configure the Stripe MCP Server
    # ====================================================================
    print("\n[Step 2] Configuring Stripe MCP server...")
    
    # Get Stripe API key from environment
    stripe_key = os.getenv("STRIPE_SECRET_KEY")
    
    if not stripe_key:
        print("\n⚠ WARNING: STRIPE_SECRET_KEY not set in environment")
        print("Please run: export STRIPE_SECRET_KEY='your-key-here'")
        print("\nYou can get a test key from:")
        print("https://dashboard.stripe.com/test/apikeys")
        return
    
    # Configure the official Stripe MCP server
    # Stripe provides two options:
    # 1. Remote server at https://mcp.stripe.com (requires OAuth)
    # 2. Local server via npx (what we're using here)
    
    mcp_config = {
        "stripe": {
            # Use the official Stripe MCP server via npx
            "command": "npx",
            "args": [
                "-y",                    # Automatically install if needed
                "@stripe/mcp",           # Official Stripe MCP package
                "--tools=all"            # Enable all available tools
            ],
            "env": {
                # Pass the Stripe API key to the MCP server
                "STRIPE_SECRET_KEY": stripe_key
            },
            "transport": "stdio"         # Communication via standard I/O
        }
    }
    
    print("✓ MCP configuration ready")
    print(f"  Using local Stripe MCP server (@stripe/mcp)")
    print(f"  Tools: all")
    
    
    # ====================================================================
    # STEP 3: Create the MCP Client
    # ====================================================================
    print("\n[Step 3] Connecting to Stripe MCP server...")
    print("  (This may take a moment while npx downloads the package)")
    
    # The MultiServerMCPClient handles all MCP protocol communication
    # You don't need to understand the protocol details
    async with MultiServerMCPClient(mcp_config) as mcp_client:
        print("✓ Connected to Stripe MCP server")
        
        
        # ================================================================
        # STEP 4: Load Tools from Stripe MCP Server
        # ================================================================
        print("\n[Step 4] Loading tools from Stripe MCP server...")
        
        # The MCP server exposes tools through the standardized protocol
        # get_tools() automatically discovers and converts them
        tools = await mcp_client.get_tools()
        
        print(f"✓ Loaded {len(tools)} tools from Stripe")
        print("\nAvailable Stripe tools:")
        
        # Display some of the available tools
        for i, tool in enumerate(tools[:10]):
            print(f"  - {tool.name}")
        
        if len(tools) > 10:
            print(f"  ... and {len(tools) - 10} more tools")
        
        
        # ================================================================
        # STEP 5: Create the ReACT Agent
        # ================================================================
        print("\n[Step 5] Creating ReACT agent with Stripe tools...")
        
        # Create an agent that can use the Stripe MCP tools
        agent = create_react_agent(llm, tools)
        
        print("✓ Agent ready")
        
        
        # ================================================================
        # STEP 6: Use the Agent for Payment Processing
        # ================================================================
        print("\n" + "=" * 70)
        print("EXAMPLE 1: Create a customer and process payment")
        print("=" * 70)
        
        request1 = """
        I need you to:
        1. Create a new Stripe customer with email "demo@example.com" 
           and name "Demo Customer"
        2. Then show me the customer details
        """
        
        print(f"\nRequest: {request1}")
        print("\n--- Agent Processing ---")
        
        response1 = await agent.ainvoke({
            "messages": [{"role": "user", "content": request1}]
        })
        
        final_message1 = response1["messages"][-1].content
        
        print("\n" + "=" * 70)
        print("AGENT RESPONSE:")
        print("=" * 70)
        print(f"\n{final_message1}\n")
        
        
        # ================================================================
        # EXAMPLE 2: List Products
        # ================================================================
        print("\n" + "=" * 70)
        print("EXAMPLE 2: Query Stripe products")
        print("=" * 70)
        
        request2 = "Can you list my Stripe products?"
        
        print(f"\nRequest: {request2}")
        print("\n--- Agent Processing ---")
        
        response2 = await agent.ainvoke({
            "messages": [{"role": "user", "content": request2}]
        })
        
        final_message2 = response2["messages"][-1].content
        
        print("\n" + "=" * 70)
        print("AGENT RESPONSE:")
        print("=" * 70)
        print(f"\n{final_message2}\n")
        
        
        # ================================================================
        # EXAMPLE 3: Search Stripe Documentation
        # ================================================================
        print("\n" + "=" * 70)
        print("EXAMPLE 3: Search Stripe knowledge base")
        print("=" * 70)
        
        request3 = """
        Search the Stripe documentation to find information about 
        payment intents and how they work.
        """
        
        print(f"\nRequest: {request3}")
        print("\n--- Agent Processing ---")
        
        response3 = await agent.ainvoke({
            "messages": [{"role": "user", "content": request3}]
        })
        
        final_message3 = response3["messages"][-1].content
        
        print("\n" + "=" * 70)
        print("AGENT RESPONSE:")
        print("=" * 70)
        print(f"\n{final_message3}\n")
    
    
    # ====================================================================
    # Summary
    # ====================================================================
    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS:")
    print("=" * 70)
    print("""
1. OFFICIAL STRIPE MCP SERVER
   - Stripe provides @stripe/mcp package via npm
   - No need to build or maintain your own MCP server
   - Automatically stays up to date with Stripe API changes
   
2. TWO CONNECTION OPTIONS
   - Remote: https://mcp.stripe.com (requires OAuth setup)
   - Local: npx @stripe/mcp (what we used here)
   
3. COMPREHENSIVE TOOL SET
   The Stripe MCP server provides tools for:
   - Customer management (create, list customers)
   - Product and pricing (create products, prices)
   - Payment processing (payment intents, payment links)
   - Invoicing (create, finalize invoices)
   - Subscriptions (create, cancel, update)
   - Refunds (create refunds)
   - Documentation search (search Stripe knowledge base)
   
4. NO CUSTOM INTEGRATION CODE
   - LangChain's MultiServerMCPClient handles the protocol
   - Tools are automatically discovered and converted
   - The agent uses them like any other LangChain tool
   
5. PRODUCTION CONSIDERATIONS
   - Use restricted API keys for security
   - Enable human confirmation for sensitive operations
   - Consider using the remote MCP server for production
   - Monitor tool usage and set appropriate rate limits
   
NEXT STEPS:
- Get your Stripe API keys: https://dashboard.stripe.com/apikeys
- Explore all available tools in the Stripe MCP documentation
- Learn about OAuth setup for the remote MCP server
- Review Stripe's MCP security best practices
""")
    
    print("\n" + "=" * 70)
    print("PRODUCTION ALTERNATIVE: Remote MCP Server")
    print("=" * 70)
    print("""
For production use, you can use Stripe's hosted remote MCP server
instead of running it locally. This eliminates the need for npx:

mcp_config = {
    "stripe": {
        "url": "https://mcp.stripe.com",
        "transport": "http"
    }
}

This requires OAuth setup but provides better security and 
doesn't require Node.js to be installed on your server.

See: https://docs.stripe.com/mcp for complete documentation
""")


if __name__ == "__main__":
    # Check for required environment variables
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set")
        print("Please run: export OPENAI_API_KEY='your-key-here'")
    else:
        # Run the async main function
        asyncio.run(main())
