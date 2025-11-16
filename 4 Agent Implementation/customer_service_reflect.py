"""
E-Commerce Customer Service Agent with Reflexion Pattern
This agent demonstrates autonomous reasoning with self-reflection
to improve response quality and correctness.
"""

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from typing import Dict, List, Optional
import os

# ============================================================================
# REFLEXION PATTERN IMPLEMENTATION
# The agent looks back at its output to evaluate correctness
# ============================================================================

class CustomerServiceAgent:
    """
    An autonomous agent that uses reflexion to improve response quality.
    The agent generates an initial response, reflects on it, and then
    provides an improved final response.
    """
    
    def __init__(
        self, 
        model_name: str = "gpt-4", 
        temperature: float = 0.3
    ):
        """
        Initialize the customer service agent.
        
        Args:
            model_name: The LLM model to use
            temperature: Controls randomness (lower = more deterministic)
        """
        # Configure the LLM with appropriate temperature to reduce
        # hallucinations as per section 2.5.1
        self.llm = ChatOpenAI(
            model=model_name, 
            temperature=temperature,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # STUB: Section 4.3 - Memory implementation will go here
        self.conversation_history: List = []
        
        # STUB: Section 4.2 - Tool definitions will go here
        self.tools = []
        
        # Define the customer service persona and task as per 2.7.1.1
        self.system_prompt = """You are a helpful customer service agent 
for an e-commerce company. Your role is to assist customers with their 
inquiries about orders, products, returns, and general support. 

Always be professional, empathetic, and solution-oriented. If you are 
unsure about something, acknowledge it honestly rather than making up 
information."""
        
    def _generate_initial_response(
        self, 
        customer_query: str
    ) -> str:
        """
        Generate the initial response to the customer query.
        This is the first step in the Reflexion pattern.
        
        Args:
            customer_query: The customer's question or issue
            
        Returns:
            Initial response from the agent
        """
        # Create prompt with clear task definition (section 2.7.1.2)
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "{query}")
        ])
        
        # Generate initial response
        chain = prompt | self.llm
        response = chain.invoke({"query": customer_query})
        
        return response.content
    
    def _reflect_on_response(
        self, 
        customer_query: str, 
        initial_response: str
    ) -> Dict[str, any]:
        """
        Reflect on the initial response to evaluate its quality.
        This implements the self-reflection mechanism from section 4.1.1.
        
        The agent analyzes:
        - Accuracy and completeness of the response
        - Tone and professionalism
        - Whether any improvements are needed
        
        Args:
            customer_query: Original customer question
            initial_response: The agent's initial response
            
        Returns:
            Dictionary with reflection results and suggestions
        """
        # Create a reflection prompt that asks the LLM to show its
        # thinking as per section 2.7.1.2.3
        reflection_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a quality control expert reviewing 
customer service responses. Analyze the response critically and 
provide feedback on:

1. Accuracy: Is the information correct?
2. Completeness: Does it fully address the customer's question?
3. Tone: Is it appropriately professional and empathetic?
4. Clarity: Is it easy to understand?
5. Improvements: What specific changes would make it better?

Provide your analysis in a structured way, then suggest specific 
improvements if needed."""),
            ("human", """Customer Query: {query}

Agent Response: {response}

Please analyze this response and provide your feedback.""")
        ])
        
        # Get reflection from the LLM
        chain = reflection_prompt | self.llm
        reflection = chain.invoke({
            "query": customer_query,
            "response": initial_response
        })
        
        return {
            "feedback": reflection.content,
            "needs_improvement": "improvement" in reflection.content.lower()
        }
    
    def _generate_improved_response(
        self, 
        customer_query: str,
        initial_response: str,
        reflection: Dict[str, any]
    ) -> str:
        """
        Generate an improved response based on reflection feedback.
        This completes the Reflexion loop.
        
        Args:
            customer_query: Original customer question
            initial_response: The initial response
            reflection: Feedback from reflection step
            
        Returns:
            Improved final response
        """
        improvement_prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", """Customer Query: {query}

Initial Response: {initial_response}

Reflection Feedback: {feedback}

Based on this feedback, provide an improved response that addresses 
any identified issues while maintaining the helpful aspects of the 
original response.""")
        ])
        
        chain = improvement_prompt | self.llm
        improved = chain.invoke({
            "query": customer_query,
            "initial_response": initial_response,
            "feedback": reflection["feedback"]
        })
        
        return improved.content
    
    def process_query(
        self, 
        customer_query: str, 
        show_thinking: bool = False
    ) -> str:
        """
        Process a customer query using the Reflexion pattern.
        
        This method implements the complete reflection loop:
        1. Generate initial response
        2. Reflect on the response quality
        3. Generate improved response if needed
        
        Args:
            customer_query: The customer's question or issue
            show_thinking: If True, display the reflection process
            
        Returns:
            Final response to the customer
        """
        # Step 1: Generate initial response
        print("\n[Generating initial response...]")
        initial_response = self._generate_initial_response(customer_query)
        
        if show_thinking:
            print(f"\nInitial Response:\n{initial_response}\n")
            print("-" * 60)
        
        # Step 2: Reflect on the response (section 4.1.1)
        print("\n[Reflecting on response quality...]")
        reflection = self._reflect_on_response(
            customer_query, 
            initial_response
        )
        
        if show_thinking:
            print(f"\nReflection:\n{reflection['feedback']}\n")
            print("-" * 60)
        
        # Step 3: Generate improved response if needed
        if reflection["needs_improvement"]:
            print("\n[Generating improved response...]")
            final_response = self._generate_improved_response(
                customer_query,
                initial_response,
                reflection
            )
        else:
            print("\n[Initial response deemed sufficient]")
            final_response = initial_response
        
        # STUB: Section 4.3 - Store in memory (to be implemented)
        # self._store_in_memory(customer_query, final_response)
        
        return final_response
    
    # ========================================================================
    # STUBS FOR FUTURE IMPLEMENTATION
    # ========================================================================
    
    # STUB: Section 4.2 - Tool Use (External API, MCP, Custom Tools)
    def _call_external_api(self, endpoint: str, params: Dict) -> any:
        """
        STUB: Will integrate external APIs for order status, inventory.
        Section 4.2.1
        """
        pass
    
    # STUB: Section 4.2.2 - MCP Tools Integration
    def _connect_mcp_tool(self, tool_name: str) -> any:
        """
        STUB: Will connect to Model Context Protocol tools.
        Section 4.2.2
        """
        pass
    
    # STUB: Section 4.3 - Memory Implementation
    def _store_in_memory(
        self, 
        query: str, 
        response: str
    ) -> None:
        """
        STUB: Will implement persistent memory using vector stores.
        Sections 4.3.1, 4.3.2
        """
        pass
    
    # STUB: Section 4.4 - Feedback Loops
    def _get_human_feedback(self, response: str) -> Dict:
        """
        STUB: Will implement human-in-the-loop feedback.
        Section 4.4.2.1
        """
        pass


# ============================================================================
# DEMONSTRATION CODE
# ============================================================================

def main():
    """
    Demonstrate the Reflexion pattern in action.
    """
    print("=" * 70)
    print("E-COMMERCE CUSTOMER SERVICE AGENT WITH REFLEXION")
    print("=" * 70)
    
    # Initialize the agent
    agent = CustomerServiceAgent(
        model_name="gpt-4",
        temperature=0.3
    )
    
    # Example customer queries
    sample_queries = [
        "I ordered a laptop 3 days ago but haven't received tracking info",
        "Can I return a product if I changed my mind?",
        "What's your policy on damaged items during shipping?"
    ]
    
    # Process first query with visible thinking
    query = sample_queries[0]
    print(f"\nCustomer Query: {query}\n")
    
    final_response = agent.process_query(
        query, 
        show_thinking=True
    )
    
    print("\n" + "=" * 70)
    print("FINAL RESPONSE TO CUSTOMER:")
    print("=" * 70)
    print(f"\n{final_response}\n")


if __name__ == "__main__":
    main()
