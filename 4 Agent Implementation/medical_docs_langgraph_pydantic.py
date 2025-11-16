from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from datetime import datetime
import operator

# Define structured output using Pydantic
class MedicalSummary(BaseModel):
    """Structured medical summary output"""
    chief_complaint: str = Field(description="Primary reason for visit")
    assessment: str = Field(description="Clinical assessment")
    plan: str = Field(description="Treatment plan")
    confidence_score: float = Field(
        description="Confidence in summary accuracy",
        ge=0.0,
        le=1.0
    )

class ReviewResult(BaseModel):
    """Structured review output"""
    approved: bool = Field(description="Whether summary is approved")
    issues: list[str] = Field(
        default_factory=list,
        description="List of identified issues"
    )
    severity: Literal["low", "medium", "high"] = Field(
        description="Severity of issues found"
    )
    requires_human: bool = Field(
        description="Whether human review is required"
    )

# Define memory structure using Pydantic
class FeedbackMemory(BaseModel):
    """Stores feedback for learning"""
    timestamp: datetime
    original_draft: str
    correction: str
    issue_type: str
    feedback_source: Literal["checker", "human"]

# State with reducers for list accumulation
class DocumentState(TypedDict):
    patient_notes: str
    draft_summary: MedicalSummary
    draft_text: str
    review_result: ReviewResult
    final_summary: str
    iteration_count: int
    feedback_history: Annotated[list[FeedbackMemory], operator.add]
    status: str
    human_feedback: str

# Initialize LLM with specific configuration
llm = ChatOpenAI(
    model="gpt-4",
    temperature=0.3,
    max_tokens=1000
)

# Structured LLM for output validation
structured_llm = llm.with_structured_output(MedicalSummary)
review_llm = llm.with_structured_output(ReviewResult)

# Global feedback store for learning
persistent_feedback_store = []

# Maker Agent with self-reflection
def maker_agent(state: DocumentState) -> DocumentState:
    """
    Drafts medical summary with initial self-reflection.
    Demonstrates: Reflexion pattern, structured output.
    """
    # Create initial draft
    prompt = f"""You are a medical documentation assistant.
    Create a structured summary from these patient notes:
    
    {state['patient_notes']}
    
    Provide:
    - Chief Complaint: Primary reason for visit
    - Assessment: Clinical findings and diagnosis
    - Plan: Treatment recommendations
    - Confidence Score: Your confidence in this summary (0.0 to 1.0)
    
    Use proper medical terminology and be thorough."""
    
    draft = structured_llm.invoke(prompt)
    
    # Self-reflection step
    reflection_prompt = f"""Review this medical summary you just created:
    
    Chief Complaint: {draft.chief_complaint}
    Assessment: {draft.assessment}
    Plan: {draft.plan}
    
    Are there any obvious errors, missing information, or terminology 
    issues? Respond with 'CONFIDENT' or describe concerns briefly."""
    
    reflection = llm.invoke(reflection_prompt).content
    
    # Adjust confidence based on self-reflection
    if "CONFIDENT" not in reflection.upper():
        draft.confidence_score = min(draft.confidence_score, 0.7)
    
    # Format as text for review
    draft_text = f"""Chief Complaint: {draft.chief_complaint}

Assessment: {draft.assessment}

Plan: {draft.plan}"""
    
    state['draft_summary'] = draft
    state['draft_text'] = draft_text
    state['status'] = 'drafted'
    state['iteration_count'] = state.get('iteration_count', 0) + 1
    
    return state

# Checker Agent with maker/checker pattern
def checker_agent(state: DocumentState) -> DocumentState:
    """
    Reviews draft for accuracy and compliance.
    Demonstrates: Maker/checker pattern, structured output validation.
    """
    draft = state['draft_summary']
    
    # Load previous feedback for context
    feedback_context = ""
    if persistent_feedback_store:
        recent_feedback = persistent_feedback_store[-3:]
        feedback_context = "\n\nPrevious common issues:\n"
        for fb in recent_feedback:
            feedback_context += f"- {fb.issue_type}\n"
    
    prompt = f"""You are a medical QA reviewer. Review this summary:
    
    {state['draft_text']}
    
    Check for:
    1. Medical terminology accuracy
    2. Completeness of information
    3. Compliance with documentation standards
    4. Logical consistency
    
    Draft confidence score: {draft.confidence_score}
    {feedback_context}
    
    Provide:
    - approved: true/false
    - issues: list of specific problems found
    - severity: low/medium/high
    - requires_human: true if human review needed"""
    
    review = review_llm.invoke(prompt)
    state['review_result'] = review
    
    # Create feedback entry for internal learning
    if not review.approved:
        feedback = FeedbackMemory(
            timestamp=datetime.now(),
            original_draft=state['draft_text'],
            correction="Issues identified by checker",
            issue_type=", ".join(review.issues[:2]),
            feedback_source="checker"
        )
        state['feedback_history'] = [feedback]
    
    # Determine next step
    if review.approved:
        state['final_summary'] = state['draft_text']
        state['status'] = 'approved'
    elif review.requires_human or review.severity == "high":
        state['status'] = 'flagged_for_review'
    else:
        state['status'] = 'needs_revision'
    
    return state

# Revision Agent demonstrates iteration loop
def revision_agent(state: DocumentState) -> DocumentState:
    """
    Revises draft based on checker feedback.
    Demonstrates: ReACT pattern of Reason, Act, Observe.
    """
    # Reasoning step
    issues = state['review_result'].issues
    
    # Prevent infinite loops
    if state['iteration_count'] >= 3:
        state['status'] = 'flagged_for_review'
        return state
    
    # Action: Create revised summary
    prompt = f"""You are revising a medical summary. 
    
    Original summary:
    {state['draft_text']}
    
    Issues identified:
    {chr(10).join(f"- {issue}" for issue in issues)}
    
    Create an improved version addressing these issues.
    Maintain the same structure: Chief Complaint, Assessment, Plan."""
    
    revised = structured_llm.invoke(prompt)
    
    # Update state with revision
    revised_text = f"""Chief Complaint: {revised.chief_complaint}

Assessment: {revised.assessment}

Plan: {revised.plan}"""
    
    state['draft_summary'] = revised
    state['draft_text'] = revised_text
    state['status'] = 'revised'
    state['iteration_count'] += 1
    
    return state

# Human in the Loop with feedback capture
def human_review(state: DocumentState) -> DocumentState:
    """
    Enables nurse practitioner review and feedback collection.
    Demonstrates: Human in the loop, external feedback loops.
    """
    print("\n" + "="*70)
    print("FLAGGED FOR HUMAN REVIEW")
    print("="*70)
    print(f"\nIteration: {state['iteration_count']}")
    print(f"\nDraft Summary:\n{state['draft_text']}")
    
    if state['review_result']:
        print(f"\nChecker Feedback:")
        print(f"  Severity: {state['review_result'].severity}")
        print(f"  Issues:")
        for issue in state['review_result'].issues:
            print(f"    - {issue}")
    
    print("\n" + "="*70)
    print("Options:")
    print("  1. Type 'approve' to accept the draft")
    print("  2. Enter corrected summary")
    print("  3. Type 'reject' to flag for escalation")
    print("="*70)
    
    human_input = input("\nYour decision: ").strip()
    
    if human_input.lower() == 'approve':
        state['final_summary'] = state['draft_text']
        state['status'] = 'human_approved'
    elif human_input.lower() == 'reject':
        state['final_summary'] = "REJECTED - Escalated to senior staff"
        state['status'] = 'rejected'
    else:
        state['final_summary'] = human_input
        state['human_feedback'] = human_input
        state['status'] = 'human_corrected'
        
        # Store human feedback for learning
        feedback = FeedbackMemory(
            timestamp=datetime.now(),
            original_draft=state['draft_text'],
            correction=human_input,
            issue_type="Human correction applied",
            feedback_source="human"
        )
        state['feedback_history'] = [feedback]
        
        # Add to persistent store for future learning
        persistent_feedback_store.append(feedback)
    
    return state

# Routing functions with conditional edges
def route_after_check(
    state: DocumentState
) -> Literal["revision", "human_review", "end"]:
    """Routes based on checker results"""
    status = state['status']
    
    if status == 'approved':
        return "end"
    elif status == 'flagged_for_review':
        return "human_review"
    elif status == 'needs_revision':
        return "revision"
    else:
        return "end"

def route_after_revision(
    state: DocumentState
) -> Literal["checker"]:
    """Always routes back to checker after revision"""
    return "checker"

# Build the graph with all components
def create_medical_doc_workflow():
    """
    Creates the complete medical documentation workflow.
    Demonstrates: StateGraph, nodes, edges, conditional routing.
    """
    workflow = StateGraph(DocumentState)
    
    # Add nodes representing different agents
    workflow.add_node("maker", maker_agent)
    workflow.add_node("checker", checker_agent)
    workflow.add_node("revision", revision_agent)
    workflow.add_node("human_review", human_review)
    
    # Define edges for workflow flow
    workflow.set_entry_point("maker")
    workflow.add_edge("maker", "checker")
    
    # Conditional edges based on checker results
    workflow.add_conditional_edges(
        "checker",
        route_after_check,
        {
            "revision": "revision",
            "human_review": "human_review",
            "end": END
        }
    )
    
    # Revision loops back to checker
    workflow.add_conditional_edges(
        "revision",
        route_after_revision,
        {"checker": "checker"}
    )
    
    # Human review ends the workflow
    workflow.add_edge("human_review", END)
    
    # Compile with checkpointer for persistence
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    return app

# Example usage demonstrating the complete workflow
if __name__ == "__main__":
    app = create_medical_doc_workflow()
    
    # Test case 1: Simple case that should pass
    print("\n" + "="*70)
    print("TEST CASE 1: Routine Visit")
    print("="*70)
    
    initial_state = {
        "patient_notes": """Patient: John Doe, 45yo male
        Presents with persistent cough for 2 weeks, low-grade fever 
        (99.8F), fatigue. Non-smoker. No recent travel. 
        Lungs clear on auscultation. No wheezing.
        Vitals: BP 120/80, HR 72, RR 16, O2 Sat 98% on room air.
        Prescribed azithromycin 500mg daily for 5 days.
        Follow-up in 1 week if symptoms persist.""",
        "draft_text": "",
        "final_summary": "",
        "iteration_count": 0,
        "feedback_history": [],
        "status": "pending",
        "human_feedback": ""
    }
    
    config = {"configurable": {"thread_id": "patient_001"}}
    
    result = app.invoke(initial_state, config)
    
    print("\n" + "="*70)
    print("FINAL RESULT - TEST CASE 1")
    print("="*70)
    print(f"\nStatus: {result['status']}")
    print(f"Iterations: {result['iteration_count']}")
    print(f"\nFinal Summary:\n{result['final_summary']}")
    
    if result['feedback_history']:
        print(f"\nFeedback collected: {len(result['feedback_history'])} items")
    
    # Test case 2: Complex case that might need revision
    print("\n\n" + "="*70)
    print("TEST CASE 2: Complex Case with Potential Issues")
    print("="*70)
    
    complex_state = {
        "patient_notes": """Patient presents with chest pain. 
        Has history of HTN. Pain started yesterday. 
        ECG done. Gave aspirin.""",
        "draft_text": "",
        "final_summary": "",
        "iteration_count": 0,
        "feedback_history": [],
        "status": "pending",
        "human_feedback": ""
    }
    
    config2 = {"configurable": {"thread_id": "patient_002"}}
    
    result2 = app.invoke(complex_state, config2)
    
    print("\n" + "="*70)
    print("FINAL RESULT - TEST CASE 2")
    print("="*70)
    print(f"\nStatus: {result2['status']}")
    print(f"Iterations: {result2['iteration_count']}")
    print(f"\nFinal Summary:\n{result2['final_summary']}")
    
    # Display learning from feedback
    print("\n" + "="*70)
    print("PERSISTENT FEEDBACK STORE")
    print("="*70)
    print(f"Total feedback items collected: "
          f"{len(persistent_feedback_store)}")
    for i, fb in enumerate(persistent_feedback_store, 1):
        print(f"\n{i}. [{fb.feedback_source}] {fb.issue_type}")
        print(f"   Timestamp: {fb.timestamp}")
