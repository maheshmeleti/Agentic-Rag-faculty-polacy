import os
from utils.common_functions import read_yaml
from src.agent.tool_utils import tool_name, tool_description
from src.agent.rag_tool_creator import RAGToolCreator

from typing import Annotated, Sequence, TypedDict, Literal 
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import  BaseMessage, HumanMessage

from langchain import hub
from langchain_core.output_parsers import StrOutputParser

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
import pprint

class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

model = "llama3.2:3b"
# # model = "qwen2.5"
llm = ChatOllama(model=model, base_url="http://localhost:11434")

config_path = "config.yaml"
config = read_yaml(config_path)
db_root_path = config["data"]["vector_db_path"]
rag_tool_creator = RAGToolCreator(config_path=config_path)
db_name = '2020-2021'
tool_name_ = tool_name(db_name)
tool_description_ = tool_description(db_name)
db_path = os.path.join(db_root_path, db_name)
retriever = rag_tool_creator.create_tool(
                    db_path=db_path,
                    tool_name=tool_name_,
                    tool_description=tool_description_,
                )
tools = [retriever]
# question = "what are the procedures for faculty appointments for regular ranks?"
# print(retriever.invoke(question))

def is_quiz_request(question: str) -> bool:
    """Check if the user is requesting a quiz to be generated."""
    quiz_keywords = ["create a quiz", "generate a quiz", "make a quiz", "quiz about"]
    return any(keyword in question.lower() for keyword in quiz_keywords)

def grade_documents(state) -> Literal["generate", "rewrite"]:
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (messages): The current state

    Returns:
        str: A decision for whether the documents are relevant or not
    """

    print("---CHECK RELEVANCE---")

    messages = state["messages"]
    question = messages[0].content

    # First check if this is a quiz request
    if is_quiz_request(question):
        print("---DECISION: QUIZ REQUEST---")
        return "generate_quiz"

    # Data model
    class grade(BaseModel):
        """Binary score for relevance check."""

        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    # LLM with tool and validation
    llm_with_structured_output = llm.with_structured_output(grade)

    # Prompt
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
        input_variables=["context", "question"],
    )

    # Chain
    chain = prompt | llm_with_structured_output

    messages = state["messages"]
    last_message = messages[-1]

    question = messages[0].content
    docs = last_message.content

    scored_result = chain.invoke({"question": question, "context": docs})

    score = scored_result.binary_score

    if score == "yes":
        print("---DECISION: DOCS RELEVANT---")
        return "generate"

    else:
        print("---DECISION: DOCS NOT RELEVANT---")
        print(score)
        return "rewrite"
    
def agent(state):
    """
    Invokes the agent model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply end.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with the agent response appended to messages
    """
    print("---CALL AGENT---")
    messages = state["messages"]

    llm_with_tools = llm.bind_tools(tools, tool_choice="required")
    response = llm_with_tools.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}

def rewrite(state):
    """
    Transform the query to produce a better question.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with re-phrased question
    """

    print("---TRANSFORM QUERY---")
    messages = state["messages"]
    question = messages[0].content

    msg = [
        HumanMessage(
            content=f""" \n 
    Look at the input and try to reason about the underlying semantic intent / meaning. \n 
    Here is the initial question:
    \n ------- \n
    {question} 
    \n ------- \n
    Formulate an improved question: """,
        )
    ]

    # Grader
    response = llm.invoke(msg)
    return {"messages": [response]}

def generate(state):
    """
    Generate answer

    Args:
        state (messages): The current state

    Returns:
         dict: summerized answer generation
    """
    print("---GENERATE---")
    messages = state["messages"]
    question = messages[0].content
    last_message = messages[-1]

    docs = last_message.content

    # Prompt
    # prompt = hub.pull("rlm/rag-prompt")
    prompt = PromptTemplate(template="""
        You are a highly accurate summarization assistant. Your task is to summarize the provided context strictly based on the information in the documents. 
        Do not include any information that is not explicitly mentioned in the context. Avoid adding assumptions, external knowledge, or hallucinations.

        Here is the context:
        {context}

        Provide a concise and accurate summary of the context in your response.
        """,
        input_variables=["context"])



    # Post-processing
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Chain
    rag_chain = prompt | llm | StrOutputParser()

    # Run
    # response = rag_chain.invoke({"context": docs, "question": question})
    response = rag_chain.invoke({"context": docs})
    return {"messages": [response]}

def generate_quiz(state):
    messages = state["messages"]
    last_message = messages[-1]
    context = last_message.content

    prompt = PromptTemplate(
        template="""
        Create a 3-5 question multiple choice quiz about clemson university based on the following context \n.
        {context} \n\n
        Format each question like this:
        
        Q1. [Question text]
        A) [Option A]
        B) [Option B]
        C) [Option C]
        D) [Option D]
        Answer: [Correct letter]
        
        [Brief explanation]
        """,
        input_variables=["context"]
    )

    # Chain
    rag_chain = prompt | llm | StrOutputParser()

    # Run
    response = rag_chain.invoke({"context": context})
    return {"messages": [response]}

graph_builder = StateGraph(State)

graph_builder.add_node("agent", agent)
retriever = ToolNode([retriever])
graph_builder.add_node("retriever", retriever)

graph_builder.add_node("rewrite", rewrite)
graph_builder.add_node("generate", generate)
graph_builder.add_node("generate_quiz", generate_quiz)


graph_builder.add_edge(START, "agent")

graph_builder.add_conditional_edges( 
    "agent",

    tools_condition,
    {
        "tools": "retriever",
        END: END
    }
)

# graph_builder.add_conditional_edges(
#     "retriever",
#     grade_documents
# )

graph_builder.add_conditional_edges(
    "retriever",
    grade_documents,
    {
        "generate_quiz": "generate_quiz",
        "generate": "generate",
        "rewrite": "rewrite",
    }
)

graph_builder.add_edge("generate_quiz", END)
graph_builder.add_edge("generate", END)
graph_builder.add_edge("rewrite", "agent")

graph = graph_builder.compile()

# input = "what are the procedures for faculty appointments for regular ranks?"
input = "create a quiz for procedures for faculty appointments for regular ranks?"

query = {"messages": [HumanMessage(input)]}

for output in graph.stream(query):
    for key, value in output.items():
        print(f"Output from node '{key}':")
        print("----")
        print(value)

    print("\n------\n")