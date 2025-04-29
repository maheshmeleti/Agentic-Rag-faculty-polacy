import os
from typing import Literal, Sequence, Annotated, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from utils.common_functions import read_yaml
from src.logger import get_logger
from src.agent.tool_utils import tool_name, tool_description
from src.agent.rag_tool_creator import RAGToolCreator
from src.agent.evaluation_agent_prompts import generate_prompt, quiz_prompt, evaluation_prompt

class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


class SelfEvaluationAgent:
    def __init__(self, config_path: str = "config.yaml", folders: list = None):
        """
        Initialize the SelfEvaluationAgent with configuration and tools.
        """
        self.logger = get_logger(self.__class__.__name__)
        self.logger.debug("Initializing SelfEvaluationAgent...")
        self.config_path = config_path
        self.config = read_yaml(config_path)
        self.rag_tool_creator = RAGToolCreator(config_path=config_path)
        self.llm = self._initialize_llm()
        self.tools = []
        self.folders = folders or []
        self._initialize_tools()
        self.logger.debug("SelfEvaluationAgent initialized successfully.")

    def _initialize_llm(self) -> ChatOllama:
        """
        Initialize the language model (LLM) based on the configuration.
        """
        self.logger.debug("Initializing LLM...")
        model = self.config["llm"]["model"]
        base_url = self.config["llm"]["base_url"]
        llm = ChatOllama(model=model, base_url=base_url)
        self.logger.debug(f"LLM initialized with model: {model}.")
        return llm

    def _initialize_tools(self):
        """
        Initialize tools for the agent.
        """
        self.logger.debug("Initializing tools...")
        db_root_path = self.config["data"]["vector_db_path"]
        rag_tool_creator = RAGToolCreator(config_path="config.yaml")

        for db_name in self.folders[0:1]:
            db_path = os.path.join(db_root_path, db_name)
            tool_name_ = tool_name(db_name)
            tool_description_ = tool_description(db_name)
            self.tools.append(
                rag_tool_creator.create_tool(
                    db_path=db_path,
                    tool_name=tool_name_,
                    tool_description=tool_description_,
                )
            )
            self.logger.info(f"Initialized rag tool from db {db_name}")
        

    def build_agent(self) -> StateGraph:
        """
        Build the state graph for the agent.
        """
        self.logger.debug("Building state graph...")

        # Define nodes
        def agent(state: State):
            return self._agent(state)

        def rewrite(state: State):
            return self._rewrite(state)

        def generate(state: State):
            return self._generate(state)

        def generate_quiz(state: State):
            return self._generate_quiz(state)

        def grade_documents(state: State) -> Literal["generate", "rewrite"]:
            return self._grade_documents(state)

        # Build the graph
        graph_builder = StateGraph(State)
        graph_builder.add_node("agent", agent)
        retriever = ToolNode([self.tools[0]])
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
                END: END,
            },
        )
        graph_builder.add_conditional_edges(
            "retriever",
            grade_documents,
            {
                "generate_quiz": "generate_quiz",
                "generate": "generate",
                "rewrite": "rewrite",
            },
        )
        graph_builder.add_edge("generate_quiz", END)
        graph_builder.add_edge("generate", END)
        graph_builder.add_edge("rewrite", "agent")

        graph = graph_builder.compile()
        self.logger.info("State graph built successfully.")
        return graph

    def _agent(self, state: State):
        """
        Invokes the agent model to generate a response based on the current state.
        """
        self.logger.debug("---CALL AGENT---")
        messages = state["messages"]
        llm_with_tools = self.llm.bind_tools(self.tools, tool_choice="required")
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def _rewrite(self, state: State):
        """
        Transform the query to produce a better question.
        """
        self.logger.debug("---TRANSFORM QUERY---")
        messages = state["messages"]
        question = messages[0].content

        msg = [
            HumanMessage(
                content=f"""
                Look at the input and try to reason about the underlying semantic intent/meaning.
                Here is the initial question:
                {question}
                Formulate an improved question:
                """
            )
        ]
        response = self.llm.invoke(msg)
        return {"messages": [response]}

    def _generate(self, state: State):
        """
        Generate an answer based on the provided context.
        """
        self.logger.debug("---GENERATE---")
        messages = state["messages"]
        last_message = messages[-1]
        docs = last_message.content

        prompt = generate_prompt

        rag_chain = prompt | self.llm | StrOutputParser()
        response = rag_chain.invoke({"context": docs})
        return {"messages": [response]}

    def _generate_quiz(self, state):
        messages = state["messages"]
        last_message = messages[-1]
        context = last_message.content

        prompt = quiz_prompt

        # Chain
        rag_chain = prompt | self.llm | StrOutputParser()

        # Run
        response = rag_chain.invoke({"context": context})
        return {"messages": [response]}

    def _grade_documents(self, state) -> Literal["generate", "rewrite"]:
        """
        Determines whether the retrieved documents are relevant to the question.

        Args:
            state (messages): The current state

        Returns:
            str: A decision for whether the documents are relevant or not
        """

        self.logger.info("---CHECK RELEVANCE---")

        messages = state["messages"]
        question = messages[0].content

        # First check if this is a quiz request
        if self._is_quiz_request(question):
            self.logger.info("---DECISION: QUIZ REQUEST---")
            return "generate_quiz"

        # Data model
        class grade(BaseModel):
            """Binary score for relevance check."""

            binary_score: str = Field(description="Relevance score 'yes' or 'no'")

        # LLM with tool and validation
        llm_with_structured_output = self.llm.with_structured_output(grade)

        # Prompt
        prompt = evaluation_prompt

        # Chain
        chain = prompt | llm_with_structured_output

        messages = state["messages"]
        last_message = messages[-1]

        question = messages[0].content
        docs = last_message.content

        scored_result = chain.invoke({"question": question, "context": docs})

        score = scored_result.binary_score

        if score == "yes":
            self.logger.info("---DECISION: DOCS RELEVANT---")
            return "generate"

        else:
            self.logger.info("---DECISION: DOCS NOT RELEVANT---")
            self.logger.info(score)
            return "rewrite"
        
    def _is_quiz_request(self, question: str) -> bool:
        """Check if the user is requesting a quiz to be generated."""
        quiz_keywords = ["create a quiz", "generate a quiz", "make a quiz", "quiz about"]
        return any(keyword in question.lower() for keyword in quiz_keywords)


if __name__ == "__main__":
    folders = ["2020-2021"]
    agent = SelfEvaluationAgent(folders=folders, config_path="config.yaml")
    graph = agent.build_agent()

    # png_graph = graph.get_graph().draw_mermaid_png()
    # with open("my_graph.png", "wb") as f:
    #     f.write(png_graph)

    input_query = "create a quiz for procedures for faculty appointments for regular ranks?"
    input_query = "what are the procedures for faculty appointments for regular ranks?"
    query = {"messages": [HumanMessage(input_query)]}

    response_messages = []
    for output in graph.stream(query):
        for key, value in output.items():
            print(f"Output from node '{key}':")
            print("----")
            print(value)
        print("\n------\n")

    # response_messages = []
    # for output in graph.stream(query):
    #     for key, value in output.items():
    #         if key == "generate" or key == 'generate_quiz':
    #             for msg in value["messages"]:
    #                 response_messages.append(msg)
    # print(response_messages[-1])