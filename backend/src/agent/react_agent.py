import os
import warnings
# import gradio as gr
from langchain_ollama import ChatOllama
from langgraph.graph import MessagesState, START, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.agent.rag_tool_creator import RAGToolCreator
from src.agent.tool_utils import tool_name, tool_description
from src.agent.react_prompts import system_prompt
from src.logger import get_logger
from utils.common_functions import read_yaml
from src.agent.quiz_generator_tool import QuizGeneratorTool

warnings.filterwarnings("ignore")


class ReactAgent:
    """
    A class to encapsulate the logic for the Retrieval-Augmented Generation (RAG) agent.
    """

    def __init__(self, folders: list, config_path: str = "config.yaml"):
        """
        Initialize the ReactAgent with configuration.
        """
        self.logger = get_logger(self.__class__.__name__)
        self.logger.debug("Initializing ReactAgent...")
        self.config_path = config_path
        self.config = read_yaml(config_path)
        self.llm = self._initialize_llm()
        self.rag_folders = folders
        self.tools = []
        self._initialize_RAG_tools()
        self._initialize_quiz_tool()  # Add quiz tool
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.logger.debug("ReactAgent initialized successfully.")

    def build_agent(self) -> StateGraph:
        """
        Build the state graph for the agent.
        """
        self.logger.debug("Building state graph...")
        sys_msg = SystemMessage(content=system_prompt)

        # Node
        def assistant(state: MessagesState):
            return {"messages": [self.llm_with_tools.invoke([sys_msg] + state["messages"])]}

        # Graph
        builder = StateGraph(MessagesState)

        # Define nodes: these do the work
        builder.add_node("assistant", assistant)
        builder.add_node("tools", ToolNode(self.tools))

        # Define edges: these determine the control flow
        builder.add_edge(START, "assistant")
        builder.add_conditional_edges("assistant", tools_condition)
        builder.add_edge("tools", "assistant")
        graph = builder.compile()
        self.logger.info("State graph built successfully.")
        return graph


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

    def _initialize_RAG_tools(self):
        """
        Initialize tools using the RAGToolCreator.
        """
        self.logger.debug("Initializing tools...")
        db_root_path = self.config["data"]["vector_db_path"]
        rag_tool_creator = RAGToolCreator(config_path="config.yaml")

        for db_name in self.rag_folders:
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

    def _initialize_quiz_tool(self):
        """Initialize the quiz generation tool."""
        self.logger.info("Initializing quiz tool...")
        quiz_tool = QuizGeneratorTool(config_path=self.config_path)
        self.tools.append(quiz_tool)
        self.logger.debug("Quiz tool initialized successfully.")    

    
if __name__ == '__main__':
    from pprint import pprint
    input = "compare polacies of 2018-2019 to 2020-2021 and tell what are the major updates"
    input = "create a quiz on faculty appointments for regular ranks?"
    query = {"messages": [HumanMessage(input)]}
    folders = ['2020-2021']
    agent = ReactAgent(folders)
    graph = agent.build_agent()
    graph.invoke(query)
    for output in graph.stream(query):
        for key, value in output.items():
            pprint(f"Output from node '{key}':")
            pprint("----")
            pprint(value, indent=4, width=120)

        pprint("\n------\n")
