# src/agent/quiz_tool.py
from typing import Dict, Any
from pydantic import Field
from langchain_core.tools import BaseTool
from langchain_ollama import ChatOllama
from utils.common_functions import read_yaml
from src.logger import get_logger


class QuizGeneratorTool(BaseTool):
    name: str = "generate_quiz"
    description: str = """Use this tool when the user asks to create, make, or generate a quiz about academic policies. 
    The input should be a dictionary with a 'topic' key containing the quiz topic."""
    config_path: str = Field(default="config.yaml", description="Path to the configuration file.")
    config: Dict[str, Any] = Field(default_factory=dict, description="Configuration loaded from the YAML file.")
    logger: Any = Field(default=None, exclude=True)  # Exclude logger from Pydantic validation
    llm: ChatOllama = Field(default=None, exclude=True)  # Exclude llm from Pydantic validation
    
    def __init__(self, config_path: str = "config.yaml"):
        super().__init__()
        self.logger = get_logger(self.__class__.__name__)
        self.config = read_yaml(config_path)
        self.llm = self._initialize_llm()
        self.logger.debug("Quiz tool initialized")

    def _run(self, input: Dict[str, Any]) -> str:
        """
        Generate a quiz based on the provided topic.
        """
        try:
            # Validate input
            topic = input.get("topic")
            if not topic:
                return "Error: Please specify a topic for the quiz."

            # Construct the prompt
            prompt = self._construct_prompt(topic)

            # Invoke the LLM
            response = self.llm.invoke(prompt)
            if not response or not response.content:
                return "Error: Failed to generate a quiz. The response was empty."

            return response.content

        except Exception as e:
            self.logger.error(f"Error generating quiz: {e}")
            return "I encountered an error while generating the quiz. Please try again."

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

    def _construct_prompt(self, topic: str) -> str:
        """
        Construct the prompt for the LLM based on the topic.
        """
        return f"""
        Create a 3-5 question multiple choice quiz about {topic} based on Clemson University's academic policies.
        Format each question like this:
        
        Q1. [Question text]
        A) [Option A]
        B) [Option B]
        C) [Option C]
        D) [Option D]
        Answer: [Correct letter]
        
        [Brief explanation]
        
        ---
        """