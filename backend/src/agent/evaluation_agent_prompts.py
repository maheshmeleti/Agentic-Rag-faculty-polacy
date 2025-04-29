from langchain_core.prompts import PromptTemplate

generate_prompt = PromptTemplate(
            template="""
            You are a highly accurate summarization assistant. Your task is to summarize the provided context strictly based on the information in the documents.
            Do not include any information that is not explicitly mentioned in the context. Avoid adding assumptions, external knowledge, or hallucinations.

            Here is the context:
            {context}

            Provide a concise and accurate summary of the context in your response, just write Here is information about your query form the documents and print the points.
            """,
            input_variables=["context"],
        )

quiz_prompt = PromptTemplate(
            template="""
            Create a 3-5 question multiple choice quiz about clemson university based on the following context \n.
            {context} \n\n
            Format each ouput like this:

            Here is your quiz - \n
            
            Q1. [Question text]
            A) [Option A] 
            B) [Option B] 
            C) [Option C] 
            D) [Option D]
            Answer: [Correct letter]

            dont write anything on top 
            
            """,
            input_variables=["context"]
        )

evaluation_prompt = PromptTemplate(
            template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
            Here is the retrieved document: \n\n {context} \n\n
            Here is the user question: {question} \n
            If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
            input_variables=["context", "question"],
        )