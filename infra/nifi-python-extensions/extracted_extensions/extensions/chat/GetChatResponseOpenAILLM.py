# SPDX-License-Identifier: Apache-2.0

import re

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain.chains import SequentialChain
from langchain.chains import ConversationalRetrievalChain

from pinecone import Pinecone
from langchain_pinecone.vectorstores import PineconeVectorStore

from nifiapi.properties import PropertyDescriptor
from nifiapi.properties import StandardValidators
from nifiapi.properties import ExpressionLanguageScope
from nifiapi.flowfiletransform import FlowFileTransform, FlowFileTransformResult


class GetChatResponseOpenAILLM(FlowFileTransform):
    class Java:
        implements = ["org.apache.nifi.python.processor.FlowFileTransform"]

    class ProcessorDetails:

        dependencies = [
            "langchain",
            "langchain-community",
            "langchain-openai",
            "langchain-pinecone",
            "pinecone-client",
            "openai",
            "tiktoken",
        ]
        version = "0.0.1-dev0"
        description = "Performs a similarity search in Pinecone based on the query/question that is asked and returns list of similar text docs with metadata."
        tags = [
            "Pinecone",
            "AzureOpenAI",
            "AI",
            "Vector Similarity Search",
            "Vector Database",
        ]

    # AzureOpenAI
    AZURE_OPENAI_API_KEY = PropertyDescriptor(
        name="AzureOpenAI API Key",
        description="The API key to connect to AzureOpeanAI services",
        required=True,
        sensitive=True,
    )
    AZURE_VERSION = PropertyDescriptor(
        name="AzureOpenAI version",
        description="The version of the used AzureOpeanAI services",
        required=True,
        sensitive=True,
    )
    AZURE_ENDPOINT = PropertyDescriptor(
        name="AzureOpenAI endpoint",
        description="The endpoint for the used AzureOpeanAI services",
        required=True,
        sensitive=True,
    )
    # Embedding Model
    AZURE_OPENAI_EMBEDDING_MODEL = PropertyDescriptor(
        name="AzureOpenAI Embedding Model name",
        description="The OpenAI embedding model to use to convert query/question to a text embedding which is then used to search for similar docs.",
        required=True,
        default_value="ttext-embedding-ada-002",
        allowable_values=[
            "ttext-embedding-ada-002",
            "text-embedding-ada-002",
            "text-davinci-001",
            "text-curie-001",
            "text-babbage-001",
            "text-ada-001",
        ],
    )
    # Parameters of the Chat Model
    OPENAI_LLM_MODEL = PropertyDescriptor(
        name="OpenAI LLM Model",
        description="The OpenAI LLM model to answer the user question/query",
        required=True,
        default_value="gpt-4o",
        allowable_values=[
            "gpt-4o",
            "gpt-4o-turbo",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-0301",
            "text-davinci-003",
            "text-davinci-002",
            "code-davinci-002",
        ],
    )
    OPENAI_LLM_TEMPERATURE = PropertyDescriptor(
        name="LLM temperature",
        default_value="0",
        description="The temperature controls how much randomness is in the output. O means no randomness while 1 means high randomness. Valid values from 0-1",
        required=True,
    )
    OPENAI_MAX_TOEKNS = PropertyDescriptor(
        name="LLM Max Tokens",
        default_value=None,
        description="Max Token can achieve",
        required=True,
    )
    # Pinecone
    PINECONE_API_KEY = PropertyDescriptor(
        name="Pinecone API Key",
        description="The API key to connect to the Pinecone to get relevant documents for the question",
        required=True,
        sensitive=True,
    )
    PINECONE_INDEX_NAME = PropertyDescriptor(
        name="Index Name",
        description="The Pinecone index to store the embeddings.",
        required=True,
    )
    # General Parameters: Question, ChatHistory, Nbr of Docs returned and username
    QUESTION = PropertyDescriptor(
        name="question",
        default_value="Hello Can You help me Today",
        description="The question/chat that the LLM needs to answer/respond to.",
        required=False,
        expression_language_scope=ExpressionLanguageScope.FLOWFILE_ATTRIBUTES,
    )
    CHAT_HISTORY = PropertyDescriptor(
        name="chat_history",
        default_value=None,
        description="The previous chat history so the LLM has more context",
        required=False,
        expression_language_scope=ExpressionLanguageScope.FLOWFILE_ATTRIBUTES,
    )
    property_descriptors = [
        AZURE_OPENAI_API_KEY,
        AZURE_VERSION,
        AZURE_ENDPOINT,
        AZURE_OPENAI_EMBEDDING_MODEL,
        OPENAI_LLM_MODEL,
        OPENAI_LLM_TEMPERATURE,
        OPENAI_MAX_TOEKNS,
        PINECONE_API_KEY,
        PINECONE_INDEX_NAME,
        QUESTION,
        CHAT_HISTORY,
    ]

    def __init__(self, **kwargs):
        self.Azure_openai_embedding_service = None
        self.pc = None
        self.chat_llm = None

    def getPropertyDescriptors(self):
        return self.property_descriptors

    def onScheduled(self, context):
        self.logger.info(
            "Initializing AzureOpenAI, AzureOpenAI Embedding Service and Pinecone."
        )

        # Configure the AzureOpenAI Embedding Service
        azure_openai_api_key = context.getProperty(self.AZURE_OPENAI_API_KEY).getValue()
        azure_version = context.getProperty(self.AZURE_VERSION).getValue()
        azure_endpoint = context.getProperty(self.AZURE_ENDPOINT).getValue()
        azure_openai_embedding_model = context.getProperty(
            self.AZURE_OPENAI_EMBEDDING_MODEL
        ).getValue()

        self.azure_embedding_service = AzureOpenAIEmbeddings(
            api_key=azure_openai_api_key,
            azure_endpoint=azure_endpoint,
            azure_deployment=azure_openai_embedding_model,
            api_version=azure_version,
        )

        # Correcting the property names
        pinecone_api_key = context.getProperty(self.PINECONE_API_KEY).getValue()
        pinecone_index_name = context.getProperty(self.PINECONE_INDEX_NAME).getValue()

        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index = self.pc.Index(pinecone_index_name)

        vector_store = PineconeVectorStore(
            index=self.index, embedding=self.azure_embedding_service
        )

        # Configuring the parameters of the model
        temperature = context.getProperty(self.OPENAI_LLM_TEMPERATURE).getValue()
        llm_model_name = context.getProperty(self.OPENAI_LLM_MODEL).getValue()

        _template = """
                Given the following extracted parts of a long document and a question,
                create a final answer with references ("SOURCES") unless identified below.
                Always try to provide a detailed answer based on the provided information.

                QUESTION: {question}
                ========= =========
                FINAL ANSWER: """

        QA_PROMPT = PromptTemplate.from_template(_template)

        max_tokens = context.getProperty(self.OPENAI_MAX_TOEKNS).getValue()
        if max_tokens is not None:
            max_tokens = int(max_tokens)

        # Create the chain
        self.chat_llm = AzureChatOpenAI(
            openai_api_key=azure_openai_api_key,
            api_version=azure_version,
            azure_endpoint=azure_endpoint,
            temperature=temperature,
            azure_deployment=llm_model_name,
            max_tokens=max_tokens,
            timeout=None,
        )

        memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )
        self.qa = ConversationalRetrievalChain.from_llm(
            llm=self.chat_llm, memory=memory, retriever=vector_store.as_retriever()
        )
        # Set the custom prompt template directly in the chain
        # self.qa.combine_docs_chain.llm_chain.prompt = QA_PROMPT

    def transform(self, context, flowFile):

        # General parameters
        question = (
            context.getProperty(self.QUESTION)
            .evaluateAttributeExpressions(flowFile)
            .getValue()
        )

        # Chat history
        chat_history = (
            context.getProperty(self.CHAT_HISTORY)
            .evaluateAttributeExpressions(flowFile)
            .getValue()
        )
        if chat_history is None:
            chat_history = ""

        escaped_chat_history = chat_history.replace("\n", "-")
        self.logger.info("Escaped Chat History is: " + escaped_chat_history)

        regex_pattern = r"\((.*?)\)"
        array_of_tuples_chat_history = re.findall(regex_pattern, chat_history)
        array_of_tuples_chat_history = [
            tuple_str.split('", "') for tuple_str in array_of_tuples_chat_history
        ]
        array_of_tuples_chat_history = [
            (value[1:-1], value2[1:-1])
            for value, value2 in array_of_tuples_chat_history
        ]

        self.logger.info(
            "******* Inside transform of GetChatResponseOpenAILLM with question: "
            + question
        )
        try:
            result = self.qa(
                {"question": question, "chat_history": array_of_tuples_chat_history}
            )
            retrieved_docs = result.get("retrieved_documents", [])
            self.logger.info(f"Retrieved documents: {retrieved_docs}")

            self.logger.info(f"Result: {result}")
            answer = result.get("answer", "No answer found.")
            answer = answer.replace("\n", "").replace("'", "\\'")
            self.logger.info("LLM answer for question[" + question + "] is: " + answer)
        except Exception as e:
            self.logger.error(f"Error occurred during QA processing: {str(e)}")
            answer = "Error processing request."

        return FlowFileTransformResult(relationship="success", contents=answer)
