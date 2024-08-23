import re

from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai import AzureChatOpenAI

from pinecone import Pinecone
from langchain_pinecone.vectorstores import PineconeVectorStore

from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain

from langchain.chains import ConversationalRetrievalChain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT

from nifiapi.properties import PropertyDescriptor
from nifiapi.properties import StandardValidators
from nifiapi.properties import ExpressionLanguageScope
from nifiapi.flowfiletransform import FlowFileTransform, FlowFileTransformResult

import logging

logging.basicConfig(
    filemode="nifi-app.log",
    path="opt/nifi/nifi-curretn/logs",
    level=logging.DEBUG,  # or logging.INFO if you want to capture info level and above
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


class GetChatResponseOpenAILLM(FlowFileTransform):
    class Java:
        implements = ["org.apache.nifi.python.processor.FlowFileTransform"]

    class ProcessorDetails:
        dependencies = [
            "langchain",
            "langchain_community",
            "langchain-openai",
            "langchain-pinecone",
            "pinecone-client",
            "tiktoken",
        ]
        version = "0.0.1-SNAPSHOT"
        description = "Performs a similarity search in Pinecone based on the query/question that is asked and returns list of similar text docs with metadata."
        tags = [
            "Pinecone",
            "AzureOpenAI",
            "AI",
            "Vector Similarity Search",
            "Vector Database",
        ]

    def __init__(self, **kwargs):
        # AzureOpenAI
        self.azure_openai_api_key = PropertyDescriptor(
            name="AzureOpenAI API Key",
            description="The API key to connect to AzureOpeanAI services",
            required=True,
            sensitive=True,
        )
        self.azure_version = PropertyDescriptor(
            name="AzureOpenAI version",
            description="The version of the used AzureOpeanAI services",
            required=True,
            sensitive=True,
        )
        self.azure_endpoint = PropertyDescriptor(
            name="AzureOpenAI endpoint",
            description="The endpoint for the used AzureOpeanAI services",
            required=True,
            sensitive=True,
        )

        # Embedding Model
        self.azure_openai_embedding_api_key = PropertyDescriptor(
            name="AzureOpenAI Embedding Model API Key",
            description="The API key to connect to AzureOpeanAI Embedding Model services",
            required=True,
            sensitive=True,
        )
        self.azure_embedding_version = PropertyDescriptor(
            name="AzureOpenAI version",
            description="The version of the used AzureOpeanAI services",
            required=True,
            sensitive=True,
        )
        self.azure_embedding_endpoint = PropertyDescriptor(
            name="AzureOpenAI endpoint",
            description="The endpoint for the used AzureOpeanAI services",
            required=True,
            sensitive=True,
        )
        self.azure_openai_embedding_model = PropertyDescriptor(
            name="AzureOpenAI Embedding Model name",
            description="The OpenAI embedding model to use to convert query/question to a text embedding which is then used to search for similar docs.",
            required=True,
            default_value="text-embedding-ada-002",
            allowable_values=[
                "ttext-embedding-ada-002",
                "text-embedding-ada-002",
                "text-davinci-001",
                "text-curie-001",
                "text-babbage-001",
                "text-ada-001",
            ],
        )

        # Parameters of the model
        self.openai_llm_model = PropertyDescriptor(
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
        self.openai_llm_temperature = PropertyDescriptor(
            name="LLM temperature",
            default_value="0.0",
            description="The temperature controls how much randomness is in the output. O means no randomness while 1 means high randomness. Valid values from 0-1",
            required=True,
        )

        # Pinecone
        self.pinecone_api_key = PropertyDescriptor(
            name="Pinecone API Key",
            description="The API key to connect to the Pinecone to get relevant documents for the question",
            required=True,
            sensitive=True,
        )
        self.pinecone_index_name = PropertyDescriptor(
            name="Index Name",
            description="The Pinecone index to store the embeddings.",
            required=True,
        )

        # General Parameters: Question, ChatHistory, Nbr of Docs returned and username
        self.question = PropertyDescriptor(
            name="question",
            default_value="0",
            description="The question/chat that the LLM needs to answer/respond to.",
            required=False,
            expression_language_scope=ExpressionLanguageScope.FLOWFILE_ATTRIBUTES,
        )
        self.chat_history = PropertyDescriptor(
            name="chat_history",
            default_value="0",
            description="The previous chat history so the LLM has more context",
            required=False,
            expression_language_scope=ExpressionLanguageScope.FLOWFILE_ATTRIBUTES,
        )
        self.search_results_size = PropertyDescriptor(
            name="Number of Similar Documents to Return",
            default_value="10",
            description="The number of similar documents to return from the similarity searech",
            required=False,
        )
        self.user = PropertyDescriptor(
            name="User Name",
            description="The name of the user asking the questions.",
            required=False,
        )

        self.descriptors = [
            self.azure_openai_api_key,
            self.openai_llm_model,
            self.azure_openai_embedding_api_key,
            self.azure_embedding_version,
            self.azure_embedding_endpoint,
            self.azure_openai_embedding_model,
            self.openai_llm_temperature,
            self.question,
            self.chat_history,
            self.pinecone_api_key,
            self.pinecone_index_name,
            self.user,
            self.search_results_size,
        ]

    def getPropertyDescriptors(self):
        return self.descriptors

    def onScheduled(self, context):
        self.logger.info(
            "Initializing AzureOpenAI, AzureOpenAI Embedding Service and Pinecone."
        )

        # Configure the AzureOpenAI Service
        azure_openai_api_key = context.getProperty(
            self.azure_openai_api_key.name
        ).getValue()
        azure_version = context.getProperty(self.azure_version.name).getValue()
        azure_endpoint = context.getProperty(self.azure_endpoint.name).getValue()

        # Configure the AzureOpenAI Embedding Service
        azure_openai_embedding_api_key = context.getProperty(
            self.azure_openai_embedding_api_key.name
        ).getValue()
        azure_embedding_version = context.getProperty(
            self.azure_embedding_version.name
        ).getValue()
        azure_embedding_endpoint = context.getProperty(
            self.azure_embedding_endpoint.name
        ).getValue()
        azure_openai_embedding_model = context.getProperty(
            self.azure_openai_embedding_model.name
        ).getValue()

        azure_embedding_service = AzureOpenAIEmbeddings(
            api_key=azure_openai_embedding_api_key,
            azure_endpoint=azure_embedding_endpoint,
            api_version=azure_embedding_version,
            model=azure_openai_embedding_model,
        )

        # Configure Pinecone
        pinecone_api_key = context.getProperty(self.pinecone_api_key.name).getValue()
        pinecone_index_name = context.getProperty(
            self.pinecone_index_name.name
        ).getValue()

        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index = self.pc.Index(pinecone_index_name)

        pinecone_vector_store = PineconeVectorStore(
            index=self.index, embedding=azure_embedding_service
        )

        # Configuring the parameters of the model
        temperature = context.getProperty(self.openai_llm_temperature.name).getValue()
        llm_model_name = context.getProperty(self.openai_llm_model.name).getValue()

        _template = """ Given the following extracted parts of a long document and a question,
            create a final answer with references ("SOURCES") unless identified below.
            But if you are asked something similar to what your purpose is as an AI Assistant, then answer with the following:
            I'm a helpful assistant for {username} answering his questions based on the informations from the website.
            Also, ALWAYS return a "SOURCES" part in your answer.

            QUESTION: {question}
            =========
            {summaries}
            =========
            FINAL ANSWER:"""
        QA_PROMPT = PromptTemplate.from_template(_template)

        # Create the chain
        chat_llm = AzureChatOpenAI(
            api_key=azure_openai_api_key,
            api_version=azure_version,
            azure_endpoint=azure_endpoint,
            temperature=temperature,
            model=llm_model_name,
        )

        question_generator = LLMChain(
            llm=chat_llm, prompt=CONDENSE_QUESTION_PROMPT, verbose=True
        )
        doc_chain = load_qa_with_sources_chain(
            chat_llm, chain_type="stuff", verbose=False, prompt=QA_PROMPT
        )
        self.qa_chain = ConversationalRetrievalChain(
            retriever=pinecone_vector_store.as_retriever(),
            question_generator=question_generator,
            combine_docs_chain=doc_chain,
            verbose=False,
        )

    def transform(self, context, flowFile):

        # General parameters
        user = context.getProperty(self.user.name).getValue()
        question = (
            context.getProperty(self.question.name)
            .evaluateAttributeExpressions(flowFile)
            .getValue()
        )

        # Chat history
        chat_history = (
            context.getProperty(self.chat_history.name)
            .evaluateAttributeExpressions(flowFile)
            .getValue()
        )
        escaped_chat_history = chat_history.replace("\n", "-")
        self.logger.info("Escaped Chat History is: " + escaped_chat_history)

        regex_pattern = r"\[\('([^']+)', '([^']+)'\)\]"
        array_of_tuples_chat_history = re.findall(regex_pattern, chat_history)
        array_of_tuples_chat_history = [
            tuple_str.split('", "') for tuple_str in array_of_tuples_chat_history
        ]
        array_of_tuples_chat_history = [
            (value[1:-1], value2[1:-1])
            for value, value2 in array_of_tuples_chat_history
        ]

        self.logger.info(
            "********* Inside transform of GetChatResponseOpenAILLM with question: "
            + question
        )
        result = self.qa_chain(
            {
                "username": user,
                "question": question,
                "chat_history": array_of_tuples_chat_history,
            }
        )
        answer = result["answer"]
        answer = answer.replace("\n", "").replace("'", "\\'")
        self.logger.info("LLM answer for question[" + question + "] is: " + answer)

        return FlowFileTransformResult(relationship="success", contents=answer)
