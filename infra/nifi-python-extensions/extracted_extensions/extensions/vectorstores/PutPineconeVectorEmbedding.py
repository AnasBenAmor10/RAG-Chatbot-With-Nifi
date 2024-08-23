# SPDX-License-Identifier: Apache-2.0

from langchain_openai import AzureOpenAIEmbeddings
from pinecone import ServerlessSpec, Pinecone
from langchain_pinecone.vectorstores import PineconeVectorStore
from langchain_core.documents import Document
from uuid import uuid4
from nifiapi.flowfiletransform import FlowFileTransform, FlowFileTransformResult
from nifiapi.properties import (
    PropertyDescriptor,
    StandardValidators,
    ExpressionLanguageScope,
)
import json
import logging
import time


class PutPineconeVectorEmbedding(FlowFileTransform):
    class Java:
        implements = ["org.apache.nifi.python.processor.FlowFileTransform"]

    class ProcessorDetails:
        version = "2.0.0-dev0"
        description = """Publishes the chunk documents to Pinecone. The Incoming data must be in single JSON per Line format, each with two keys: 'text' and 'metadata'.
                       The text must be a string, while metadata must be a map with strings for values. Any additional fields will be ignored."""
        tags = [
            "pinecone",
            "vector",
            "vectordb",
            "vectorstore",
            "embeddings",
            "text",
            "LLM",
        ]
        dependencies = [
            "pinecone-client",
            "langchain-openai",
            "langchain-community",
            "langchain-pinecone",
            "langchain-core",
            "openai",
        ]

    PINECONE_API_KEY = PropertyDescriptor(
        name="Pinecone API Key",
        description="The API Key to use in order to authenticate with Pinecone",
        sensitive=True,
        required=True,
        validators=[StandardValidators.NON_EMPTY_VALIDATOR],
    )
    PINECONE_INDEX_NAME = PropertyDescriptor(
        name="Index Name",
        description="The name of the Pinecone index.",
        sensitive=False,
        required=True,
        validators=[StandardValidators.NON_EMPTY_VALIDATOR],
        expression_language_scope=ExpressionLanguageScope.FLOWFILE_ATTRIBUTES,
    )
    NAMESPACE = PropertyDescriptor(
        name="Namespace",
        description="The name of the Pinecone Namespace to put the documents to.",
        required=False,
        validators=[StandardValidators.NON_EMPTY_VALIDATOR],
        expression_language_scope=ExpressionLanguageScope.FLOWFILE_ATTRIBUTES,
    )
    METRIC = PropertyDescriptor(
        name="Metric",
        description="The method of measuring the similarity or distance between vectors.",
        required=False,
        validators=[StandardValidators.NON_EMPTY_VALIDATOR],
        allowable_values=["cosine", "euclidean", "dot_product"],
        default_value="euclidean",
    )
    DIMENSION = PropertyDescriptor(
        name="Vector Embedding dimension",
        description="The dimension of vector embedding model",
        required=True,
        validators=[StandardValidators.POSITIVE_INTEGER_VALIDATOR],
        default_value="1536",
    )

    AZURE_API_KEY = PropertyDescriptor(
        name="AzureOpenAI API Key",
        description="The API key to connect to AzureOpenAI services",
        required=True,
        sensitive=True,
        validators=[StandardValidators.NON_EMPTY_VALIDATOR],
    )

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
        validators=[StandardValidators.NON_EMPTY_VALIDATOR],
    )

    AZURE_OPENAI_ENDPOINT = PropertyDescriptor(
        name="AzureOpenAI Endpoint",
        description="The endpoint URL for the Azure OpenAI service.",
        required=True,
        validators=[StandardValidators.NON_EMPTY_VALIDATOR],
    )

    AZURE_OPENAI_API_VERSION = PropertyDescriptor(
        name="AzureOpenAI Version",
        description="The Azure OpenAI service version.",
        required=True,
        validators=[StandardValidators.NON_EMPTY_VALIDATOR],
    )

    properties = [
        PINECONE_API_KEY,
        PINECONE_INDEX_NAME,
        NAMESPACE,
        METRIC,
        DIMENSION,
        AZURE_API_KEY,
        AZURE_OPENAI_EMBEDDING_MODEL,
        AZURE_OPENAI_ENDPOINT,
        AZURE_OPENAI_API_VERSION,
    ]

    def __init__(self, **kwargs):
        self.pc = None
        self.Azure_openai_embedding_service = None

    def getPropertyDescriptors(self):
        return self.properties

    def onScheduled(self, context):
        self.logger.info("Initializing AzureOpenAI Embedding Service")
        try:
            # Get the properties from the processor needed to configure the OpenAI Embedding model
            azure_openai_api_key = context.getProperty(self.AZURE_API_KEY).getValue()
            embedding_model = context.getProperty(
                self.AZURE_OPENAI_EMBEDDING_MODEL
            ).getValue()
            version = context.getProperty(self.AZURE_OPENAI_API_VERSION).getValue()
            endpoint = context.getProperty(self.AZURE_OPENAI_ENDPOINT).getValue()

            # Initialize OpenAI Embedding Service
            self.azure_embedding_service = AzureOpenAIEmbeddings(
                api_key=azure_openai_api_key,
                azure_endpoint=endpoint,
                azure_deployment=embedding_model,
                api_version=version,
            )
            self.logger.info("AzureOpenAI Embedding Service initialized successfully.")
        except Exception as e:
            self.logger.error(f"Error initializing AzureOpenAI Embedding Service: {e}")
            return FlowFileTransformResult("failure")

        try:
            pinecone_api_key = context.getProperty(self.PINECONE_API_KEY).getValue()

            # Initialize Pinecone client
            self.logger.info("Initializing Pinecone client.")
            self.pc = Pinecone(api_key=pinecone_api_key)

        except Exception as e:
            self.logger.error(f"Error initializing Pinecone client: {e}")
            return FlowFileTransformResult("failure")

    def transform(self, context, flowfile):
        self.logger.info("Transform method started.")

        try:
            self.logger.info("Inside transform of PutPineconeVectorEmbedding..")

            # Read the content of the FlowFile
            chunked_docs_string = flowfile.getContentsAsBytes().decode("utf-8")
            self.logger.info(f"Raw content from FlowFile: {chunked_docs_string}")

            # Parse the JSON content
            try:
                chunk_docs_json_list_deserialized = json.loads(chunked_docs_string)
                self.logger.info(
                    f"Deserialized JSON: {chunk_docs_json_list_deserialized}"
                )
            except json.JSONDecodeError as e:
                self.logger.error(f"JSON decoding failed: {e}")
                return FlowFileTransformResult("failure")

            # Verify the structure of the JSON
            if not isinstance(chunk_docs_json_list_deserialized, list):
                raise ValueError("The deserialized JSON is not a list.")

            # Check if the index exists
            try:
                indexes = self.pc.list_indexes()
                self.logger.info(f"Existing Pinecone indexes: {indexes}")
            except Exception as e:
                self.logger.error(f"Error listing indexes: {e}")
                return FlowFileTransformResult("failure")

            pinecone_index_name = context.getProperty(
                self.PINECONE_INDEX_NAME
            ).getValue()
            existing_indexes = [
                index_info["name"] for index_info in self.pc.list_indexes()
            ]

            if pinecone_index_name not in existing_indexes:
                self.pc.create_index(
                    name=pinecone_index_name,
                    dimension=int(context.getProperty(self.DIMENSION).getValue()),
                    metric=context.getProperty(self.METRIC).getValue(),
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                )
                while not self.pc.describe_index(pinecone_index_name).status["ready"]:
                    time.sleep(1)

            index = self.pc.Index(pinecone_index_name)

            # Initialize vector store
            try:
                vector_store = PineconeVectorStore(
                    index=index, embedding=self.azure_embedding_service
                )
                self.logger.info("Vector store initialized successfully.")
            except Exception as e:
                self.logger.error(f"Error initializing vector store: {e}")
                return FlowFileTransformResult("failure")

            documents = []
            for item in chunk_docs_json_list_deserialized:
                if not isinstance(item, dict):
                    raise ValueError(f"Item is not a dictionary: {item}")
                if "page_content" not in item or "metadata" not in item:
                    raise ValueError(
                        f"Dictionary missing 'page_content' or 'metadata': {item}"
                    )

                # Create a Document object for each entry
                document = Document(
                    page_content=item["page_content"].strip(),
                    metadata={
                        "source": item["metadata"]["source"],
                        "chunk_index": item["metadata"]["chunk_index"],
                        "chunk_count": item["metadata"]["chunk_count"],
                    },
                )
                documents.append(document)

            self.logger.info(
                f"The number of text documents to be embedded: {len(documents)}"
            )

            # Generate UUIDs
            uuids = [str(uuid4()) for _ in range(len(documents))]

            # Add documents to vector store
            try:
                vector_store.add_documents(documents=documents, ids=uuids)
                self.logger.info(
                    "Documents successfully added to Pinecone vector store."
                )
            except Exception as e:
                self.logger.error(
                    f"Error adding documents to Pinecone vector store: {e}"
                )
                return FlowFileTransformResult("failure")

        except Exception as e:
            self.logger.error(f"Error in transform: {e}")
            return FlowFileTransformResult("failure")

        return FlowFileTransformResult(relationship="success")
