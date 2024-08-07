# SPDX-License-Identifier: Apache-2.0

import json
import logging
from nifiapi.properties import PropertyDescriptor, StandardValidators
from nifiapi.flowfiletransform import FlowFileTransform, FlowFileTransformResult
from langchain_openai import AzureOpenAIEmbeddings


class GetVectorEmbedding(FlowFileTransform):
    class Java:
        implements = ["org.apache.nifi.python.processor.FlowFileTransform"]

    class ProcessorDetails:
        dependencies = [
            "langchain",
            "langchain-openai",
            "langchain-community",
            "openai",
        ]
        version = "2.0.0.dev0"
        description = "Creates text embeddings for each text chunk using OpenAI embedding model services via Langchain libraries"
        tags = ["AI", "OpenAI", "Embeddings", "Langchain", "Vectors"]

    API_KEY = PropertyDescriptor(
        name="AzureOpenAI API Key",
        description="The API key to connect to AzureOpenAI services",
        required=True,
        sensitive=True,
        validators=[StandardValidators.NON_EMPTY_VALIDATOR],
    )

    Embedding_model = PropertyDescriptor(
        name="OpenAI Embedding Models",
        description="The OpenAI embedding model to use when creating the text embedding vector.",
        required=True,
        default_value="text-embedding-ada-002",
        allowable_values=[
            "text-embedding-ada-002",
            "ttext-embedding-ada-002",
            "text-davinci-001",
            "text-curie-001",
            "text-babbage-001",
            "text-ada-001",
            "text-embedding-3-small",
            "ext-embedding-3-large",
        ],
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

    chunk_size = PropertyDescriptor(
        name="Chunk Size",
        description="The number of characters that each text chunk will have when used by OpenAI to create the text embedding.",
        default_value="1000",
        required=True,
        validators=[StandardValidators.NON_EMPTY_VALIDATOR],
    )

    property_descriptors = [
        API_KEY,
        Embedding_model,
        AZURE_OPENAI_ENDPOINT,
        AZURE_OPENAI_API_VERSION,
        chunk_size,
    ]

    def __init__(self, **kwargs):
        self.Azure_openai_embedding_service = None

    def getPropertyDescriptors(self):
        return self.property_descriptors

    def onScheduled(self, context):
        self.logger.info("Initializing AzureOpenAI Embedding Service")

        # Get the properties from the processor needed to configure the OpenAI Embedding model
        azure_openai_api_key = context.getProperty(self.API_KEY).getValue()
        embedding_model = context.getProperty(self.Embedding_model).getValue()
        version = context.getProperty(self.AZURE_OPENAI_API_VERSION).getValue()
        endpoint = context.getProperty(self.AZURE_OPENAI_ENDPOINT).getValue()

        # Initialize OpenAI Embedding Service
        self.Azure_openai_embedding_service = AzureOpenAIEmbeddings(
            azure_endpoint=endpoint,
            api_key=azure_openai_api_key,
            azure_deployment=embedding_model,
            openai_api_version=version,
            model=embedding_model,
        )

    def transform(self, context, flowfile):
        try:
            self.logger.info("Inside transform of GetVectorEmbedding..")

            # Read the content of the FlowFile
            chunked_docs_string = flowfile.getContentsAsBytes().decode("utf-8")
            self.logger.info(f"Raw content from FlowFile: {chunked_docs_string}")

            # Parse the JSON content
            try:
                chunk_docs_json_list_deserialized = json.loads(chunked_docs_string)
            except json.JSONDecodeError as e:
                self.logger.error(f"JSON decoding failed: {e}")
                return FlowFileTransformResult("failure")

            self.logger.info(f"Deserialized JSON: {chunk_docs_json_list_deserialized}")

            # Verify the structure of the JSON
            if not isinstance(chunk_docs_json_list_deserialized, list):
                raise ValueError("The deserialized JSON is not a list.")

            for item in chunk_docs_json_list_deserialized:
                if not isinstance(item, dict):
                    raise ValueError(f"Item is not a dictionary: {item}")
                if "text" not in item or "metadata" not in item:
                    raise ValueError(f"Dictionary missing 'text' or 'metadata': {item}")

            self.logger.info(
                "The number of text documents to be embedded are: "
                + str(len(chunk_docs_json_list_deserialized))
            )

            texts = []
            metadatas = []

            for doc_dict in chunk_docs_json_list_deserialized:
                texts.append(doc_dict["text"])
                metadata = doc_dict["metadata"]
                metadatas.append(metadata)

            # Create embeddings for each text block
            vector_embeddings = self.Azure_openai_embedding_service.embed_documents(
                texts=texts
            )

            # Combine text, metadata, and embeddings into JSON
            json_list_with_text_embeddings = []

            for text, vector_embedding, metadata in zip(
                texts, vector_embeddings, metadatas
            ):
                text_embedding_json = {
                    "text": text,
                    "embedding": vector_embedding,
                    "metadata": metadata,
                }
                json_list_with_text_embeddings.append(text_embedding_json)

            # Convert the list of JSON objects into a single JSON string
            json_embedding_string = json.dumps(json_list_with_text_embeddings)

            return FlowFileTransformResult(
                "success",
                contents=json_embedding_string,
                attributes={"mime.type": "application/json"},
            )
        except Exception as e:
            self.logger.error(f"Error in transform: {e}")
            return FlowFileTransformResult("failure")
