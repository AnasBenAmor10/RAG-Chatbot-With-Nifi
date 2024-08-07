# SPDX-License-Identifier: Apache-2.0

from pinecone import ServerlessSpec, Pinecone
from nifiapi.flowfiletransform import FlowFileTransform, FlowFileTransformResult
from nifiapi.properties import (
    PropertyDescriptor,
    StandardValidators,
    ExpressionLanguageScope,
)
import json
import uuid
import logging


class PutPineconeVectorEmbedding(FlowFileTransform):
    class Java:
        implements = ["org.apache.nifi.python.processor.FlowFileTransform"]

    class ProcessorDetails:
        version = "2.0.0-SNAPSHOT"
        description = """Publishes JSON data to Pinecone. The Incoming data must be in single JSON per Line format, each with two keys: 'text' and 'metadata'.
                       The text must be a string, while metadata must be a map with strings for values. Any additional fields will be ignored."""
        tags = [
            "pinecone",
            "vector",
            "vectordb",
            "vectorstore",
            "embeddings",
            "ai",
            "artificial intelligence",
            "ml",
            "machine learning",
            "text",
            "LLM",
        ]
        dependencies = ["pinecone-client"]

    PINECONE_API_KEY = PropertyDescriptor(
        name="Pinecone API Key",
        description="The API Key to use in order to authenticate with Pinecone",
        sensitive=True,
        required=True,
        validators=[StandardValidators.NON_EMPTY_VALIDATOR],
    )
    PINECONE_ENV = PropertyDescriptor(
        name="Pinecone Environment",
        description="The name of the Pinecone Environment. This can be found in the Pinecone console next to the API Key.",
        sensitive=False,
        required=True,
        validators=[StandardValidators.NON_EMPTY_VALIDATOR],
    )
    INDEX_NAME = PropertyDescriptor(
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

    properties = [
        PINECONE_API_KEY,
        PINECONE_ENV,
        INDEX_NAME,
        NAMESPACE,
        METRIC,
        DIMENSION,
    ]
    pc = None

    def __init__(self, **kwargs):
        pass

    def getPropertyDescriptors(self):
        return self.properties

    def onScheduled(self, context):
        self.logger.info("onScheduled called. Preparing to initialize Pinecone client.")

        # Check if Pinecone and other dependencies are available
        try:
            self.logger.info("Checking for Pinecone and other dependencies...")
            import pinecone

            self.logger.info("Pinecone module is available.")
        except ImportError as e:
            self.logger.error(f"Missing dependency: {e}")
            raise e

        # Initialize Pinecone client
        try:
            api_key = context.getProperty(self.PINECONE_API_KEY).getValue()
            pinecone_env = context.getProperty(self.PINECONE_ENV).getValue()
            self.logger.info(f"API Key: {api_key}, Environment: {pinecone_env}")

            self.pc = Pinecone(api_key=api_key, environment=pinecone_env)
            self.logger.info("Pinecone client initialized successfully.")
        except Exception as e:
            self.logger.error(f"Error initializing Pinecone client: {e}")
            raise e

    def transform(self, context, flowfile):
        self.logger.info("Transform method started.")

        # Read the content of the FlowFile
        try:
            embedding_docs_string = flowfile.getContentsAsBytes().decode("utf-8")
            self.logger.info(f"Raw content from FlowFile: {embedding_docs_string}")
        except Exception as e:
            self.logger.error(f"Failed to read content from FlowFile: {e}")
            return FlowFileTransformResult("failure")

        # Parse the JSON content
        try:
            embedding_docs_json_list_deserialized = json.loads(embedding_docs_string)
            self.logger.info(
                f"Deserialized JSON: {embedding_docs_json_list_deserialized}"
            )
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decoding failed: {e}")
            return FlowFileTransformResult("failure")

        # First, check if our index already exists. If it doesn't, we create it
        try:
            index_name = (
                context.getProperty(self.INDEX_NAME)
                .evaluateAttributeExpressions(flowfile)
                .getValue()
            )
            namespace = (
                context.getProperty(self.NAMESPACE)
                .evaluateAttributeExpressions(flowfile)
                .getValue()
            )
            indexes = self.pc.list_indexes()
            if index_name not in indexes:
                self.logger.info(f"Index {index_name} does not exist. Creating index.")
                try:
                    self.pc.create_index(
                        name=index_name,
                        dimension=int(context.getProperty(self.DIMENSION).getValue()),
                        metric=context.getProperty(self.METRIC).getValue(),
                        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                    )
                    self.logger.info(f"Index {index_name} created successfully.")
                except Exception as create_exception:
                    if "ALREADY_EXISTS" in str(create_exception):
                        self.logger.info(
                            f"Index {index_name} already exists. Skipping creation."
                        )
                    else:
                        self.logger.error(f"Error creating index: {create_exception}")
                        return FlowFileTransformResult("failure")

            pinecone_index = self.pc.Index(index_name)
        except Exception as e:
            self.logger.error(f"Error interacting with Pinecone index: {e}")
            return FlowFileTransformResult("failure")

        # Fetch existing vectors and their metadata using the correct IDs
        try:
            ids_to_fetch = []
            for item in embedding_docs_json_list_deserialized:
                entry_source = item["metadata"]["source"]
                chunk_index = item["metadata"]["chunk_index"]
                unique_key = f"{entry_source}_{chunk_index}"
                ids_to_fetch.append(unique_key)

            existing_source = {}
            if ids_to_fetch:
                response = pinecone_index.fetch(ids=ids_to_fetch, namespace=namespace)
                for vector_id, vector_data in response.get("vectors", {}).items():
                    source = vector_data.get("metadata", {}).get("source")
                    if source:
                        existing_source[source] = vector_id
                self.logger.info(f"Fetched existing sources: {existing_source}")
        except Exception as e:
            self.logger.error(f"Error fetching existing vectors: {e}")
            return FlowFileTransformResult("failure")

        # Upsert entries to Pinecone
        try:
            entries = []
            for item in embedding_docs_json_list_deserialized:
                entry_source = item["metadata"]["source"]
                chunk_index = item["metadata"]["chunk_index"]
                unique_key = f"{entry_source}_{chunk_index}"
                entry_id = existing_source.get(unique_key, str(uuid.uuid4()))
                entry = {
                    "id": entry_id,
                    "values": item["embedding"],
                    "metadata": {
                        "source": entry_source,
                        "chunk_index": chunk_index,
                        "chunk_count": item["metadata"]["chunk_count"],
                    },
                }
                entries.append(entry)
            pinecone_index.upsert(entries, namespace=namespace)
            self.logger.info(f"Upserted {len(entries)} entries to Pinecone.")
        except Exception as e:
            self.logger.error(f"Error during upsert to Pinecone: {e}")
            return FlowFileTransformResult("failure")

        return FlowFileTransformResult(relationship="success")
