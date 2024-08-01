# SPDX-License-Identifier: Apache-2.0

from nifiapi.flowfiletransform import FlowFileTransform, FlowFileTransformResult
from nifiapi.properties import (
    ExpressionLanguageScope,
    PropertyDescriptor,
    StandardValidators,
)
import json
import logging


class SpiderCloudCrawler(FlowFileTransform):
    class Java:
        implements = ["org.apache.nifi.python.processor.FlowFileTransform"]

    class ProcessorDetails:
        version = "2.0.0.dev0"
        description = """Crawls web pages using the SpiderCloud API and processes the content. The content is expected to be in markdown format, 
                         and the output will be in JSON format with the extracted content and metadata."""
        tags = ["web", "crawler", "spidercloud", "json", "document"]
        dependencies = ["requests"]
        input_required = False

    API_KEY = PropertyDescriptor(
        name="API Key",
        description="The API Key to authenticate with the SpiderCloud service.",
        required=True,
        validators=[StandardValidators.NON_EMPTY_VALIDATOR],
    )
    URL = PropertyDescriptor(
        name="URL",
        description="The URL to start crawling.",
        required=True,
        validators=[StandardValidators.URL_VALIDATOR],
    )
    DEPTH = PropertyDescriptor(
        name="Crawl Depth",
        description="The depth for the web crawl.",
        required=True,
        default_value="1",
        validators=[StandardValidators.POSITIVE_INTEGER_VALIDATOR],
    )
    LIMIT = PropertyDescriptor(
        name="Limit",
        description="The maximum number of pages to crawl.",
        required=True,
        default_value="10",
        validators=[StandardValidators.POSITIVE_INTEGER_VALIDATOR],
    )
    RETURN_FORMAT = PropertyDescriptor(
        name="Return Format",
        description="The format to return the crawled data in.",
        required=True,
        allowable_values=["markdown", "html", "text"],
        default_value="markdown",
    )

    property_descriptors = [API_KEY, URL, DEPTH, LIMIT, RETURN_FORMAT]

    def __init__(self, **kwargs):
        logging.basicConfig(level=logging.INFO)

    def getPropertyDescriptors(self):
        return self.property_descriptors

    def get_api_key(self, context):
        return context.getProperty(self.API_KEY).getValue()

    def start_crawling(self, api_key, url, depth, limit, return_format):
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        json_data = {
            "url": url,
            "depth": depth,
            "limit": limit,
            "return_format": return_format,
        }
        try:
            response = requests.post(
                "https://api.spider.cloud/crawl", headers=headers, json=json_data
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"Error during crawling: {e}")
            raise

    def process_crawl_data(self, crawl_data):
        docs = []
        if isinstance(crawl_data, list):
            for item in crawl_data:
                if isinstance(item, dict) and "url" in item and "content" in item:
                    url = item["url"]
                    content = item["content"]
                    if url.startswith("http"):
                        doc = {"page_content": content, "metadata": {"url": url}}
                        docs.append(doc)
                    else:
                        logging.warning(f"Ignored invalid URL: {url}")
                else:
                    logging.warning("Invalid item format, skipping.")
        else:
            logging.error("API response is not a list.")
            raise ValueError("API response is not a list.")
        return docs

    def to_json(self, docs) -> str:
        json_docs = []
        for i, doc in enumerate(docs):
            doc["metadata"]["chunk_index"] = i
            doc["metadata"]["chunk_count"] = len(docs)
            json_docs.append(json.dumps(doc))
        return "\n".join(json_docs)

    def transform(self, context):
        try:
            # Extract properties
            api_key = self.get_api_key(context)
            url = context.getProperty(self.URL).getValue()
            depth = context.getProperty(self.DEPTH).asInteger()
            limit = context.getProperty(self.LIMIT).asInteger()
            return_format = context.getProperty(self.RETURN_FORMAT).getValue()

            # Perform crawling
            crawl_data = self.start_crawling(api_key, url, depth, limit, return_format)
            documents = self.process_crawl_data(crawl_data)
            output_json = self.to_json(documents)

            # Return result
            return FlowFileTransformResult(
                "success",
                contents=output_json,
                attributes={"mime.type": "application/json"},
            )
        except Exception as e:
            logging.error(f"Error in transform: {e}")
            return FlowFileTransformResult("failure")
