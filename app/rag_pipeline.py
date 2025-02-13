from haystack import Document, Pipeline
from haystack.dataclasses import ChatMessage
from haystack.components.embedders import (
    SentenceTransformersTextEmbedder,
    SentenceTransformersDocumentEmbedder
)
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.routers import TransformersTextRouter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.builders import ChatPromptBuilder
from haystack_integrations.components.generators.mistral import MistralChatGenerator
from datasets import load_dataset


def create_rag_pipeline(
        classifier_path: str,
        embedding_model_name: str = "BAAI/bge-small-en-v1.5",
        top_k=5,
        generator: OpenAIChatGenerator = None
        ) -> Pipeline:
    """
    Creates and configures a Retrieval-Augmented Generation (RAG) pipeline.

    Args:
        classifier_path (str): Path to the classifier model for text routing.
        embedding_model_name (str): Name of the embedding model to use.
        top_k (int): Number of top documents to retrieve.
        generator (OpenAIChatGenerator, optional): Generator component for chat.
            Defaults to MistralChatGenerator.

    Returns:
        Pipeline: Configured RAG pipeline.
    """
    # 1. Initialize components
    text_router = TransformersTextRouter(model=classifier_path)
    text_router.warm_up()

    # 2. Document Store
    document_store = initialize_document_store(
        embedding_model_name=embedding_model_name,
        dataset_name="bilgeyucel/seven-wonders")

    # 3. Retriever components
    text_embedder = SentenceTransformersTextEmbedder(
        model=embedding_model_name
    )

    retriever = InMemoryEmbeddingRetriever(
        document_store=document_store,
        top_k=top_k,
    )

    # 4. Generator component
    if generator is None:
        generator = MistralChatGenerator()

    # 5. Prompt configuration
    prompt_builder = create_prompt_builder()

    # 6. Build pipeline
    pipeline = Pipeline()
    pipeline.add_component("router", text_router)
    pipeline.add_component("text_embedder", text_embedder)
    pipeline.add_component("retriever", retriever)
    pipeline.add_component("prompt_builder", prompt_builder)
    pipeline.add_component("generator", generator)

    # 7. Connect components
    pipeline.connect("router.LABEL_1", "text_embedder.text")
    pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    pipeline.connect("retriever.documents", "prompt_builder.documents")
    pipeline.connect("router.LABEL_0", "prompt_builder.question")
    pipeline.connect("prompt_builder.prompt", "generator.messages")

    return pipeline


def initialize_document_store(
        embedding_model_name: str,
        dataset_name: str,
        ) -> InMemoryDocumentStore:
    """
    Initialize the document store with sample data and embeddings.

    Args:
        embedding_model_name (str): Name of the embedding model to use.
        dataset_name (str): Name of the dataset used to retrieve.

    Returns:
        InMemoryDocumentStore: Initialized document store.
    """
    dataset = load_dataset(dataset_name, split="train")
    docs = [Document(content=doc["content"], meta=doc["meta"]) for doc in dataset]

    doc_embedder = SentenceTransformersDocumentEmbedder(
        model=embedding_model_name
    )
    doc_embedder.warm_up()

    docs_with_embeddings = doc_embedder.run(docs)
    document_store = InMemoryDocumentStore()
    document_store.write_documents(docs_with_embeddings["documents"])

    return document_store



def create_prompt_builder() -> ChatPromptBuilder:
    """
    Creates a prompt builder with a predefined template.

    Returns:
        ChatPromptBuilder: Configured prompt builder.
    """
    prompt_template = [
        ChatMessage.from_user("""
You are an expert in the subject matter. Below is the context extracted from relevant documents.
Answer the following question using only the information provided in the context.
If the answer cannot be deduced from the context, reply with something like:
"I didn't find enough complete information to answer this question correctly."

Context:
{% for document in documents %}
    {{ document.content }}
{% endfor %}

If the 'Context' is empty, answer the following question based solely on your internal expertise.
Provide a detailed, accurate, and concise answer. If you are unsure, state "I'm not sure" instead of guessing.

Question: {{question}}
Answer:
""")
    ]

    return ChatPromptBuilder(template=prompt_template)
