"""
Interface Streamlit de referência para o pipeline RAG.

Este arquivo é uma evolução separada do script original. Ele não altera o
arquivo `rag_pipeline_comentado.py`; apenas demonstra como disponibilizar uma
interface de chat para upload de documentos TXT e perguntas sobre o conteúdo.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.chains.qa_with_sources.retrieval import (
    RetrievalQAWithSourcesChain,
)
from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()

APP_TITLE = "RAG Document Assistant"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_TOP_K = 3


def require_environment_variable(name: str) -> str:
    """Retorna uma variável obrigatória ou interrompe a interface com uma mensagem."""
    value = os.getenv(name)

    if not value:
        st.error(
            f"A variável `{name}` não foi configurada. "
            "Revise o arquivo `.env` antes de continuar."
        )
        st.stop()

    return value


def save_uploaded_file(file_bytes: bytes, original_name: str) -> Path:
    """Salva o upload em um arquivo temporário identificável pelo hash do conteúdo."""
    suffix = Path(original_name).suffix or ".txt"
    file_hash = hashlib.sha256(file_bytes).hexdigest()[:16]

    upload_dir = Path(".streamlit_data") / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    file_path = upload_dir / f"{file_hash}{suffix}"
    file_path.write_bytes(file_bytes)

    return file_path


@st.cache_resource(show_spinner=False)
def build_qa_chain(
    file_bytes: bytes,
    original_name: str,
    chunk_size: int,
    chunk_overlap: int,
    top_k: int,
    chat_model_name: str,
    embedding_model_name: str,
) -> RetrievalQAWithSourcesChain:
    """Cria e mantém em cache o pipeline RAG para uma configuração específica."""
    api_key = require_environment_variable("OPENAI_API_KEY")
    file_path = save_uploaded_file(file_bytes, original_name)

    loader = TextLoader(str(file_path), encoding="utf-8")
    raw_documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    documents = splitter.split_documents(raw_documents)

    if not documents:
        raise ValueError("O documento não produziu nenhum chunk.")

    embeddings = OpenAIEmbeddings(
        model=embedding_model_name,
        api_key=api_key,
    )

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
    )

    retriever = vector_store.as_retriever(
        search_kwargs={"k": top_k},
    )

    llm = ChatOpenAI(
        model=chat_model_name,
        temperature=0,
        api_key=api_key,
    )

    return RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        verbose=False,
    )


def normalize_sources(result: dict[str, Any]) -> list[dict[str, str]]:
    """Converte os documentos retornados em uma estrutura simples para a interface."""
    normalized: list[dict[str, str]] = []

    for index, document in enumerate(result.get("source_documents", []), start=1):
        normalized.append(
            {
                "label": f"Fonte {index}",
                "source": str(document.metadata.get("source", "Fonte não informada")),
                "content": document.page_content[:500].strip(),
            }
        )

    return normalized


def render_message(message: dict[str, Any]) -> None:
    """Renderiza uma mensagem e, quando existentes, seus trechos de origem."""
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        sources = message.get("sources", [])
        if sources:
            with st.expander("Trechos utilizados na resposta"):
                for source in sources:
                    st.markdown(f"**{source['label']} — {source['source']}**")
                    st.caption(source["content"])


def main() -> None:
    """Configura e executa a interface Streamlit."""
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="📚",
        layout="wide",
    )

    st.title("📚 RAG Document Assistant")
    st.write(
        "Envie um arquivo `.txt` e faça perguntas usando uma base vetorial "
        "temporária criada com LangChain, OpenAI e ChromaDB."
    )

    with st.sidebar:
        st.header("Configurações")

        chat_model_name = st.text_input(
            "Modelo de chat",
            value=os.getenv("OPENAI_CHAT_MODEL", ""),
            placeholder="Informe um modelo compatível",
        )

        embedding_model_name = st.text_input(
            "Modelo de embeddings",
            value=os.getenv("OPENAI_EMBEDDING_MODEL", ""),
            placeholder="Informe um modelo de embeddings",
        )

        chunk_size = st.number_input(
            "Tamanho do chunk",
            min_value=200,
            max_value=5000,
            value=DEFAULT_CHUNK_SIZE,
            step=100,
        )

        chunk_overlap = st.number_input(
            "Sobreposição entre chunks",
            min_value=0,
            max_value=1000,
            value=DEFAULT_CHUNK_OVERLAP,
            step=25,
        )

        top_k = st.slider(
            "Quantidade de trechos recuperados",
            min_value=1,
            max_value=10,
            value=DEFAULT_TOP_K,
        )

        uploaded_file = st.file_uploader(
            "Documento de conhecimento",
            type=["txt"],
        )

        if st.button("Limpar conversa", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        render_message(message)

    if not uploaded_file:
        st.info("Envie um arquivo TXT na barra lateral para iniciar.")
        return

    if not chat_model_name or not embedding_model_name:
        st.warning(
            "Informe o modelo de chat e o modelo de embeddings na barra lateral "
            "ou configure essas informações no arquivo `.env`."
        )
        return

    if chunk_overlap >= chunk_size:
        st.error("A sobreposição deve ser menor que o tamanho do chunk.")
        return

    try:
        with st.spinner("Preparando documento e banco vetorial..."):
            qa_chain = build_qa_chain(
                file_bytes=uploaded_file.getvalue(),
                original_name=uploaded_file.name,
                chunk_size=int(chunk_size),
                chunk_overlap=int(chunk_overlap),
                top_k=int(top_k),
                chat_model_name=chat_model_name,
                embedding_model_name=embedding_model_name,
            )

    except Exception as error:
        st.exception(error)
        return

    prompt = st.chat_input("Faça uma pergunta sobre o documento")

    if not prompt:
        return

    user_message = {
        "role": "user",
        "content": prompt,
    }
    st.session_state.messages.append(user_message)
    render_message(user_message)

    try:
        with st.chat_message("assistant"):
            with st.spinner("Consultando a base de conhecimento..."):
                result = qa_chain.invoke({"question": prompt})

            answer = result.get("answer", "Nenhuma resposta foi produzida.")
            sources = normalize_sources(result)

            st.markdown(answer)

            if sources:
                with st.expander("Trechos utilizados na resposta"):
                    for source in sources:
                        st.markdown(f"**{source['label']} — {source['source']}**")
                        st.caption(source["content"])

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": sources,
            }
        )

    except Exception as error:
        st.error("Não foi possível processar a pergunta.")
        st.exception(error)


if __name__ == "__main__":
    main()
