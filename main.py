# ======================================================================================
# IMPORTAÇÕES
# ======================================================================================
# Esta seção reúne as bibliotecas utilizadas no pipeline RAG:
#   1. configuração do ambiente;
#   2. carregamento e divisão do documento;
#   3. criação de embeddings;
#   4. armazenamento vetorial no ChromaDB;
#   5. recuperação de contexto;
#   6. geração de respostas com fontes.
#
# IMPORTANTE:
# O código abaixo mantém os imports e o comportamento originalmente fornecidos.
# Existe um conflito de nomes entre:
#
#     from openai import OpenAI
#
# e:
#
#     from langchain_openai import OpenAIEmbeddings, OpenAI
#
# O segundo import substitui a referência anterior chamada `OpenAI`.
# Consequentemente, quando `OpenAI(...)` é usado mais abaixo, ele representa a
# classe importada de `langchain_openai`, e não necessariamente o cliente do SDK
# oficial importado na primeira linha. A correção é indicada no README, mas não
# foi aplicada aqui para preservar a lógica original.
# ======================================================================================

# Biblioteca padrão para acesso a variáveis de ambiente e recursos do sistema operacional.
import os

# Carrega variáveis declaradas em um arquivo `.env`.
from dotenv import load_dotenv

# Integração do LangChain com o banco vetorial ChromaDB.
from langchain_chroma import Chroma

# Componentes de integração entre LangChain e OpenAI.
# `OpenAIEmbeddings` transforma textos em vetores numéricos.
# `OpenAI` representa o modelo de linguagem disponibilizado pela integração.
from langchain_openai import (
    ChatOpenAI,
    OpenAIEmbeddings,
)

from langchain_ollama import (
    ChatOllama,
    OllamaEmbeddings,
)


# Divisor de texto que tenta preservar parágrafos, frases e palavras.
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Loader utilizado para transformar um arquivo de texto em documentos LangChain.
from langchain_community.document_loaders import TextLoader

# Chain clássica para perguntas e respostas com identificação das fontes recuperadas.
from langchain_classic.chains.qa_with_sources.retrieval import (
    RetrievalQAWithSourcesChain,
)


# ======================================================================================
# 1. CONFIGURAÇÃO DO AMBIENTE E DA CHAVE DA API
# ======================================================================================

# Carrega as variáveis do arquivo `.env` para o ambiente da aplicação.
load_dotenv()

def escolher_provider():

    provider_env = os.getenv(
        "AI_PROVIDER",
        "ask",
    ).lower()

    if provider_env in ["ollama", "openai"]:
        return provider_env

    print()
    print("=" * 60)
    print("SELECIONE O PROVIDER")
    print("=" * 60)
    print()
    print("1 - LOCAL  | Ollama + Qwen")
    print("2 - CLOUD  | OpenAI")
    print()

    while True:

        opcao = input(
            "Digite 1 ou 2: "
        ).strip()

        if opcao == "1":
            return "ollama"

        if opcao == "2":
            return "openai"

        print(
            "Opção inválida. Digite 1 ou 2."
        )

def criar_modelos(provider):

    # ========================================================
    # OLLAMA
    # ========================================================

    if provider == "ollama":

        base_url = os.getenv(
            "OLLAMA_BASE_URL",
            "http://localhost:11434",
        )

        chat_model = os.getenv(
            "OLLAMA_CHAT_MODEL",
            "qwen3.5:0.8b",
        )

        embedding_model = os.getenv(
            "OLLAMA_EMBEDDING_MODEL",
            "qwen3-embedding:0.6b",
        )

        print()
        print("=" * 60)
        print("MODELO LOCAL")
        print("=" * 60)

        print(f"Provider....: Ollama")
        print(f"LLM.........: {chat_model}")
        print(f"Embeddings..: {embedding_model}")

        llm = ChatOllama(
            model=chat_model,
            base_url=base_url,
            temperature=0,
        )

        embeddings = OllamaEmbeddings(
            model=embedding_model,
            base_url=base_url,
        )

        return llm, embeddings

    # ========================================================
    # OPENAI
    # ========================================================

    if provider == "openai":

        api_key = os.getenv(
            "OPENAI_API_KEY"
        )

        if not api_key:

            raise ValueError(
                "OPENAI_API_KEY não encontrada no .env."
            )

        chat_model = os.getenv(
            "OPENAI_CHAT_MODEL",
            "gpt-5-mini",
        )

        embedding_model = os.getenv(
            "OPENAI_EMBEDDING_MODEL",
            "text-embedding-3-small",
        )

        print()
        print("=" * 60)
        print("MODELO OPENAI")
        print("=" * 60)

        print(f"Provider....: OpenAI")
        print(f"LLM.........: {chat_model}")
        print(f"Embeddings..: {embedding_model}")

        llm = ChatOpenAI(
            model=chat_model,
            api_key=api_key,
        )

        embeddings = OpenAIEmbeddings(
            model=embedding_model,
            api_key=api_key,
        )

        return llm, embeddings

    raise ValueError(
        f"Provider inválido: {provider}"
    )


# Exibe os primeiros caracteres da chave apenas como verificação.
#
# ALERTA DE SEGURANÇA:
# Mesmo uma parte da chave não deve ser exibida em logs, notebooks compartilhados,
# capturas de tela ou ambientes de produção. Esta linha foi mantida somente porque
# já fazia parte do código original.
#


# ======================================================================================
# 2. DEFINIÇÃO E CARREGAMENTO DO ARQUIVO DE DADOS
# ======================================================================================

# Define o caminho do arquivo que será usado como base de conhecimento.
#
# O arquivo `eleven_madison_park_data.txt` deve estar no diretório de execução
# do script ou notebook.
DATA_FILE_PATH = "./docs/eleven_madison_park_data.txt"
print(f"Data file path set to: {DATA_FILE_PATH}")

# Informa no console que o processo de leitura será iniciado.
print(f"Attempting to load data from: {DATA_FILE_PATH}")

# Cria o loader responsável por ler o arquivo de texto.
#
# O encoding UTF-8 permite processar corretamente acentos e outros caracteres
# Unicode presentes no conteúdo.
loader = TextLoader(DATA_FILE_PATH, encoding="utf-8")

# Carrega o arquivo.
#
# O TextLoader retorna uma lista de objetos `Document`.
# Cada objeto normalmente possui:
#   - `page_content`: conteúdo textual;
#   - `metadata`: informações como o caminho do arquivo de origem.
raw_documents = loader.load()
print(f"Successfully loaded {len(raw_documents)} document(s).")

# Exibe os primeiros 500 caracteres do primeiro documento para uma validação rápida.
#
# Esta verificação ajuda a confirmar se o arquivo correto foi carregado e se o
# encoding foi interpretado adequadamente.
print(raw_documents[0].page_content[:500] + "...")


# ======================================================================================
# 3. DIVISÃO DO DOCUMENTO EM CHUNKS
# ======================================================================================

# Documentos muito grandes não devem ser enviados integralmente para a etapa de busca.
# Por isso, o conteúdo será dividido em partes menores chamadas `chunks`.
print("\nSplitting the loaded document into smaller chunks...")

CHUNK_SIZE = int(
    os.getenv(
        "CHUNK_SIZE",
        "1000",
    )
)

CHUNK_OVERLAP = int(
    os.getenv(
        "CHUNK_OVERLAP",
        "150",
    )
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)

# Divide os documentos carregados em vários objetos `Document` menores.
# Os metadados do documento original são preservados nos chunks.
documents = text_splitter.split_documents(raw_documents)

# Essa validação evita que as etapas de embeddings e armazenamento vetorial sejam
# executadas com uma coleção vazia.
if not documents:
    raise ValueError(
        "Error: Splitting resulted in zero documents. "
        "Check the input file and splitter settings."
    )

print(f"Document split into {len(documents)} chunks.")

print("\n--- Example Chunk (Chunk 2) ---")
print(documents[2].page_content)
.
print("\n--- Metadata for Chunk 2 ---")
print(documents[2].metadata)


# ======================================================================================
# INICIALIZAÇÃO DO MODELO DE CHAT E EMBEDDINGS
# ======================================================================================

print("Initializing OpenAI Embeddings model...")

provider = escolher_provider()

llm, embeddings = criar_modelos(
    provider
)

print("\nCreating ChromaDB vector store and embedding documents...")

vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
)

vector_count = vector_store._collection.count()
print(f"ChromaDB vector store created with {vector_count} items.")

# Garante que o banco vetorial não foi criado vazio.
if vector_count == 0:
    raise ValueError(
        "Vector store creation resulted in 0 items. Check previous steps."
    )

stored_data = vector_store._collection.get(
    include=["embeddings", "documents"],
    limit=1,
)

# Exibe o conteúdo textual do primeiro registro armazenado.
print("First chunk text:\n", stored_data["documents"][0])

print("\nEmbedding vector:\n", stored_data["embeddings"][0])

# Exibe a quantidade de dimensões do vetor.
print(
    f"\nFull embedding has "
    f"{len(stored_data['embeddings'][0])} dimensions."
)

TOP_K = int(
    os.getenv(
        "TOP_K",
        "3",
    )
)
retriever = vector_store.as_retriever(
    search_kwargs={
        "k": TOP_K
    }
)
print("Retriever configured successfully from vector store.")

qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    verbose=True,
)

print("RetrievalQAWithSourcesChain created")


# ======================================================================================
# 11. TESTE COMPLETO DO PIPELINE RAG
# ======================================================================================

# Define uma pergunta de teste que deverá ser respondida com base no documento.
print("\n--- Testing the Full RAG Chain ---")

chain_test_query = "What kind of food does Eleven Madison Park serve?"
print(f"Query: {chain_test_query}")

try:
    # Executa a chain.
    #
    # O método `invoke` recebe um dicionário. Nesta chain, a pergunta é enviada
    # através da chave `question`.
    result = qa_chain.invoke({"question": chain_test_query})

    # Exibe a resposta produzida pelo modelo.
    print("\n--- Answer ---")
    print(result.get("answer", "No answer generated."))

    # Exibe as fontes identificadas pela chain.
    print("\n--- Sources ---")
    print(result.get("sources", "No sources identified."))

    # Exibe pequenos trechos dos documentos recuperados.
    #
    # Essa inspeção é útil para verificar se a resposta foi construída a partir
    # de conteúdos realmente relacionados à pergunta.
    if "source_documents" in result:
        print("\n--- Source Document Snippets ---")

        for i, doc in enumerate(result["source_documents"]):
            content_snippet = doc.page_content[:250].strip()
            print(f"Doc {i + 1}: {content_snippet}")

except Exception as e:
    # Captura falhas gerais durante a execução completa da chain.
    #
    # Em um projeto modular, recomenda-se criar exceções específicas para:
    #   - erro de configuração;
    #   - erro no carregamento do documento;
    #   - erro na geração de embeddings;
    #   - erro de comunicação com o modelo;
    #   - erro na recuperação vetorial.
    print(f"\nAn error occurred while running the chain: {e}")
