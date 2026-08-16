# ======================================================================================
# RAG COM DOIS PROVIDERS: OLLAMA/QWEN (LOCAL) E OPENAI (CLOUD)
# ======================================================================================
#
# Objetivo deste script:
#   1. carregar um documento de texto;
#   2. dividir o conteúdo em chunks;
#   3. gerar embeddings;
#   4. armazenar os vetores no ChromaDB;
#   5. recuperar os chunks mais relevantes para uma pergunta;
#   6. montar o contexto recuperado;
#   7. enviar contexto + pergunta para o LLM;
#   8. permitir escolher entre:
#        - Ollama + Qwen, executando localmente;
#        - OpenAI, executando via API.
#
# O mesmo pipeline RAG é utilizado nos dois modos. O que muda é apenas:
#   - o modelo de chat (LLM);
#   - o modelo de embeddings.
#
# Configuração sugerida no .env:
#
#   AI_PROVIDER=ask
#
#   OLLAMA_BASE_URL=http://localhost:11434
#   OLLAMA_CHAT_MODEL=qwen3.5:0.8b
#   OLLAMA_EMBEDDING_MODEL=qwen3-embedding:0.6b
#
#   OPENAI_API_KEY=sua-chave-aqui
#   OPENAI_CHAT_MODEL=gpt-5-nano
#   OPENAI_EMBEDDING_MODEL=text-embedding-3-small
#
#   DATA_FILE_PATH=./docs/eleven_madison_park_data.txt
#   CHUNK_SIZE=1000
#   CHUNK_OVERLAP=150
#   TOP_K=3
#   TEST_QUERY=What kind of food does Eleven Madison Park serve?
#
# Dependências principais com uv:
#
#   uv add python-dotenv langchain-core langchain-text-splitters \
#       langchain-chroma langchain-openai langchain-ollama chromadb
#
# Depois:
#
#   uv lock
#   uv sync
#   uv run python main.py
# ======================================================================================


# ======================================================================================
# 1. IMPORTAÇÕES
# ======================================================================================

# Biblioteca padrão para variáveis de ambiente.
import os

# Path facilita leitura e validação de caminhos de arquivos.
from pathlib import Path

# Carrega as variáveis declaradas no arquivo .env.
from dotenv import load_dotenv

# Objeto Document usado pelo LangChain para armazenar texto + metadados.
from langchain_core.documents import Document

# Prompt estruturado compatível tanto com ChatOllama quanto com ChatOpenAI.
from langchain_core.prompts import ChatPromptTemplate

# Integração do LangChain com o ChromaDB.
from langchain_chroma import Chroma

# Modelos OpenAI.
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Modelos Ollama.
from langchain_ollama import ChatOllama, OllamaEmbeddings

# Divisor de texto em chunks.
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ======================================================================================
# 2. CARREGAMENTO DAS VARIÁVEIS DE AMBIENTE
# ======================================================================================

load_dotenv()


# ======================================================================================
# 3. CONFIGURAÇÕES GERAIS DO RAG
# ======================================================================================

DATA_FILE_PATH = os.getenv(
    "DATA_FILE_PATH",
    "./docs/eleven_madison_park_data.txt",
)

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

TOP_K = int(
    os.getenv(
        "TOP_K",
        "3",
    )
)

TEST_QUERY = os.getenv(
    "TEST_QUERY",
    "What kind of food does Eleven Madison Park serve?",
)


# ======================================================================================
# 4. ESCOLHA DO PROVIDER
# ======================================================================================


def escolher_provider() -> str:
    """Seleciona o provider de IA que será utilizado pelo pipeline.

    Valores aceitos em AI_PROVIDER:
        ask     -> mostra o menu no terminal;
        ollama  -> entra diretamente no modo local;
        openai  -> entra diretamente no modo cloud.
    """

    provider_env = os.getenv(
        "AI_PROVIDER",
        "ask",
    ).strip().lower()

    if provider_env in {"ollama", "openai"}:
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
        opcao = input("Digite 1 ou 2: ").strip()

        if opcao == "1":
            return "ollama"

        if opcao == "2":
            return "openai"

        print("Opção inválida. Digite 1 ou 2.")


# ======================================================================================
# 5. CRIAÇÃO DO LLM E DO MODELO DE EMBEDDINGS
# ======================================================================================


def criar_modelos(provider: str):
    """Cria o LLM e o modelo de embeddings conforme o provider escolhido.

    Retorno:
        tuple: (llm, embeddings)
    """

    # ----------------------------------------------------------------------------------
    # MODO LOCAL: OLLAMA + QWEN
    # ----------------------------------------------------------------------------------
    if provider == "ollama":
        base_url = os.getenv(
            "OLLAMA_BASE_URL",
            "http://localhost:11434",
        )

        # Modelo pequeno usado para gerar a resposta final.
        chat_model = os.getenv(
            "OLLAMA_CHAT_MODEL",
            "qwen3.5:0.8b",
        )

        # Modelo especializado em embeddings.
        embedding_model = os.getenv(
            "OLLAMA_EMBEDDING_MODEL",
            "qwen3-embedding:0.6b",
        )

        print()
        print("=" * 60)
        print("MODELO LOCAL")
        print("=" * 60)
        print("Provider....: Ollama")
        print(f"LLM.........: {chat_model}")
        print(f"Embeddings..: {embedding_model}")
        print(f"Servidor....: {base_url}")

        # reasoning=False é importante neste exemplo porque queremos que o Qwen
        # retorne diretamente a resposta final do RAG, sem separar a saída em
        # raciocínio/thinking e conteúdo final.
        llm = ChatOllama(
            model=chat_model,
            base_url=base_url,
            temperature=0,
            reasoning=False,
        )

        embeddings = OllamaEmbeddings(
            model=embedding_model,
            base_url=base_url,
        )

        return llm, embeddings

    # ----------------------------------------------------------------------------------
    # MODO CLOUD: OPENAI
    # ----------------------------------------------------------------------------------
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY não encontrada. "
                "Configure a variável no arquivo .env para utilizar o provider OpenAI."
            )

        # Modelo pequeno e de baixo custo para a geração das respostas.
        chat_model = os.getenv(
            "OPENAI_CHAT_MODEL",
            "gpt-5-nano",
        )

        embedding_model = os.getenv(
            "OPENAI_EMBEDDING_MODEL",
            "text-embedding-3-small",
        )

        print()
        print("=" * 60)
        print("MODELO OPENAI")
        print("=" * 60)
        print("Provider....: OpenAI")
        print(f"LLM.........: {chat_model}")
        print(f"Embeddings..: {embedding_model}")

        # A API key nunca é impressa no terminal.
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
        f"Provider inválido: {provider}. Use 'ollama' ou 'openai'."
    )


# ======================================================================================
# 6. CARREGAMENTO DO DOCUMENTO
# ======================================================================================


def carregar_documento() -> list[Document]:
    """Carrega o TXT configurado em DATA_FILE_PATH sem usar langchain-community.

    Para um arquivo de texto simples, pathlib já é suficiente. Depois criamos um
    Document do LangChain manualmente para preservar o formato esperado pelo splitter.
    """

    print()
    print("=" * 60)
    print("CARREGANDO DOCUMENTO")
    print("=" * 60)

    caminho = Path(DATA_FILE_PATH)

    print(f"Arquivo......: {caminho}")

    if not caminho.is_file():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {caminho}\n"
            "Verifique DATA_FILE_PATH no arquivo .env."
        )

    texto = caminho.read_text(
        encoding="utf-8",
    )

    if not texto.strip():
        raise ValueError(
            f"O arquivo está vazio: {caminho}"
        )

    documento = Document(
        page_content=texto,
        metadata={
            "source": str(caminho),
        },
    )

    raw_documents = [documento]

    print(f"Documentos...: {len(raw_documents)}")

    print()
    print("Prévia do documento:")
    print("-" * 60)
    print(texto[:500].strip())
    print("-" * 60)

    return raw_documents


# ======================================================================================
# 7. DIVISÃO DO DOCUMENTO EM CHUNKS
# ======================================================================================


def criar_chunks(raw_documents: list[Document]) -> list[Document]:
    """Divide os documentos em chunks menores para busca semântica."""

    print()
    print("=" * 60)
    print("CRIANDO CHUNKS")
    print("=" * 60)
    print(f"Chunk size...: {CHUNK_SIZE}")
    print(f"Overlap......: {CHUNK_OVERLAP}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    documents = text_splitter.split_documents(
        raw_documents,
    )

    if not documents:
        raise ValueError(
            "A divisão do documento resultou em zero chunks. "
            "Verifique o arquivo e as configurações do splitter."
        )

    print(f"Chunks.......: {len(documents)}")

    # Mostra o terceiro chunk quando ele existir. Caso contrário, usa o primeiro.
    example_index = 2 if len(documents) >= 3 else 0
    example_document = documents[example_index]

    print()
    print(f"Exemplo do chunk {example_index + 1}:")
    print("-" * 60)
    print(example_document.page_content[:700].strip())
    print("-" * 60)
    print(f"Metadata.....: {example_document.metadata}")

    return documents


# ======================================================================================
# 8. CRIAÇÃO DO VECTOR STORE
# ======================================================================================


def criar_vector_store(
    documents: list[Document],
    embeddings,
):
    """Gera embeddings dos chunks e armazena os vetores no ChromaDB."""

    print()
    print("=" * 60)
    print("CRIANDO EMBEDDINGS + CHROMADB")
    print("=" * 60)
    print("Gerando embeddings dos chunks...")

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
    )

    # _collection é um atributo interno. Aqui é usado apenas para inspeção didática.
    vector_count = vector_store._collection.count()

    print(f"Vetores......: {vector_count}")

    if vector_count == 0:
        raise ValueError(
            "O ChromaDB foi criado, mas nenhum vetor foi armazenado."
        )

    return vector_store


# ======================================================================================
# 9. INSPEÇÃO DE UM EMBEDDING
# ======================================================================================


def inspecionar_embedding(vector_store) -> None:
    """Mostra texto, primeiras posições e dimensão de um embedding armazenado."""

    stored_data = vector_store._collection.get(
        include=["embeddings", "documents"],
        limit=1,
    )

    documentos = stored_data.get("documents") or []
    embeddings = stored_data.get("embeddings")

    if not documentos or embeddings is None or len(embeddings) == 0:
        print("Não foi possível inspecionar o primeiro embedding.")
        return

    first_embedding = embeddings[0]

    # Converte explicitamente para float do Python para não exibir np.float64(...).
    primeiros_valores = [
        round(float(valor), 6)
        for valor in first_embedding[:10]
    ]

    print()
    print("=" * 60)
    print("INSPEÇÃO DO PRIMEIRO EMBEDDING")
    print("=" * 60)
    print("Texto:")
    print(documentos[0][:300].strip())
    print()
    print(f"Primeiros valores: {primeiros_valores}")
    print(f"Dimensões........: {len(first_embedding)}")


# ======================================================================================
# 10. CONFIGURAÇÃO DO RETRIEVER
# ======================================================================================


def criar_retriever(vector_store):
    """Transforma o ChromaDB em um retriever configurado com TOP_K."""

    retriever = vector_store.as_retriever(
        search_kwargs={
            "k": TOP_K,
        }
    )

    print()
    print("=" * 60)
    print("RETRIEVER CONFIGURADO")
    print("=" * 60)
    print(f"Top K........: {TOP_K}")

    return retriever


# ======================================================================================
# 11. PROMPT DO RAG
# ======================================================================================

# Em vez de usar RetrievalQAWithSourcesChain, montamos explicitamente o pipeline:
#
#   pergunta -> retriever -> chunks -> contexto -> prompt -> LLM -> resposta
#
# Isso facilita a depuração e funciona igualmente com ChatOllama e ChatOpenAI.
RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Você é um assistente de perguntas e respostas baseado em documentos. "
            "Responda usando somente as informações presentes no contexto fornecido. "
            "Não utilize conhecimento externo e não invente informações. "
            "Se a resposta não estiver presente no contexto, diga exatamente: "
            "'Não encontrei essa informação nos documentos.' "
            "Responda de forma direta e no mesmo idioma da pergunta.",
        ),
        (
            "human",
            "CONTEXTO:\n\n{contexto}\n\n"
            "PERGUNTA:\n{pergunta}\n\n"
            "RESPOSTA:",
        ),
    ]
)


# ======================================================================================
# 12. NORMALIZAÇÃO DA RESPOSTA DO MODELO
# ======================================================================================


def extrair_texto_resposta(resposta) -> str:
    """Extrai texto de uma resposta LangChain de forma defensiva.

    Normalmente response.content é uma string. Algumas integrações podem retornar
    uma lista de blocos. Esta função cobre os dois formatos.
    """

    content = getattr(resposta, "content", "")

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        partes = []

        for item in content:
            if isinstance(item, str):
                partes.append(item)
                continue

            if isinstance(item, dict):
                texto = item.get("text") or item.get("content")

                if texto:
                    partes.append(str(texto))

        return "\n".join(partes).strip()

    return str(content).strip() if content else ""


# ======================================================================================
# 13. EXECUÇÃO DE UMA PERGUNTA
# ======================================================================================


def executar_pergunta(
    retriever,
    llm,
    pergunta: str,
) -> dict:
    """Executa o RAG manualmente e retorna resposta + chunks utilizados."""

    pergunta = pergunta.strip()

    if not pergunta:
        raise ValueError("A pergunta não pode estar vazia.")

    print()
    print("=" * 60)
    print("PERGUNTA")
    print("=" * 60)
    print(pergunta)

    # ------------------------------------------------------------------------------
    # 1. Retrieval
    # ------------------------------------------------------------------------------
    source_documents = retriever.invoke(
        pergunta,
    )

    if not source_documents:
        answer = "Não encontrei essa informação nos documentos."

        print()
        print("=" * 60)
        print("RESPOSTA")
        print("=" * 60)
        print(answer)

        return {
            "answer": answer,
            "source_documents": [],
        }

    # ------------------------------------------------------------------------------
    # 2. Contexto
    # ------------------------------------------------------------------------------
    # Cada chunk recebe um número e sua origem para facilitar inspeção e rastreabilidade.
    blocos_contexto = []

    for index, document in enumerate(
        source_documents,
        start=1,
    ):
        source = document.metadata.get(
            "source",
            "Fonte desconhecida",
        )

        blocos_contexto.append(
            f"[CHUNK {index}]\n"
            f"Fonte: {source}\n"
            f"{document.page_content.strip()}"
        )

    contexto = "\n\n".join(
        blocos_contexto,
    )

    # ------------------------------------------------------------------------------
    # 3. Prompt
    # ------------------------------------------------------------------------------
    mensagens = RAG_PROMPT.format_messages(
        contexto=contexto,
        pergunta=pergunta,
    )

    # ------------------------------------------------------------------------------
    # 4. Geração da resposta
    # ------------------------------------------------------------------------------
    resposta_modelo = llm.invoke(
        mensagens,
    )

    answer = extrair_texto_resposta(
        resposta_modelo,
    )

    # Se o provider responder sem conteúdo, tratamos como erro explícito.
    # Isso é preferível a imprimir uma seção "RESPOSTA" vazia.
    if not answer:
        raise RuntimeError(
            "O LLM retornou uma resposta vazia. "
            "Se estiver usando Ollama/Qwen, confirme que o modelo está atualizado "
            "e que ChatOllama está configurado com reasoning=False."
        )

    # ------------------------------------------------------------------------------
    # 5. Saída
    # ------------------------------------------------------------------------------
    print()
    print("=" * 60)
    print("RESPOSTA")
    print("=" * 60)
    print(answer)

    print()
    print("=" * 60)
    print("CHUNKS UTILIZADOS COMO FONTE")
    print("=" * 60)

    for index, document in enumerate(
        source_documents,
        start=1,
    ):
        source = document.metadata.get(
            "source",
            "Fonte desconhecida",
        )

        snippet = document.page_content[:350].strip()

        print()
        print(f"[{index}] {source}")
        print("-" * 60)
        print(snippet)
        print("...")

    return {
        "answer": answer,
        "source_documents": source_documents,
    }


# ======================================================================================
# 14. CHAT INTERATIVO OPCIONAL
# ======================================================================================


def iniciar_chat(retriever, llm) -> None:
    """Permite continuar fazendo perguntas até o usuário digitar 'sair'."""

    print()
    print("=" * 60)
    print("CHAT RAG")
    print("=" * 60)
    print("Digite uma pergunta ou 'sair' para encerrar.")

    while True:
        pergunta = input("\nVocê: ").strip()

        if pergunta.lower() in {"sair", "exit", "quit"}:
            print("\nAplicação encerrada.")
            break

        if not pergunta:
            continue

        try:
            executar_pergunta(
                retriever,
                llm,
                pergunta,
            )
        except Exception as error:
            print()
            print("=" * 60)
            print("ERRO DURANTE A PERGUNTA")
            print("=" * 60)
            print(f"Tipo.........: {type(error).__name__}")
            print(f"Mensagem.....: {error}")


# ======================================================================================
# 15. FUNÇÃO PRINCIPAL
# ======================================================================================


def main() -> None:
    """Orquestra todas as etapas do pipeline RAG."""

    print()
    print("=" * 60)
    print("RAG - OLLAMA/QWEN OU OPENAI")
    print("=" * 60)

    try:
        # 1. Carrega e divide o documento.
        raw_documents = carregar_documento()
        documents = criar_chunks(
            raw_documents,
        )

        # 2. Seleciona provider.
        provider = escolher_provider()

        print()
        print(f"Provider selecionado: {provider.upper()}")

        # 3. Cria LLM + embeddings.
        llm, embeddings = criar_modelos(
            provider,
        )

        # 4. Gera embeddings e constrói ChromaDB.
        vector_store = criar_vector_store(
            documents,
            embeddings,
        )

        # 5. Inspeção didática do primeiro vetor.
        inspecionar_embedding(
            vector_store,
        )

        # 6. Configura retriever.
        retriever = criar_retriever(
            vector_store,
        )

        # 7. Executa a pergunta de teste definida no .env.
        executar_pergunta(
            retriever,
            llm,
            TEST_QUERY,
        )

        # 8. Se CHAT_MODE=true, mantém um chat aberto após a pergunta de teste.
        chat_mode = os.getenv(
            "CHAT_MODE",
            "false",
        ).strip().lower() in {
            "1",
            "true",
            "yes",
            "sim",
        }

        if chat_mode:
            iniciar_chat(
                retriever,
                llm,
            )

    except Exception as error:
        print()
        print("=" * 60)
        print("ERRO DURANTE A EXECUÇÃO DO RAG")
        print("=" * 60)
        print(f"Tipo.........: {type(error).__name__}")
        print(f"Mensagem.....: {error}")


# ======================================================================================
# 16. PONTO DE ENTRADA
# ======================================================================================

if __name__ == "__main__":
    main()
