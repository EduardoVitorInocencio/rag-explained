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

# Obtém a chave da OpenAI armazenada na variável de ambiente `OPENAI_API_KEY`.
# O arquivo `.env` deve conter uma linha semelhante a:
#
#     OPENAI_API_KEY=sua-chave-aqui
#
# Atenção: caso a variável não exista, `openai_api_key` receberá `None`.


# Inicializa o objeto chamado `OpenAI` usando a chave carregada.
#
# IMPORTANTE:
# Devido ao conflito de imports explicado no início do arquivo, neste ponto o nome
# `OpenAI` corresponde à última classe importada com esse nome.
openai_client = OpenAI(api_key=openai_api_key)
print("OpenAI client successfully configured.")

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

# Inicializa o divisor recursivo.
#
# Parâmetros:
#   - chunk_size=1000:
#       tenta limitar cada trecho a aproximadamente 1.000 caracteres;
#
#   - chunk_overlap=150:
#       repete aproximadamente 150 caracteres entre chunks consecutivos.
#
# O overlap ajuda a reduzir a perda de contexto quando uma informação está localizada
# exatamente no limite entre dois trechos.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
)

# Divide os documentos carregados em vários objetos `Document` menores.
# Os metadados do documento original são preservados nos chunks.
documents = text_splitter.split_documents(raw_documents)

# Interrompe a execução caso nenhum chunk tenha sido criado.
#
# Essa validação evita que as etapas de embeddings e armazenamento vetorial sejam
# executadas com uma coleção vazia.
if not documents:
    raise ValueError(
        "Error: Splitting resulted in zero documents. "
        "Check the input file and splitter settings."
    )

print(f"Document split into {len(documents)} chunks.")

# Exibe o terceiro chunk da lista como exemplo.
#
# ATENÇÃO:
# O índice `2` representa o terceiro item. Caso o documento produza menos de três
# chunks, esta linha poderá gerar `IndexError`. O comportamento foi mantido para
# preservar o código original.
print("\n--- Example Chunk (Chunk 2) ---")
print(documents[2].page_content)

# Exibe os metadados associados ao mesmo chunk.
# Para o TextLoader, normalmente será apresentado o caminho em `source`.
print("\n--- Metadata for Chunk 2 ---")
print(documents[2].metadata)


# ======================================================================================
# 4. INICIALIZAÇÃO DO MODELO DE EMBEDDINGS
# ======================================================================================

# Embeddings são representações vetoriais do texto.
# Textos semanticamente semelhantes tendem a produzir vetores próximos no espaço
# multidimensional, permitindo realizar pesquisas por similaridade.
print("Initializing OpenAI Embeddings model...")

# Cria a integração de embeddings utilizando a chave configurada anteriormente.
#
# O LangChain utiliza este componente posteriormente para:
#   1. vetorizar cada chunk;
#   2. vetorizar a pergunta;
#   3. comparar a pergunta com os chunks armazenados.
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

print("OpenAI Embeddings model initialized.")


# ======================================================================================
# 5. CRIAÇÃO DO VECTOR STORE NO CHROMADB
# ======================================================================================

# O vector store armazena os vetores e os documentos associados.
# Durante a criação, cada chunk será enviado ao modelo de embeddings.
print("\nCreating ChromaDB vector store and embedding documents...")

# Cria uma coleção Chroma em memória a partir dos documentos.
#
# Entradas:
#   - documents: lista de chunks;
#   - embedding: modelo usado para gerar os vetores.
#
# Como não foi informado `persist_directory`, os dados não são explicitamente
# configurados aqui para persistência em disco.
vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
)

# Consulta a quantidade de elementos diretamente na coleção interna do Chroma.
#
# ATENÇÃO:
# `_collection` é um atributo interno da implementação. Acessos iniciados por
# sublinhado são considerados detalhes internos e podem mudar entre versões.
# Esta abordagem foi mantida porque pertence ao código original.
vector_count = vector_store._collection.count()
print(f"ChromaDB vector store created with {vector_count} items.")

# Garante que o banco vetorial não foi criado vazio.
if vector_count == 0:
    raise ValueError(
        "Vector store creation resulted in 0 items. Check previous steps."
    )


# ======================================================================================
# 6. INSPEÇÃO DOS DADOS ARMAZENADOS NO CHROMA
# ======================================================================================

# Recupera um item da coleção para fins educacionais e de diagnóstico.
#
# O parâmetro `include` solicita:
#   - o texto original do chunk;
#   - o vetor de embedding gerado para esse texto.
stored_data = vector_store._collection.get(
    include=["embeddings", "documents"],
    limit=1,
)

# Exibe o conteúdo textual do primeiro registro armazenado.
print("First chunk text:\n", stored_data["documents"][0])

# Exibe o embedding completo.
#
# Observação:
# Vetores de embeddings normalmente possuem muitas dimensões, tornando essa saída
# extensa e pouco adequada para logs de produção.
print("\nEmbedding vector:\n", stored_data["embeddings"][0])

# Exibe a quantidade de dimensões do vetor.
print(
    f"\nFull embedding has "
    f"{len(stored_data['embeddings'][0])} dimensions."
)


# ======================================================================================
# 7. TESTE DE BUSCA POR SIMILARIDADE
# ======================================================================================

# Antes de criar a chain completa, esta etapa valida se o vector store consegue
# localizar chunks semanticamente relacionados a uma pergunta.
print("\n--- Testing Similarity Search in Vector Store ---")

# Pergunta usada apenas para testar a recuperação vetorial.
test_query = "Who is Daniel Humm?"
print(f"Searching for documents similar to: '{test_query}'")

try:
    # Realiza a busca por similaridade.
    #
    # O parâmetro `k=3` solicita os três chunks considerados mais relevantes
    # para a pergunta informada.
    similar_docs = vector_store.similarity_search(test_query, k=3)
    print(f"\nFound {len(similar_docs)} similar documents:")

    # Percorre os documentos encontrados e exibe um resumo de cada resultado.
    for i, doc in enumerate(similar_docs):
        print(f"\n--- Document {i + 1} ---")

        # Limita a visualização aos primeiros 700 caracteres para evitar uma saída
        # muito extensa no console.
        content_snippet = doc.page_content[:700].strip() + "..."

        # Recupera o caminho de origem registrado nos metadados.
        # Caso não exista, utiliza o texto padrão `Unknown Source`.
        source = doc.metadata.get("source", "Unknown Source")

        print(f"Content Snippet: {content_snippet}")
        print(f"Source: {source}")

except Exception as e:
    # Captura erros ocorridos durante a busca e permite diagnosticar o problema.
    #
    # Em uma aplicação maior, recomenda-se substituir `print` pelo módulo `logging`
    # e tratar exceções mais específicas.
    print(f"An error occurred during similarity search: {e}")


# ======================================================================================
# 8. CONFIGURAÇÃO DO RETRIEVER
# ======================================================================================

# Converte o vector store em um retriever.
#
# O retriever oferece uma interface padronizada para recuperar documentos e permite
# que a implementação do armazenamento seja substituída futuramente.
#
# `search_kwargs={"k": 3}` configura o retorno dos três chunks mais relevantes.
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
print("Retriever configured successfully from vector store.")


# ======================================================================================
# 9. CONFIGURAÇÃO DO MODELO DE LINGUAGEM
# ======================================================================================

# Inicializa o modelo responsável por gerar a resposta final usando os chunks
# recuperados como contexto.
#
# A temperatura controla a variabilidade das respostas:
#   - valores menores tendem a produzir respostas mais determinísticas;
#   - valores maiores aumentam a diversidade e a criatividade.
#
# OBSERVAÇÃO:
# O comentário original indicava uma resposta mais factual, mas o valor 1.3 é
# relativamente alto para um fluxo de perguntas e respostas baseado em documentos.
# O valor foi mantido para não alterar o comportamento fornecido.
llm = OpenAI(
    temperature=1.3,
    openai_api_key=openai_api_key,
)

print("OpenAI LLM successfully initialized.")


# ======================================================================================
# 10. CRIAÇÃO DA CHAIN DE PERGUNTAS E RESPOSTAS COM FONTES
# ======================================================================================

# Cria a chain que conecta:
#   pergunta -> retriever -> documentos relevantes -> LLM -> resposta com fontes.
#
# Parâmetros principais:
#
#   - chain_type="stuff":
#       envia os documentos recuperados diretamente no contexto do modelo;
#
#   - retriever=retriever:
#       usa o recuperador configurado sobre o ChromaDB;
#
#   - return_source_documents=True:
#       inclui no resultado os objetos `Document` realmente utilizados;
#
#   - verbose=True:
#       exibe detalhes internos da execução no console.
#
# O tipo `stuff` é simples e adequado quando os chunks recuperados cabem no limite
# de contexto do modelo. Estratégias diferentes podem ser avaliadas para coleções
# maiores.
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
