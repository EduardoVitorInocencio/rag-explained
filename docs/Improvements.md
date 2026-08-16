# Pipeline RAG com LangChain, ChromaDB e Streamlit

Este projeto demonstra um fluxo de **Retrieval-Augmented Generation (RAG)** sobre um arquivo de texto. O documento é carregado, dividido em chunks, convertido em embeddings, armazenado no ChromaDB e consultado por meio de um retriever antes da geração da resposta.

## Arquivos entregues

```text
rag_langchain_streamlit/
├── rag_pipeline_comentado.py
├── streamlit_app_example.py
├── .env.example
├── .gitignore
└── README.md
```

- `rag_pipeline_comentado.py`: código original com comentários reorganizados e ampliados, sem mudança intencional da lógica.
- `streamlit_app_example.py`: exemplo independente de interface de chat com upload de arquivo TXT.
- `.env.example`: modelo de configuração das variáveis de ambiente.
- `.gitignore`: arquivos que não devem ser versionados.
- `README.md`: instalação, execução, arquitetura sugerida e próximos passos.

---

## 1. Fluxo atual

O pipeline executa estas etapas:

1. Carrega as variáveis do arquivo `.env`.
2. Lê o arquivo `eleven_madison_park_data.txt`.
3. Converte o conteúdo em documentos LangChain.
4. Divide o texto em chunks com sobreposição.
5. Gera embeddings para cada chunk.
6. Armazena os documentos no ChromaDB.
7. Testa uma pesquisa por similaridade.
8. Converte o vector store em um retriever.
9. Cria uma chain de perguntas e respostas com fontes.
10. Executa uma pergunta de teste.

A documentação atual do LangChain apresenta loaders, splitters, embeddings, vector stores e retrievers como componentes substituíveis de um pipeline de recuperação. Essa separação é a principal base para tornar o projeto flexível.

---

## 2. Pontos de atenção identificados

### 2.1 Conflito entre classes chamadas `OpenAI`

O código possui estes imports:

```python
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings, OpenAI
```

O segundo import redefine o nome `OpenAI`. Portanto, a linha:

```python
openai_client = OpenAI(api_key=openai_api_key)
```

não utiliza necessariamente a classe importada do SDK oficial na primeira linha.

O arquivo comentado apenas documenta esse comportamento. Uma refatoração futura deveria usar aliases explícitos, por exemplo:

```python
from openai import OpenAI as OpenAIClient
from langchain_openai import OpenAI as LangChainOpenAI
```

### 2.2 Exibição parcial da chave

O trecho abaixo expõe parte de uma credencial:

```python
print(openai_api_key[:15])
```

Isso não deve ser mantido em produção, logs compartilhados, notebooks publicados ou capturas de tela.

### 2.3 Falta de validação da variável de ambiente

Caso `OPENAI_API_KEY` não esteja configurada, o fatiamento `openai_api_key[:15]` causará erro antes de uma mensagem clara para o usuário.

### 2.4 Acesso a atributo interno do Chroma

O código utiliza:

```python
vector_store._collection
```

O prefixo `_` indica uma implementação interna. Ela pode mudar entre versões sem garantia de compatibilidade pública.

### 2.5 Acesso fixo ao terceiro chunk

O trecho abaixo exige pelo menos três chunks:

```python
documents[2]
```

Arquivos pequenos podem produzir menos chunks e gerar `IndexError`.

### 2.6 Temperatura alta para RAG factual

O código usa:

```python
temperature=1.3
```

Temperaturas altas aumentam a variabilidade. Para uma aplicação que precisa responder estritamente com base em documentos, normalmente é preferível testar valores mais baixos.

### 2.7 Vector store não persistente

Sem `persist_directory`, a base é reconstruída sempre que o processo é reiniciado. Isso aumenta tempo e consumo da API quando o mesmo documento é usado repetidamente.

### 2.8 Chain clássica

`RetrievalQAWithSourcesChain` permanece disponível no pacote `langchain-classic`, mas novas versões do projeto podem avaliar as APIs modernas de RAG e a composição por Runnables. A migração deve ser tratada em uma etapa separada para não misturar atualização de dependências com reorganização estrutural.

---

## 3. Instalação com `uv`

### 3.1 Criar a pasta

```bash
mkdir rag-langchain-streamlit
cd rag-langchain-streamlit
```

Copie os arquivos deste pacote para a pasta.

### 3.2 Inicializar o projeto

```bash
uv init
```

O `uv` gerencia dependências no `pyproject.toml`, cria o ambiente virtual em `.venv` e registra versões resolvidas no arquivo `uv.lock`.

### 3.3 Adicionar as dependências

```bash
uv add openai \
  python-dotenv \
  langchain-openai \
  langchain-chroma \
  langchain-text-splitters \
  langchain-community \
  langchain-classic \
  chromadb \
  streamlit
```

No PowerShell, o comando também pode ser executado em uma única linha:

```powershell
uv add openai python-dotenv langchain-openai langchain-chroma langchain-text-splitters langchain-community langchain-classic chromadb streamlit
```

### 3.4 Dependências de desenvolvimento opcionais

```bash
uv add --dev pytest ruff mypy
```

Responsabilidades:

- `pytest`: testes automatizados;
- `ruff`: análise estática e formatação;
- `mypy`: verificação de tipos.

### 3.5 Sincronizar o ambiente

```bash
uv sync
```

---

## 4. Configuração do ambiente

Crie o arquivo `.env` a partir do exemplo:

### Windows PowerShell

```powershell
Copy-Item .env.example .env
```

### Linux ou macOS

```bash
cp .env.example .env
```

Preencha:

```dotenv
OPENAI_API_KEY=sua-chave
OPENAI_CHAT_MODEL=nome-do-modelo-de-chat
OPENAI_EMBEDDING_MODEL=nome-do-modelo-de-embeddings
```

Os nomes de modelos podem variar conforme disponibilidade, conta e versão da API. Mantenha-os em variáveis de ambiente para evitar alterações no código.

Nunca envie o `.env` para o Git. O `.gitignore` fornecido já o exclui.

---

## 5. Executar o script comentado

Coloque o arquivo abaixo na raiz:

```text
eleven_madison_park_data.txt
```

Execute:

```bash
uv run python rag_pipeline_comentado.py
```

O script deverá:

1. carregar o arquivo;
2. imprimir uma amostra;
3. criar os chunks;
4. gerar os embeddings;
5. criar o ChromaDB em memória;
6. testar uma busca semântica;
7. executar uma pergunta completa.

---

## 6. Executar a interface Streamlit

Inicie a aplicação:

```bash
uv run streamlit run streamlit_app_example.py
```

Na interface:

1. confira os modelos configurados;
2. envie um arquivo `.txt`;
3. ajuste o tamanho dos chunks;
4. ajuste o overlap;
5. escolha quantos trechos serão recuperados;
6. faça perguntas no campo de chat;
7. expanda as fontes para inspecionar os chunks utilizados.

A interface usa:

- `st.file_uploader` para receber o documento;
- `st.chat_input` para receber perguntas;
- `st.chat_message` para mostrar a conversa;
- `st.session_state` para manter o histórico;
- `st.cache_resource` para evitar reconstruir o pipeline em cada interação equivalente.

---

## 7. Estrutura modular recomendada

Quando o projeto evoluir além da prova de conceito, uma estrutura possível é:

```text
rag-project/
├── pyproject.toml
├── uv.lock
├── .env.example
├── README.md
├── data/
│   ├── raw/
│   └── processed/
├── storage/
│   └── chroma/
├── src/
│   └── rag_app/
│       ├── __init__.py
│       ├── config/
│       │   ├── __init__.py
│       │   └── settings.py
│       ├── ingestion/
│       │   ├── __init__.py
│       │   ├── loaders.py
│       │   └── file_registry.py
│       ├── processing/
│       │   ├── __init__.py
│       │   ├── splitters.py
│       │   └── metadata.py
│       ├── embeddings/
│       │   ├── __init__.py
│       │   └── factory.py
│       ├── vectorstores/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   └── chroma_store.py
│       ├── retrieval/
│       │   ├── __init__.py
│       │   ├── retriever_factory.py
│       │   └── reranker.py
│       ├── chains/
│       │   ├── __init__.py
│       │   ├── prompts.py
│       │   └── qa_chain.py
│       ├── services/
│       │   ├── __init__.py
│       │   ├── indexing_service.py
│       │   └── rag_service.py
│       ├── interfaces/
│       │   ├── __init__.py
│       │   ├── cli.py
│       │   └── streamlit_app.py
│       ├── exceptions.py
│       └── logging_config.py
└── tests/
    ├── unit/
    └── integration/
```

### Responsabilidade dos módulos

| Módulo | Responsabilidade |
|---|---|
| `config/settings.py` | Centralizar variáveis, modelos, caminhos e parâmetros |
| `ingestion/loaders.py` | Carregar TXT, PDF, DOCX, HTML e outras fontes |
| `ingestion/file_registry.py` | Registrar hash, nome, data e status dos documentos |
| `processing/splitters.py` | Selecionar estratégia de chunking |
| `processing/metadata.py` | Padronizar metadados e identificadores |
| `embeddings/factory.py` | Selecionar OpenAI, Hugging Face, Ollama ou outro provider |
| `vectorstores/base.py` | Definir o contrato comum de armazenamento |
| `vectorstores/chroma_store.py` | Implementar persistência e consultas no Chroma |
| `retrieval/retriever_factory.py` | Configurar `k`, filtros e tipo de busca |
| `retrieval/reranker.py` | Reordenar os resultados recuperados |
| `chains/prompts.py` | Centralizar prompts e instruções |
| `chains/qa_chain.py` | Montar a chain de resposta |
| `services/indexing_service.py` | Orquestrar carregamento, chunks e indexação |
| `services/rag_service.py` | Orquestrar perguntas, recuperação e resposta |
| `interfaces/cli.py` | Expor o serviço pelo terminal |
| `interfaces/streamlit_app.py` | Expor o mesmo serviço pela interface web |
| `exceptions.py` | Definir erros específicos do domínio |
| `logging_config.py` | Padronizar logs e níveis de diagnóstico |

---

## 8. Princípio central de flexibilidade

A interface não deve conhecer detalhes do Chroma, do modelo ou do loader. Ela deveria depender somente de um serviço:

```python
answer = rag_service.ask(question)
```

O `RagService` recebe suas dependências na inicialização:

```python
rag_service = RagService(
    loader=loader,
    splitter=splitter,
    vector_store=vector_store,
    retriever=retriever,
    llm=llm,
)
```

Com isso, é possível trocar:

- Chroma por Pinecone, Qdrant, FAISS, PostgreSQL/pgvector ou outro banco;
- OpenAI Embeddings por modelos locais;
- TXT por PDF, DOCX, Markdown, HTML ou páginas web;
- Streamlit por FastAPI, CLI, desktop ou bot;
- uma chain clássica por Runnables, agentes ou LangGraph.

---

## 9. Módulos adicionais úteis

### Configuração

```bash
uv add pydantic-settings
```

Permite validar configurações, valores obrigatórios e tipos.

### Processamento por tokens

```bash
uv add tiktoken
```

Útil para limitar chunks por tokens em vez de apenas caracteres.

### Novos formatos

```bash
uv add pypdf docx2txt beautifulsoup4
```

Possibilita evoluir loaders para PDF, Word e HTML.

### Tentativas e resiliência

```bash
uv add tenacity
```

Permite implementar retries com espera progressiva para falhas temporárias.

### Observabilidade

Avalie callbacks, métricas de latência, quantidade de tokens, documentos recuperados e custo por consulta. Em ambientes reais, não registre chaves, embeddings completos ou documentos confidenciais sem uma política adequada.

---

## 10. Próximas etapas sugeridas

### Etapa 1 — Organização

- mover parâmetros para `settings.py`;
- criar funções pequenas para cada etapa;
- substituir `print` por `logging`;
- criar erros específicos;
- remover acesso direto a atributos privados.

### Etapa 2 — Persistência

- configurar `persist_directory`;
- usar identificadores de coleção;
- evitar reindexar documentos já processados;
- calcular hash do arquivo;
- registrar versão do embedding.

### Etapa 3 — Qualidade da recuperação

- testar diferentes `chunk_size` e `chunk_overlap`;
- adicionar metadados;
- aplicar filtros;
- comparar busca por similaridade e MMR;
- avaliar reranking;
- criar um conjunto de perguntas esperadas.

### Etapa 4 — Interface

- histórico por sessão;
- múltiplos documentos;
- indicador de processamento;
- botão para excluir uma base;
- visualização das fontes;
- exportação da conversa;
- autenticação, quando necessária.

### Etapa 5 — Testes

- teste do loader;
- teste do splitter;
- teste de configuração ausente;
- teste de documento vazio;
- teste do retriever com embeddings simulados;
- teste do serviço RAG sem chamada real à API;
- teste de integração da interface com uma base pequena.

---

## 11. Referências oficiais

- LangChain — Retrieval:  
  https://docs.langchain.com/oss/python/langchain/retrieval

- LangChain — Text splitters:  
  https://docs.langchain.com/oss/python/integrations/splitters

- LangChain — RecursiveCharacterTextSplitter:  
  https://docs.langchain.com/oss/python/integrations/splitters/recursive_text_splitter

- LangChain — Document loaders:  
  https://docs.langchain.com/oss/python/integrations/document_loaders

- LangChain Reference — RetrievalQAWithSourcesChain:  
  https://reference.langchain.com/python/langchain-classic/chains/qa_with_sources/retrieval/RetrievalQAWithSourcesChain

- Streamlit — Chat elements:  
  https://docs.streamlit.io/develop/api-reference/chat

- uv — Projects:  
  https://docs.astral.sh/uv/guides/projects/

- uv — Managing dependencies:  
  https://docs.astral.sh/uv/concepts/projects/dependencies/
