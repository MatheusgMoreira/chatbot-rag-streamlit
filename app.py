__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import gdown
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
import pandas as pd
import zipfile
import os

@st.cache_resource
def setup_database():
    PERSIST_DIRECTORY = 'chroma_db_gemini'
    DB_ZIP_URL = "https://github.com/MatheusgMoreira/chatbot-rag-streamlit/releases/download/v1.0.0/chroma_db_gemini.zip"
    
    if not os.path.exists(PERSIST_DIRECTORY):
        output_zip_path = "chroma_db_gemini.zip" # Nome do arquivo que será salvo localmente
        
        try:
            # 1. Usar gdown para baixar o arquivo
            gdown.download(url=DB_ZIP_URL, output=output_zip_path, quiet=False)
            
            # 2. Descompactar o arquivo baixado
            with zipfile.ZipFile(output_zip_path, 'r') as zip_ref:
                zip_ref.extractall() # Extrai para a pasta raiz
            
            # 3. Remover o arquivo .zip após a extração
            os.remove(output_zip_path)
            
        except Exception as e:
            st.error(f"Falha ao baixar ou descompactar a base de dados: {e}")
            return False
            
    return True

# Chame a função de setup no início da sua aplicação
is_db_ready = setup_database()

# O resto do seu app.py continua aqui, mas só executa se a DB estiver pronta
if is_db_ready:
    def load_resources():
        print("Executando load_resources() com Gemini...")
        load_dotenv()
        
        if not os.getenv("GOOGLE_API_KEY"):
            st.error("Chave da API do Google não encontrada. Verifique seu arquivo .env.")
            return None, None, None

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.0)
        
        persist_directory = 'chroma_db_gemini'
        
        try:
            vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
            
            # Pega todos os metadados para popular os filtros dinamicamente
            all_docs = vectordb.get()
            
            if not all_docs or not all_docs['metadatas']:
                return llm, vectordb, {}

            metadata_list = [doc for doc in all_docs['metadatas']]
            df_meta = pd.DataFrame(metadata_list)
            
            # Define as colunas que queremos usar como filtro
            filter_columns = [
                
        ]
            
            filter_options = {}
            for col in filter_columns:
                if col in df_meta.columns:
                    # Pega valores únicos, remove nulos e ordena
                    unique_values = df_meta[col].dropna().unique().tolist()
                    filter_options[col] = ['Todos'] + sorted(unique_values)
                else:
                    # Se a coluna não existir nos metadados, cria uma lista vazia
                    filter_options[col] = ['Todos']
                    
        except Exception as e:
            st.error(f"Erro ao carregar o banco de dados de '{persist_directory}': {e}")
            return None, None, None
            
        print("Recursos do Gemini e opções de filtro carregados.")
        return llm, vectordb, filter_options

    # --- Interface ---

    st.title("Chat de Análise de Chamados")

    llm, vectordb, filter_options = load_resources()

    if llm and vectordb:
        with st.sidebar:
            st.header("Filtros de Busca")
            
            # Cria um dicionário para guardar as seleções do usuário
            selected_filters = {}
            
            # Loop para criar os widgets de filtro dinamicamente
            for key, options in filter_options.items():
                # Formata o rótulo para ficar mais amigável
                label = f"Filtrar por {key.replace('_', ' ').title()}:"
                selected_filters[key] = st.selectbox(label, options=options)

        # Constrói a lista de condições de filtro
        filter_conditions = []
        for key, value in selected_filters.items():
            if value != 'Todos':
                filter_conditions.append({key: {'$eq': value}})

        # Monta o dicionário de filtro final no formato que o ChromaDB espera
        final_filter = {}
        if len(filter_conditions) > 1:
            final_filter['$and'] = filter_conditions
        elif len(filter_conditions) == 1:
            final_filter = filter_conditions[0]

        # Define os argumentos da busca, incluindo o filtro apenas se ele não estiver vazio
        search_kwargs = {'k': 5}
        if final_filter:
            search_kwargs['filter'] = final_filter

        retriever = vectordb.as_retriever(search_kwargs=search_kwargs)

        prompt_template = """
            # PERSONA
            Você é um Analista Técnico Sênior e sua tarefa é interpretar os documentos de suporte fornecidos para responder à pergunta do usuário.

            # REGRA PRINCIPAL
            - Sua única fonte da verdade é o 'Contexto' fornecido abaixo. Responda APENAS com base nessas informações.
            - É estritamente proibido usar qualquer conhecimento externo ou fazer suposições.

            # INSTRUÇÕES
            1.  **Seja Direto:** Responda à pergunta de forma objetiva, sem adicionar informações desnecessárias.
            2.  **Cite a Fonte:** Se possível, extraia trechos curtos do contexto para justificar sua resposta.
            3.  **Lide com Informação Ausente:** Se a resposta não estiver explicitamente no contexto, responda de forma clara e definitiva: "Com base nas informações fornecidas, não encontrei uma resposta para sua pergunta."

            Contexto: {contexto}
            Pergunta: {pergunta}
            Resposta:
            """
        custom_rag_prompt = PromptTemplate.from_template(prompt_template)
        
        chain = ({"contexto": retriever, "pergunta": RunnablePassthrough()} | custom_rag_prompt | llm | StrOutputParser())

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Qual sua pergunta sobre os chamados?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Analisando documentos e pensando..."):
                    try:
                        resposta_final = chain.invoke(prompt)
                        st.markdown(resposta_final)
                        st.session_state.messages.append({"role": "assistant", "content": resposta_final})
                    except Exception as e:
                        erro_msg = f"Ocorreu um erro ao processar sua pergunta: {e}"
                        st.error(erro_msg)
                        st.session_state.messages.append({"role": "assistant", "content": erro_msg})

    else:
        st.info("A aplicação não pôde ser iniciada. Verifique as configurações da API e do banco de dados.")