from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from mistralai import Mistral
import base64
from io import BytesIO
import os
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver


load_dotenv()

def ler_pdf_ocr(file_path: str) -> str:
        """
        Lê o conteúdo de um arquivo PDF usando a API oficial de OCR do Mistral AI.
        Utiliza a biblioteca mistralai para processar diretamente o PDF.
        """
        try:
            # Verificar se o arquivo existe
            if not os.path.exists(file_path):
                return f"Erro: O arquivo PDF '{file_path}' não foi encontrado."

            # Codificar o PDF para base64
            try:
                with open(file_path, "rb") as pdf_file:
                    base64_pdf = base64.b64encode(pdf_file.read()).decode('utf-8')
            except Exception as e:
                return f"Erro ao codificar o PDF: {str(e)}"

            # Verificar a chave da API Mistral
            API_KEY = os.getenv("MISTRAL_API_KEY")
            if not API_KEY:
                return "Erro: Chave da API Mistral não configurada. Defina a variável de ambiente MISTRAL_API_KEY."

            # Inicializar o cliente Mistral
            try:
                client = Mistral(api_key=API_KEY)
            except Exception as e:
                return f"Erro ao inicializar o cliente Mistral: {str(e)}"

            # Processar o PDF com OCR
            print(f"Processando o PDF '{file_path}' com OCR do Mistral...")
            try:
                ocr_response = client.ocr.process(
                    model="mistral-ocr-latest",  # Usar o modelo OCR mais recente do Mistral
                    document={
                        "type": "document_url",
                        "document_url": f"data:application/pdf;base64,{base64_pdf}"
                    }
                ) 
                return ocr_response
            except Exception as e:
                return f"Erro durante o processamento de OCR: {str(e)}"     
        except Exception as e:
            return f"Erro geral ao processar o PDF com OCR: {str(e)}"

class MessagesState(MessagesState):
    pass

class PDFOCRAgent:
    def __init__(self):
        self.tools = [ler_pdf_ocr]
        self.sys_msg = SystemMessage(content="""
        Você é um assistente útil. Para responder a perguntas envolvendo arquivos PDF, use a ferramenta 'ler_pdf_ocr', que pode ler diretamente o conteúdo de arquivos PDF no sistema local.
        Esta ferramenta utiliza exclusivamente a API oficial de OCR do Mistral AI para extrair texto de documentos PDF, incluindo PDFs digitalizados ou com imagens.
        Se o usuário pedir para analisar um arquivo PDF, **NÃO diga que você não tem acesso ao sistema de arquivos**. Em vez disso, use a ferramenta 'ler_pdf_ocr'.
        """)
        self.llm = None
        self.react_graph_memory = None
        self.initialize()
        
    def initialize(self):
        """Inicializa o agente com LLM e grafo de reação"""
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError("GOOGLE_API_KEY não está configurada")
            
        # Inicializa o LLM
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=google_api_key)
        llm_with_tools = self.llm.bind_tools(self.tools)
        
        # Função do assistente
        def assistant(state: MessagesState):
            return {"messages": [llm_with_tools.invoke([self.sys_msg] + state["messages"])]}
        
        # Construção do grafo
        builder = StateGraph(MessagesState)
        builder.add_node("assistant", assistant)
        builder.add_node("tools", ToolNode(self.tools))
        builder.add_edge(START, "assistant")
        builder.add_conditional_edges("assistant", tools_condition)
        builder.add_edge("tools", "assistant")
        
        memory = MemorySaver()
        self.react_graph_memory = builder.compile(checkpointer=memory)
        
        print("Agente inicializado com sucesso!")
    
    def process_pdf(self, pdf_path, query):
        """
        Processa um PDF usando o agente
        
        Args:
            pdf_path (str): Caminho para o arquivo PDF
            query (str): Consulta específica sobre o PDF
            
        Returns:
            dict: Resposta do agente com as mensagens
        """
        # Criar a mensagem para o agente
        message = f"Use a ferramenta ler_pdf_ocr para ler o conteudo do arquivo '{pdf_path}' e {query}."
        messages = [HumanMessage(content=message)]
        
        # Processar com o agente
        config = {"configurable": {"thread_id": f"thread_{int(__import__('time').time())}"}}
        return self.react_graph_memory.invoke({"messages": messages}, config)
