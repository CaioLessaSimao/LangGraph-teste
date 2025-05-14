from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing_extensions import TypedDict
import os
import base64
from io import BytesIO
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from mistralai import Mistral

load_dotenv()

class MessagesState(MessagesState):
    pass

# Função para ler PDF com OCR usando a API oficial do Mistral
def ler_pdf_ocr(caminho_arquivo: str) -> str:
    """
    Lê o conteúdo de um arquivo PDF usando a API oficial de OCR do Mistral AI.
    Utiliza a biblioteca mistralai para processar diretamente o PDF.
    """
    try:
        # Verificar se o arquivo existe
        if not os.path.exists(caminho_arquivo):
            return f"Erro: O arquivo PDF '{caminho_arquivo}' não foi encontrado."

        # Codificar o PDF para base64
        try:
            with open(caminho_arquivo, "rb") as pdf_file:
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
        print(f"Processando o PDF '{caminho_arquivo}' com OCR do Mistral...")
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

# Todas as ferramentas disponíveis
tools = [ler_pdf_ocr]

# Mensagem do sistema
sys_msg = SystemMessage(content="""
Você é um assistente útil. Para responder a perguntas envolvendo arquivos PDF, use a ferramenta 'ler_pdf_ocr', que pode ler diretamente o conteúdo de arquivos PDF no sistema local.
Esta ferramenta utiliza exclusivamente a API oficial de OCR do Mistral AI para extrair texto de documentos PDF, incluindo PDFs digitalizados ou com imagens.
Se o usuário pedir para analisar um arquivo PDF, **NÃO diga que você não tem acesso ao sistema de arquivos**. Em vez disso, use a ferramenta 'ler_pdf_ocr'.
""")

# Assistente com ferramentas
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=os.getenv("GOOGLE_API_KEY"))
llm_with_tools = llm.bind_tools(tools)

def assistant(state: MessagesState):
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

# Construção do grafo
builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")

memory = MemorySaver()
react_graph_memory = builder.compile(checkpointer=memory)

# Thread de conversa
config = {"configurable": {"thread_id": "1"}}

# 🧪 Exemplo de uso com um PDF
messages = [HumanMessage(content="Use a ferramenta ler_pdf_ocr para me mostrar o conteúdo completo do arquivo './Requisitos.pdf', realizando OCR em todas as páginas.")]
messages = react_graph_memory.invoke({"messages": messages}, config)

for m in messages['messages']:
    m.pretty_print()