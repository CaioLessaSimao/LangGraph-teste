from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing_extensions import TypedDict
import os
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from pypdf import PdfReader  # <--- Aqui é a mudança principal

load_dotenv()

class MessagesState(MessagesState):
    pass

# Ferramentas matemáticas
def multiply(a: int, b: int) -> int:
    return a * b

def add(a: int, b: int) -> int:
    return a + b

def divide(a: int, b: int) -> float:
    return a / b

# ✅ Ferramenta para ler PDF usando PyPDF
def ler_pdf(caminho_arquivo: str) -> str:
    """Lê o conteúdo de um arquivo PDF usando PyPDF e retorna como texto."""
    try:
        reader = PdfReader(caminho_arquivo)
        texto = ""
        for pagina in reader.pages:
            texto += pagina.extract_text() or ""
        return texto.strip() if texto else "O PDF está vazio ou não foi possível extrair o texto."
    except Exception as e:
        return f"Erro ao ler o PDF: {str(e)}"

# Todas as ferramentas disponíveis
tools = [ler_pdf]

# Mensagem do sistema
sys_msg = SystemMessage(content="""
Você é um assistente útil. Para responder a perguntas envolvendo arquivos PDF, use a ferramenta 'ler_pdf', que pode ler diretamente o conteúdo de arquivos no sistema local. 
Se o usuário pedir para analisar um arquivo PDF, **NÃO diga que você não tem acesso ao sistema de arquivos**. Em vez disso, use a ferramenta 'ler_pdf'.
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
messages = [HumanMessage(content="Use a ferramenta ler_pdf para me mostrar o conteúdo completo e bruto do arquivo './Requisitos.pdf', sem interpretar ou resumir.")]
messages = react_graph_memory.invoke({"messages": messages}, config)

for m in messages['messages']:
    m.pretty_print()