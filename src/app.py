from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import os
import tempfile
import shutil
from typing import Optional
import time
from contextlib import asynccontextmanager

# Importar a classe do agente
from agent import PDFOCRAgent

# Variável global para armazenar o agente
pdf_agent = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gerencia o ciclo de vida da aplicação.
    Executa código na inicialização (antes de yield) e finalização (após yield)
    """
    # Código de inicialização
    global pdf_agent
    try:
        pdf_agent = PDFOCRAgent()
        print("PDF OCR Agent inicializado com sucesso!")
    except Exception as e:
        print(f"Erro ao inicializar o agente: {e}")
        # Não interrompe a inicialização da API, permitindo diagnóstico via endpoints
    
    yield  # A aplicação executa aqui
    
    # Código de finalização (cleanup)
    # Se necessário, pode-se adicionar lógica de limpeza aqui
    print("Encerrando o serviço PDF OCR API...")

# Inicializar a aplicação FastAPI com o gerenciador de ciclo de vida
app = FastAPI(
    title="PDF OCR API", 
    description="API para processamento de PDFs com OCR usando Mistral AI",
    lifespan=lifespan
)

@app.get("/")
async def root():
    """Endpoint raiz para verificar se a API está funcionando"""
    return {"message": "PDF OCR API está funcionando! Envie um PDF para /process"}

@app.get("/health")
async def health_check():
    """Endpoint para verificar se o agente está inicializado corretamente"""
    global pdf_agent
    
    if pdf_agent is None:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "message": "O agente PDF OCR ainda não foi inicializado"}
        )
    
    # Verificar se as chaves de API necessárias estão disponíveis
    mistral_key = os.getenv("MISTRAL_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")
    
    if not mistral_key or not google_key:
        missing_keys = []
        if not mistral_key:
            missing_keys.append("MISTRAL_API_KEY")
        if not google_key:
            missing_keys.append("GOOGLE_API_KEY")
            
        return JSONResponse(
            status_code=503,
            content={
                "status": "config_error",  
                "message": f"Chaves de API ausentes: {', '.join(missing_keys)}"
            }
        )
    
    return {"status": "healthy", "message": "PDF OCR API está pronta para processar arquivos"}

@app.post("/process")
async def process_pdf(file: UploadFile = File(...)):
    """
    Processa um arquivo PDF com OCR
    
    Args:
        file: Arquivo PDF para processar
        query: Consulta específica sobre o conteúdo do PDF (opcional)
    """
    global pdf_agent
    query = "Extraia a data de validade deste PDF usando OCR e retorne somente ela e somente ela formatada como DD/MM/AAAA"
    
    # Verificar se o agente foi inicializado
    if pdf_agent is None:
        try:
            pdf_agent = PDFOCRAgent()
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Erro ao inicializar o agente: {str(e)}"
            )
    
    # Verificar se o arquivo é um PDF
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="O arquivo deve ser um PDF")
    
    try:
        # Criar um arquivo temporário para salvar o PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            # Copiar o conteúdo do arquivo para o arquivo temporário
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
        
        # Garantir que o arquivo foi fechado
        file.file.close()
        
        # Processar com o agente
        start_time = time.time()
        print(f"Iniciando processamento do arquivo: {file.filename}")
        
        response = pdf_agent.process_pdf(tmp_path, query)
        
        # Extrair resultados do agente
        all_messages = []
        for m in response['messages']:
            if hasattr(m, 'content'):
                all_messages.append({
                    "type": m.type,
                    "content": m.content
                })
        resposta = next(
                    (m.content for m in reversed(response['messages']) if hasattr(m, 'content')),
                    None  # valor padrão se nenhum tiver .content
                )

                
        processing_time = time.time() - start_time
        print(f"Processamento concluído em {processing_time:.2f} segundos")
        
        # Remover o arquivo temporário
        os.unlink(tmp_path)
        
        return JSONResponse(content={
            "filename": file.filename,
            "query": query,
            "processing_time_seconds": processing_time,
            "messages": resposta
        })
        
    except Exception as e:
        # Se ocorrer um erro, remover o arquivo temporário se ele existir
        if 'tmp_path' in locals():
            try:
                os.unlink(tmp_path)
            except:
                pass
        
        raise HTTPException(status_code=500, detail=f"Erro ao processar o PDF: {str(e)}")

if __name__ == "__main__":
    # Definir a porta a partir de uma variável de ambiente ou usar 8000 como padrão
    port = int(os.getenv("PORT", 8000))
    
    # Iniciar o servidor uvicorn
    print(f"Iniciando servidor na porta {port}...")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
