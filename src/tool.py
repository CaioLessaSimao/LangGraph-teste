from mistralai import Mistral
import base64
from io import BytesIO
from dotenv import load_dotenv
load_dotenv()
class ReadPDFTool():

    def execute(file_path: str) -> str:
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