FROM python:3.12-alpine

# Instalar dependências do sistema
RUN apk add --no-cache \
    gcc \
    musl-dev \
    libffi-dev \
    build-base \
    curl-dev

# Diretório de trabalho
WORKDIR /app

# Copiar arquivos
COPY . .

# Instalar dependências Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Expor a porta da API
EXPOSE 8000

# Comando padrão
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
