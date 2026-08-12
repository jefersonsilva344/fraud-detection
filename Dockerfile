FROM python:3.14-slim

# Evita criação de arquivos .pyc
ENV PYTHONDONTWRITEBYTECODE=1

# Logs aparecem imediatamente no terminal
ENV PYTHONUNBUFFERED=1

# Diretório de trabalho
WORKDIR /app

# Dependências
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Código da aplicação
COPY src ./src

# Artifact do modelo
COPY artifacts ./artifacts

# Porta da API
EXPOSE 8000

# Inicialização da aplicação
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]