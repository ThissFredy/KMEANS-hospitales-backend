# 1. Imagen base
FROM python:3.10-slim

# 2. Directorio de trabajo
WORKDIR /app

# 3. Instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copiar el código (api.py, model.py, etc.)
COPY . .

# 5. Comando de ejecución CORREGIDO
# A) Usamos "sh -c" para poder leer la variable de entorno $PORT de Render.
# B) Cambiamos "main:app" por "api:app" porque tu archivo es api.py.
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}"]