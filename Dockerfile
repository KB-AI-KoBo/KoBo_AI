FROM python:3.11.9
WORKDIR /
ENV FLASK_APP=./app/app.py
ENV FLASK_RUN_HOST=127.0.0.1
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get install -y nodejs

# Node.js 의존성 파일 복사 및 설치
COPY package.json package-lock.json* ./
RUN npm installRUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
apt-get install -y nodejs

# Node.js 의존성 파일 복사 및 설치
COPY package.json package-lock.json* ./
RUN npm install
EXPOSE 5000
COPY . .
CMD ["flask", "run", "--debug"]