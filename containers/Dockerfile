FROM python:3.10.4-slim-bullseye

RUN apt-get update && apt-get install -y \
    curl \
    bzip2 \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN mkdir ilastik; \
    curl https://files.ilastik.org/ilastik-1.3.3post3-Linux.tar.bz2 | \
    tar xjf - --strip-components 1 -C ilastik

# COPY requirements.txt ./
# RUN pip install --no-cache-dir -r requirements.txt

# # COPY . .

# # CMD [ "python", "./your-daemon-or-script.py" ]
# ENTRYPOINT [ "bash" ]
ENTRYPOINT [ "bash", "ilastik/run_ilastik.sh", "--headless", "--export_source=Probabilities" ]