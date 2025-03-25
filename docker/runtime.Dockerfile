FROM quay.io/modh/runtime-images@sha256:302d13703d7a014c62ca7d2be1add9d1dc42fd9094a949fa831dea4d58526789
USER 0
COPY ../requirements.txt requirements.txt
RUN pip install -r requirements.txt
USER 1001