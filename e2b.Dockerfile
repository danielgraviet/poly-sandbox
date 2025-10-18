# ✅ Official base image for Python execution
FROM e2bdev/code-interpreter:latest

# Install Python utilities you may want available in your sandbox
RUN pip install pytest loguru datasets

# Optional: create a startup script
RUN echo '#!/bin/bash\n\
echo "E2B sandbox ready."\n\
' > /root/.jupyter/start-up.sh && chmod +x /root/.jupyter/start-up.sh
