FROM python:3.10-slim

# ---------- System setup ----------
WORKDIR /app
RUN apt-get update && apt-get install -y git wget sed && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# ---------- Clone ComfyUI ----------
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /app/ComfyUI

# ---------- Python & Torch (CPU only) ----------
RUN python -m venv /app/ComfyUI/venv && \
    /app/ComfyUI/venv/bin/pip install --upgrade pip && \
    /app/ComfyUI/venv/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    /app/ComfyUI/venv/bin/pip install -r /app/ComfyUI/requirements.txt

# ---------- Patch ComfyUI for full CPU mode ----------
RUN sed -i 's/return torch.device(torch.cuda.current_device())/return torch.device("cpu")/' /app/ComfyUI/comfy/model_management.py && \
    sed -i 's/torch\.cuda\.is_available()/False/' /app/ComfyUI/comfy/model_management.py && \
    sed -i 's/torch\.cuda\.get_device_properties(device)/None/' /app/ComfyUI/comfy/model_management.py && \
    sed -i 's/if dt == torch.float16 and should_use_fp16(device=device, model_params=model_params):/if False:/' /app/ComfyUI/comfy/model_management.py && \
    sed -i 's/if dt == torch.float16 and should_use_fp16(device=device, model_params=model_params, manual_cast=True):/if False:/' /app/ComfyUI/comfy/model_management.py && \
    sed -i 's/if dt == torch.bfloat16 and should_use_bf16(device=device, model_params=model_params):/if False:/' /app/ComfyUI/comfy/model_management.py && \
    sed -i 's/if props.major >= 8:/if False:/' /app/ComfyUI/comfy/model_management.py && \
    sed -i 's/if props.major < 6:/if False:/' /app/ComfyUI/comfy/model_management.py && \
    sed -i 's/if x in props.name.lower():/if False:/' /app/ComfyUI/comfy/model_management.py

# ---------- Add model ----------
# RUN mkdir -p /app/ComfyUI/models/checkpoints && \
#     wget -O /app/ComfyUI/models/checkpoints/v1-5-pruned-emaonly.ckpt \
#     https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt

    # ---------- Add a lightweight checkpoint ----------
RUN mkdir -p /app/ComfyUI/models/checkpoints && \
    wget -O /app/ComfyUI/models/checkpoints/sd-turbo.safetensors \
    https://huggingface.co/stabilityai/sd-turbo/resolve/main/sd_turbo.safetensors

# ---------- Permissions ----------
RUN mkdir -p /app/ComfyUI/output && chmod -R 777 /app/ComfyUI/output

# ---------- Environment ----------
ENV CUDA_VISIBLE_DEVICES=""
ENV COMFYUI_DEVICE=cpu
WORKDIR /app/ComfyUI
EXPOSE 8288

# ---------- Run ----------
CMD ["/app/ComfyUI/venv/bin/python", "main.py", "--listen", "0.0.0.0", "--port", "8288"]
