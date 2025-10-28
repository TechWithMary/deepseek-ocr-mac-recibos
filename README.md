# DeepSeek-OCR en Mac M2: OCR para Recibos

Script simple ocr.py para leer recibos con DeepSeek-OCR local en Apple Silicon (M2/M3). Basado en mi tutorial YouTube.

## Instalación Rápida
1. Instala Miniconda: https://docs.conda.io/en/latest/miniconda.html (elige Apple M1 64-bit).
2. Clona DeepSeek-OCR: `git clone https://huggingface.co/deepseek-ai/DeepSeek-OCR`
3. En la carpeta DeepSeek-OCR: `python3 -m venv .venv; source .venv/bin/activate`
4. Instala deps: `pip install torch==2.6.0 transformers==4.46.3 tokenizers==0.20.3 einops addict easydict`
5. Copia ocr.py a la carpeta, cambia image_file a tu recibo, y corre `python ocr.py`.

## Uso
- Prompt: "<image>\nFree OCR." para simple.
- image_size=640 para detalle en números.
- test_compress=False para precisión.
- Output en output_dir/result.mmd (texto extraído).

Ver video para fixes MPS y tests con arrugada. ¡Prueba y comenta tu total!

Licencia: MIT – usa libre.
