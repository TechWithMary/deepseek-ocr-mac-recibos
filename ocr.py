import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from transformers import AutoModel, AutoTokenizer
import torch
import os
import time

# Ruta local del modelo (ajusta si tu home es diferente)
model_path = '/Users/techwithmary/huggingface/DeepSeek-OCR'

print("Cargando tokenizer y modelo...")
load_start = time.time()

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)

# Determina dispositivo
device = 'cpu'
print("Usando CPU device (estable para demo)")

# Carga modelo sin torch_dtype (original dtype)
model = AutoModel.from_pretrained(
    model_path, 
    _attn_implementation='eager', 
    trust_remote_code=True, 
    local_files_only=True,
    use_safetensors=True
)

# Convierte a bfloat16 y mueve a device
model = model.eval().to(device).to(torch.float32)

load_end = time.time()
load_time = load_end - load_start
print(f"✅ Modelo cargado en {load_time:.2f} segundos")

# Customize
prompt = "<image>\nFree OCR."
image_file = '/Users/techwithmary/Downloads/Recibo borroso.jpg'  # Tu imagen
output_path = 'output_dir'

print(f"\n{'='*50}")
print(f"Iniciando OCR...")
print(f"{'='*50}")
print(f"Imagen: {image_file}")
print(f"Device: {device}")
print("-" * 50)

# Run inference
inference_start = time.time()

res = model.infer(
    tokenizer,
    prompt=prompt,
    image_file=image_file,
    output_path=output_path,
    base_size=1024,
    image_size=640,  # Para velocidad
    crop_mode=True,  # Gundam mode de la docs
    save_results=True,
    test_compress=False  # Comprime para velocidad
)

inference_end = time.time()
inference_time = inference_end - inference_start

print("\n" + "="*50)
print("OCR COMPLETADO!")
print("="*50)
print(f"⏱️  Tiempo carga: {load_time:.2f} seg")
print(f"⚡ Tiempo inferencia: {inference_time:.2f} seg")
print(f"📊 Total: {load_time + inference_time:.2f} seg")
print(f"\n📁 Output en: {output_path}/")
print(f"   - result.mmd (Markdown)")
print(f"   - result_with_boxes.jpg (Imagen anotada)")
print("="*50)
print(res)  # Resultado en consola
