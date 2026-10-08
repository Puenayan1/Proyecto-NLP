import torch

print(f"Versión de PyTorch: {torch.__version__}")
print(f"¿CUDA está disponible?: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"Nombre de la GPU: {torch.cuda.get_device_name(0)}")
    print(f"Cantidad de GPUs detectadas: {torch.cuda.device_count()}")
else:
    print("PyTorch no detectó la GPU y está usando la CPU. Revisa los drivers o la versión de CUDA instalada en Windows.")