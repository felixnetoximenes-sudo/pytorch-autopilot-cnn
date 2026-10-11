import cv2 
import torch
import numpy as np
import torch.nn as nn
imagem = cv2.imread('images.jpg')
if imagem is None:
    print ('[AVISO!] Imagem não encontrado')

else:
    imagem_organizado = np.transpose(imagem,(2, 0, 1))
    convertendo = torch.from_numpy(imagem_organizado).float()
    imagem_normalizada = imagem_organizado / 255.0
    dados1 = convertendo.unsqueeze(0)

    rede_completa = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3), # Scanner
        nn.ReLU(),                                                # Corta Negativos
        nn.MaxPool2d(kernel_size=2, stride=2),                    # Compactador
        
        nn.Flatten(),                                             # Estica o bloco em uma linha reta
        
        # Tomada de Decisão: Recebe os 376.128 números esticados e cospe 1 número final
        nn.Linear(in_features=16 * 201 * 117, out_features=1)
    )

    # 3. Executando a decisão final
    comando_volante = rede_completa(dados1)
    
    print("\n================ DECISÃO DA IA ================")
    print(f"Formato da Saída Final: {comando_volante.shape}")
    print(f"Ângulo calculado para o Volante: {comando_volante.item():.4f} graus")
    print("===============================================")