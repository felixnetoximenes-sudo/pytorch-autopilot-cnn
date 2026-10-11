# 🏎️ Computer Vision Autopilot Pipeline (Steering Angle Regression)

Este repositório contém a implementação de um pipeline de **Visão Computacional** e **Deep Learning** ponta a ponta (*end-to-end*) desenvolvido em **PyTorch** e **OpenCV** para prever o ângulo de esterço do volante com base em imagens de estradas. 

O foco principal do projeto foi construir uma arquitetura robusta de engenharia de dados não estruturados, garantindo performance, consistência dimensional e tratamento de dados eficiente para ambientes produtivos.

---

## 🎯 Objetivos Técnicos Alcançados

* **Pipeline de Input de Alto Nível:** Manipulação e tratamento de imagens utilizando OpenCV, realizando a transposição de eixos NumPy `(H, W, C) -> (C, H, W)` e a normalização da intensidade dos pixels para o intervalo `[0.0, 1.0]`.
* **Consistência de Tensores:** Modelagem estruturada em 4 dimensões no PyTorch `[Batch_Size, Channels, Height, Width]`, garantindo compatibilidade industrial de dados para Redes Neurais Convolucionais (CNNs).
* **Processamento Batching Eficiente:** Implementação de arquitetura baseada em `DataLoader` e `TensorDataset` para processamento paralelo em lote, otimizando o uso de memória RAM/VRAM e evitando estouros de hardware.
* **Algoritmo de Aprendizado Completo:** Estruturação manual do loop de treinamento com retropropagação (*backpropagation*) utilizando a função de perda **MSELoss** e o otimizador **Adam**.

---

## 🧠 Arquitetura da Rede Neural (CNN)

O modelo foi construído utilizando a API `nn.Sequential` do PyTorch, garantindo uma esteira linear, limpa e performática de execução:

1. **nn.Conv2d:** Extração de atributos espaciais através de 16 filtros convolucionais (Kernel 3x3) operando sobre os 3 canais de cor (RGB).
2. **nn.ReLU:** Função de ativação não-linear para filtragem de gradientes, eliminando ruídos (valores negativos) e mantendo apenas as *features* ativadas.
3. **nn.MaxPool2d:** Redução dimensional espacial (pooling 2x2) que preserva os recursos mais relevantes e acelera drasticamente o tempo de inferência.
4. **nn.Flatten:** Vetorização dos mapas de atributos tridimensionais compactados em um vetor estável de 1 dimensão.
5. **nn.Linear:** Camada totalmente conectada (*Fully Connected*) que atua como regressora linear para a tomada de decisão, calculando o ângulo escalar contínuo do volante.

---

## 🛠️ Tecnologias e Ferramentas

* **Python**
* **PyTorch** (Deep Learning Framework)
* **OpenCV** (Processamento Digital de Imagens)
* **NumPy** (Álgebra Linear e Manipulação de Matrizes)

---

## 🚀 Como Executar o Projeto

1. Instale as dependências obrigatórias:
```bash
pip install torch numpy opencv-python
```

2. Para testar o modelo executando a inferência isolada sobre uma imagem real:
```bash
python piloto_automatico.py
```

3. Para executar o pipeline completo de treinamento em lote com o gerenciador de memória (`DataLoader`):
```bash
python treino_simulado.py
```
