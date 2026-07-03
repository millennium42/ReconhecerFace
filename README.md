# ReconhecerFace

Projeto simples em Python para deteccao de rostos em tempo real com webcam usando OpenCV.

## O que este projeto demonstra

Este repositorio foi usado para praticar:

- visao computacional basica
- captura de video em tempo real
- deteccao de rostos com classificadores prontos
- uso do OpenCV em Python

## Como funciona

O script:

- abre a webcam
- converte cada frame para escala de cinza
- aplica o classificador Haar Cascade do OpenCV
- desenha retangulos sobre os rostos detectados
- exibe o video em tempo real

## Stack

- Python
- OpenCV

## Como executar

### 1. Instalar dependencias

```bash
pip install opencv-python
```

### 2. Executar

```bash
python main.py
```

### 3. Encerrar

Pressione `q` para fechar a janela da aplicacao.

## Estrutura

- `main.py`: captura da webcam e deteccao de rostos

## O que eu implementei neste projeto

Neste repositorio, trabalhei em um exemplo funcional de visao computacional em Python para aprendizado de captura de video, processamento de imagem e deteccao de objetos.
