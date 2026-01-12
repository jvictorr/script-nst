# 🎨 Neural Style Transfer (Transferência de Estilo Neural)

Este projeto implementa um algoritmo de **Transferência de Estilo** utilizando **PyTorch**. Ele permite combinar o conteúdo de uma foto (ex: sua selfie) com o estilo artístico de uma pintura (ex: *A Noite Estrelada*), gerando uma nova imagem única.

O script foi projetado para rodar tanto em **CPU** quanto em **GPU (CUDA)** automaticamente, dependendo da disponibilidade do seu hardware.

---

## 📋 Dependências

Para executar este projeto, certifique-se de que seu ambiente Python possui as seguintes bibliotecas instaladas:

* **Python 3.6+**
* **Torch** e **Torchvision** (Processamento da Rede Neural)
* **Pillow (PIL)** (Manipulação de imagens)
* **Matplotlib** (Visualização de gráficos)
* **Pandas** (Exportação de dados de log)

---

## 📂 Configuração das Pastas (Importante!)

O script espera uma estrutura de pastas específica para encontrar as imagens. **Você deve criar essas pastas manualmente** na raiz do projeto antes de rodar:

```text
seu_projeto/
│
├── transferencia.py      # O arquivo principal do script
├── data.csv              # (Gerado automaticamente após rodar)
│
└── dados/                # Crie esta pasta principal
    ├── conteudo/         # Coloque suas fotos originais aqui (ex: foto.jpg)
    ├── estilo/           # Coloque as artes de estilo aqui (ex: pintura.jpg)
    └── saida/            # O resultado será salvo aqui
```

---

## 🚀 Como Rodar

### 1. Execução Rápida (Padrão)
Se você tiver uma imagem chamada'arco.jpg'na pasta'conteudo'e'starry.jpg'na pasta'estilo`, basta rodar:

```bash
python transferencia.py
```
*Configuração padrão: 500 épocas, rede VGG16.*

### 2. Escolhendo Suas Próprias Imagens
Para usar arquivos com nomes diferentes:

```bash
python transferencia.py --conteudo "minha_foto.jpg" --estilo "monet.jpg"
```
*(Nota: Os arquivos devem estar dentro das pastas'dados/conteudo'e'dados/estilo'respectivamente).*

### 3. Ajustando a Intensidade
Para mudar o equilíbrio entre a foto original e o estilo artístico:

* **Mais Estilo:** Aumente o'--peso_estilo'(ex:'10000000').
* **Mais Conteúdo Original:** Diminua o'--peso_estilo'(ex:'100000').

```bash
python transferencia.py --peso_estilo 10000000
```

### 4. Modo Alta Qualidade (Mais lento)
Para um resultado mais refinado, aumente as épocas e use a rede VGG19:

```bash
python transferencia.py --epocas 2000 --rede vgg19
```

---

## ⚙️ Argumentos Disponíveis

| Argumento | Descrição | Valor Padrão |
| :--- | :--- | :--- |
|'--conteudo'| Nome do arquivo da imagem de conteúdo |'arco.jpg'|
|'--estilo'| Nome do arquivo da imagem de estilo |'starry.jpg'|
|'--rede'| Modelo neural ('vgg16','vgg19','alexnet') |'vgg16'|
|'--epocas'| Número de iterações do treinamento |'500'|
|'--peso_estilo'| Intensidade do estilo artístico |'5e5'(500.000) |
|'--peso_conteudo`| Intensidade da preservação da foto |'1'|
|'--otimizador'| Algoritmo ('LBFGS'ou'Adam') |'LBFGS'|

---

## 📊 Saída e Resultados

Ao final da execução, o script gera:

1.  **Imagem Final:** Salva em'dados/saida/'com o nome combinado (ex:'foto_pintura.jpg').
2.  **Visualização:** Uma janela gráfica mostrando a evolução do processo.
3.  **Logs:** Um arquivo'data.csv'contendo o histórico das perdas (losses) de estilo e conteúdo para análise.