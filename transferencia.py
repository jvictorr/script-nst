import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.models as models
import pandas as pd

# --- Configurações Globais ---

# Define quais camadas da VGG serão usadas para calcular a perda (Loss)
# Camadas mais profundas (conv_4) capturam estruturas de alto nível (conteúdo)
content_layers_default = ['conv_4']
# Camadas em vários níveis capturam texturas, cores e padrões (estilo)
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

# Seleciona GPU se disponível, caso contrário usa CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define o tamanho da imagem. Imagens maiores exigem mais memória VRAM.
# 512px é um bom balanço para GPU, 128px para CPU (para ser rápido).
imsize = 512 if torch.cuda.is_available() else 128

# --- Transformações de Imagem ---

# loader: Pipeline para preparar a imagem para a rede neural
# 1. Redimensiona
# 2. Converte para Tensor PyTorch (valores entre 0 e 1)
loader = transforms.Compose([
    transforms.Resize(imsize),
    transforms.ToTensor()])

# unloader: Converte de volta de Tensor para imagem PIL (para salvar/exibir)
unloader = transforms.ToPILImage()

# Ativa o modo interativo do matplotlib para atualizar gráficos em tempo real
plt.ion() 

# --- Funções Auxiliares de Imagem ---

def image_loader(image_name, size=None):
    """Carrega uma imagem, redimensiona e retorna um objeto PIL."""
    img = Image.open(image_name)
    print(f"Tamanho original: {img.size}")
    if size is not None:
        # Se um tamanho for forçado (ex: para igualar estilo ao conteúdo)
        img = img.resize(size)
    print(f"Novo tamanho: {img.size}")
    return img

def image_save(image, output_path):
    """Salva o tensor final como um arquivo de imagem no disco."""
    image = image.squeeze(0)  # Remove a dimensão do batch (1, C, H, W) -> (C, H, W)
    image = unloader(image)
    image.save(output_path)

def prepare(img):
    """Aplica as transformações e move a imagem para a GPU/CPU."""
    image = loader(img).unsqueeze(0) # Adiciona dimensão de batch falsificada: (C, H, W) -> (1, C, H, W)
    return image.to(device, torch.float)

def show_image(tensor, title=None):
    """Exibe um tensor como imagem usando Matplotlib."""
    image = tensor.cpu().clone()  # Clona para não alterar o tensor original
    image = image.squeeze(0)      # Remove dimensão de batch
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) # Pausa breve para atualizar a plotagem

# --- Lógica Matemática da Transferência de Estilo ---

def gram_matrix(input):
    """
    Calcula a Matriz de Gram.
    A Matriz de Gram representa as correlações entre os filtros de uma camada.
    Matematicamente, é o produto escalar das features vetorizadas.
    Isso captura a 'textura' sem se importar com a 'posição' espacial.
    """
    a, b, c, d = input.size()  # a=batch size(=1), b=mapas de características, (c,d)=dimensões

    features = input.view(a * b, c * d)  # Redimensiona para vetor: (b, c*d)

    G = torch.mm(features, features.t()) # Produto matricial

    # Normalizamos dividindo pelo número total de elementos para evitar
    # que camadas maiores tenham pesos excessivos.
    return G.div(a * b * c * d)

# --- Classes de Perda (Loss Modules) ---
# Estas classes funcionam como camadas "transparentes" na rede.
# Elas calculam o erro, mas deixam a imagem passar inalterada no forward pass.

class ContentLoss(nn.Module):
    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # 'target' é a features map da imagem de conteúdo original.
        # Usamos detach() para que não calculemos gradientes para o alvo, apenas para a entrada.
        self.target = target.detach()

    def forward(self, input):
        # Loss é o erro quadrático médio entre a entrada atual e o alvo
        self.loss = F.mse_loss(input, self.target)
        return input

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        # O alvo aqui é a Matriz de Gram da imagem de estilo original
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

# --- Normalização ---

class Normalization(nn.Module):
    """Normaliza a entrada com a média e desvio padrão do ImageNet (onde a VGG foi treinada)."""
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view(-1, 1, 1) permite broadcasting para funcionar com tensores de imagem (C, H, W)
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std

# --- Construção do Modelo ---

def get_style_model_and_losses(cnn, normalization_mean, 
                                normalization_std,
                                style_img, content_img,
                                content_layers=content_layers_default,
                                style_layers=style_layers_default):
    """
    Reconstrói a CNN (ex: VGG19), inserindo camadas de Normalização, 
    ContentLoss e StyleLoss nos lugares corretos.
    """
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    content_losses = []
    style_losses = []

    # Cria um novo modelo sequencial vazio começando com a normalização
    model = nn.Sequential(normalization)

    i = 0 
    # Itera sobre cada camada da CNN original pré-treinada
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # ReLU inplace=False é importante para não quebrar o cálculo da Loss
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Camada não reconhecida: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        # Se esta camada estiver na lista de camadas de CONTEÚDO
        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        # Se esta camada estiver na lista de camadas de ESTILO
        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # Remove camadas após a última camada de perda (para economizar processamento)
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses

def get_input_optimizer(input_img, opt_name):
    """Seleciona o otimizador. LBFGS geralmente converge mais rápido para style transfer."""
    if opt_name == "Adam":
        optimizer = optim.Adam([input_img], lr=0.005)
    else:
        optimizer = optim.LBFGS([input_img])
    return optimizer

# --- Loop de Treinamento (Otimização) ---

def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=1000,
                       style_weight=1000000, content_weight=1, 
                       opt_name="LGFBS"):
    """
    Executa o processo de transferência de estilo.
    IMPORTANTE: Aqui treinamos os PIXELS da 'input_img', não os pesos da rede.
    """

    print('Construindo modelo de tranferência de estilo..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img)

    # Queremos otimizar a imagem de entrada, então ativamos gradientes nela
    input_img.requires_grad_(True)
    # Não queremos alterar a rede neural em si
    model.requires_grad_(False)

    optimizer = get_input_optimizer(input_img, opt_name)

    print('Otimizando..')
    run = [0]
    layers = [] # Para armazenar snapshots da evolução
    style_loss_list = []
    content_loss_list = []

    while run[0] <= num_steps:

        # Função closure() é obrigatória para o otimizador LBFGS
        # Ela limpa gradientes, roda o forward pass e calcula a loss
        def closure():
            with torch.no_grad():
                # Garante que os pixels fiquem entre 0 e 1
                input_img.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img) # Forward pass
            
            style_score = 0
            content_score = 0

            # Soma as perdas de todas as camadas inseridas
            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            # Aplica os pesos definidos pelo usuário
            style_score *= style_weight
            content_score *= content_weight

            style_loss_list.append(style_score.item())
            content_loss_list.append(content_score.item())

            loss = style_score + content_score
            loss.backward() # Backpropagation (calcula como mudar os pixels para reduzir o erro)

            run[0] += 1
            if run[0] % 100 == 0:
                print("iteração {}:".format(run))
                print('Perda do Estilo: {:4f} Perda do conteúdo: {:4f} Perda Total: {:4f}'.format(
                    style_score.item(), content_score.item(), style_score.item()+content_score.item()))
                print()
            
            # Salva snapshots da imagem a cada 20% do processo
            if run[0] % (num_steps/5) == 0:
                layers.append(unloader(input_img.squeeze(0)))

            return style_score + content_score

        optimizer.step(closure)

    # Correção final dos valores de pixel
    with torch.no_grad():
        input_img.clamp_(0, 1)

    return input_img, layers, style_loss_list, content_loss_list

# --- Função Principal de Execução ---

def estilizar_imagem(argumentos):
    
    # Carrega as imagens dos caminhos especificados
    content_img = image_loader(os.path.join(argumentos["caminho_conteudo"], argumentos["conteudo"]))
    # Redimensiona o estilo para ter o mesmo tamanho do conteúdo (facilita cálculos)
    style_img = image_loader(os.path.join(argumentos["caminho_estilo"], argumentos["estilo"]), content_img.size)

    content_img = prepare(content_img)
    style_img = prepare(style_img)

    # Exibe imagens iniciais
    plt.figure()
    show_image(style_img, title='Imagem de Estilo')

    plt.figure()
    show_image(content_img, title='Imagem de Conteúdo')

    # Carrega a rede neural escolhida (apenas a parte 'features', sem o classificador)
    if argumentos["rede"] == "alexnet":
        cnn = models.alexnet(pretrained=True).features.to(device).eval()
    elif argumentos["rede"] == "vgg19":
        cnn = models.vgg19(pretrained=True).features.to(device).eval()
    else:
        cnn = models.vgg16(pretrained=True).features.to(device).eval()

    # Média e Std padrão do ImageNet para normalização
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    # Define a imagem inicial: pode ser uma cópia do conteúdo ou ruído aleatório
    if argumentos["entrada"] == "ruido":
        input_img = torch.randn(content_img.data.size(), device=device)
    else:
        input_img = content_img.clone()

    plt.figure()
    show_image(input_img, title='Imagem de Entrada')

    # Roda o processo
    inicio = time.time()
    output, layers, s_loss, c_loss = run_style_transfer(cnn, cnn_normalization_mean, 
                                cnn_normalization_std,content_img, style_img, input_img, 
                                num_steps=argumentos['epocas'],style_weight=argumentos['peso_estilo'], content_weight=argumentos['peso_conteudo'],
                                opt_name=argumentos['otimizador'])
    fim = time.time()
    print(f"Tempo total de execução: {fim - inicio:.2f} segundos")

    # Plota a evolução da imagem
    fig, axs = plt.subplots(1,len(layers),figsize=(18,6))
    for i in range(len(layers)):
        axs[i].imshow(layers[i])
        axs[i].set_title('Iteração '+str(int((argumentos['epocas']/len(layers)) * (i+1))))

    plt.tight_layout()
    plt.show()

    plt.figure()
    show_image(output, title='Imagem de Saída')

    # Salva imagem e dados
    fig, ax = plt.subplots()
    output_image_name = argumentos["conteudo"].split('.')[0]+"_"+argumentos["estilo"].split('.')[0]+".jpg"
    image_save(output, os.path.join(argumentos["caminho_saida"],output_image_name))

    # Salva histórico de perda em CSV
    dict_loss = {'style_loss':s_loss,'content_loss':c_loss}
    df_loss = pd.DataFrame(dict_loss)
    df_loss.to_csv('data.csv')

    # Plota gráfico de perda
    ax.plot(s_loss, color = 'green', label = 'Perda do Estilo')
    ax.plot(c_loss, color = 'red', label = 'Perda do conteúdo')
    ax.legend(loc = 'upper left')
    plt.show()

    plt.ioff() # Desliga modo interativo
    plt.show()

# --- Entry Point (Main) ---

if __name__ == "__main__":
    
    # Define caminhos padrão baseados na localização deste script
    caminho_conteudo = os.path.join(os.path.dirname(__file__), 'dados', 'conteudo')
    caminho_saida = os.path.join(os.path.dirname(__file__), 'dados', 'saida')
    caminho_estilo = os.path.join(os.path.dirname(__file__), 'dados', 'estilo')

    os.makedirs(caminho_saida, exist_ok=True)

    # Configura argumentos da linha de comando
    parser = argparse.ArgumentParser()

    parser.add_argument("--conteudo", type=str, help="Imagem de conteudo para estilizacao", default='arco.jpg')
    parser.add_argument("--estilo", type=str, help="Imagem de estilo", default='starry.jpg')
    parser.add_argument("--entrada", type=str, help="Entrada inicial é conteudo ou ruido", default='conteudo')
    parser.add_argument("--rede", type=str, help="Rede convolucional de preferencia (vgg16, vgg19, alexnet)", default='vgg16')
    # Peso alto no estilo = imagem mais artística. Peso baixo = imagem mais realista.
    parser.add_argument("--peso_estilo", type=float, help="Peso aplicado a função de perda do estilo", default=5e5)
    parser.add_argument("--peso_conteudo", type=float, help="Peso aplicado a função de perda do estilo", default=1)
    parser.add_argument("--epocas", type=int, help="Numero de iterações do processo de transferência", default=500)
    parser.add_argument("--otimizador", type=str, help="Algoritmo de otimização (Adam ou LBFGS)", default="LBFGS")

    args = parser.parse_args()

    # Empacota argumentos em um dicionário para passar para a função principal
    argumentos = dict()
    for arg in vars(args):
        argumentos[arg] = getattr(args, arg)

    argumentos['caminho_conteudo'] = caminho_conteudo
    argumentos['caminho_saida'] = caminho_saida
    argumentos['caminho_estilo'] = caminho_estilo

    estilizar_imagem(argumentos)