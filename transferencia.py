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

content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

imsize = 512 if torch.cuda.is_available() else 128

loader = transforms.Compose([
    transforms.Resize(imsize),
    transforms.ToTensor()])

unloader = transforms.ToPILImage()
plt.ion()

def image_loader(image_name, size=None):
    img = Image.open(image_name)
    print(img.size)
    print(size)
    if size != None:
        img = img.resize(size)
    print(img.size)
    return img

def image_save(image,output_path):
    image = image.squeeze(0) 
    image = unloader(image)
    image.save(output_path)


def prepare(img):
    image = loader(img).unsqueeze(0)
    return image.to(device, torch.float)

def show_image(tensor, title=None):
    image = tensor.cpu().clone() 
    image = image.squeeze(0) 
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

def gram_matrix(input):
    a, b, c, d = input.size() 

    features = input.view(a * b, c * d) 

    G = torch.mm(features, features.t())

    return G.div(a * b * c * d)

def get_style_model_and_losses(cnn, normalization_mean, 
                                normalization_std,
                                style_img, content_img,
                                content_layers=content_layers_default,
                                style_layers=style_layers_default):

    normalization = Normalization(normalization_mean, normalization_std).to(device)

    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)

    i = 0 
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses

def get_input_optimizer(input_img, opt_name):
    if opt_name == "Adam":
        optimizer = optim.Adam([input_img],lr=0.005)
    else:
        optimizer = optim.LBFGS([input_img])
    return optimizer

def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=1000,
                       style_weight=1000000, content_weight=1, 
                       opt_name="LGFBS"):

    print('Construindo modelo de tranferência de estilo..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img)

    input_img.requires_grad_(True)
    model.requires_grad_(False)

    optimizer = get_input_optimizer(input_img, opt_name)

    print('Otimizando..')
    run = [0]
    layers = []
    style_loss_list = []
    content_loss_list = []
    while run[0] <= num_steps:

        def closure():
            with torch.no_grad():
                input_img.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            style_loss_list.append(style_score.item())
            content_loss_list.append(content_score.item())

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 100 == 0:
                print("iteração {}:".format(run))
                print('Perda do Estilo: {:4f} Perda do conteúdo: {:4f} Perda Total: {:4f}'.format(
                    style_score.item(), content_score.item(), style_score.item()+content_score.item()))
                print()
            
            if run[0] % (num_steps/5) == 0:
                layers.append(unloader(input_img.squeeze(0)))

            return style_score + content_score

        optimizer.step(closure)

    with torch.no_grad():
        input_img.clamp_(0, 1)

    return input_img, layers, style_loss_list, content_loss_list

class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std

def estilizar_imagem(argumentos):
    

    

    content_img = image_loader(os.path.join(argumentos["caminho_conteudo"], argumentos["conteudo"]))
    style_img = image_loader(os.path.join(argumentos["caminho_estilo"], argumentos["estilo"]), content_img.size)

    content_img = prepare(content_img)
    style_img = prepare(style_img)


    plt.figure()
    show_image(style_img, title='Imagem de Estilo')

    plt.figure()
    show_image(content_img, title='Imagem de Conteúdo')

    if argumentos["rede"] == "alexnet":
        cnn = models.alexnet(pretrained=True).features.to(device).eval()
    elif argumentos["rede"] == "vgg19":
        cnn = models.vgg19(pretrained=True).features.to(device).eval()
    else:
        cnn = models.vgg16(pretrained=True).features.to(device).eval()

    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    input_img = content_img.clone()
    # if you want to use white noise instead uncomment the below line:
    # input_img = torch.randn(content_img.data.size(), device=device)

    plt.figure()
    show_image(input_img, title='Imagem de Entrada')

    inicio = time.time()
    output, layers, s_loss, c_loss = run_style_transfer(cnn, cnn_normalization_mean, 
                                cnn_normalization_std,content_img, style_img, input_img, 
                                num_steps=argumentos['epocas'],style_weight=argumentos['peso_estilo'], content_weight=argumentos['peso_conteudo'],
                                opt_name=argumentos['otimizador'])
    fim = time.time()
    print(fim - inicio)

    fig, axs = plt.subplots(1,len(layers),figsize=(18,6))

    for i in range(len(layers)):
        axs[i].imshow(layers[i])
        axs[i].set_title('Iteração '+str(int((argumentos['epocas']/len(layers)) * (i+1))))

    plt.tight_layout()
    plt.show()

    plt.figure()
    show_image(output, title='Imagem de Saída')

    fig, ax = plt.subplots()

    image_save(output, os.path.join(argumentos["caminho_saida"],argumentos["conteudo"]+"_"+argumentos["estilo"]+".jpg"))

    dict_loss = {'style_loss':s_loss,'content_loss':c_loss}

    df_loss = pd.DataFrame(dict_loss)

    df_loss.to_csv('data.csv')

    ax.plot(s_loss, color = 'green', label = 'Perda do Estilo')
    ax.plot(c_loss, color = 'red', label = 'Perda do conteúdo')
    ax.legend(loc = 'upper left')
    plt.show()


    plt.ioff()
    plt.show()


if __name__ == "__main__":
    
    caminho_conteudo = os.path.join(os.path.dirname(__file__), 'dados', 'conteudo')
    caminho_saida = os.path.join(os.path.dirname(__file__), 'dados', 'saida')
    caminho_estilo = os.path.join(os.path.dirname(__file__), 'dados', 'estilo')

    os.makedirs(caminho_saida, exist_ok=True)

    
    parser = argparse.ArgumentParser()

    parser.add_argument("--conteudo", type=str, help="Imagem de conteudo para estilizacao", default='arco.jpg')
    parser.add_argument("--estilo", type=str, help="Imagem de estilo", default='starry.jpg')
    parser.add_argument("--entrada", type=str, help="Imagem de estilo", default='conteudo')
    parser.add_argument("--rede", type=str, help="Imagem de estilo", default='vgg16')
    parser.add_argument("--peso_estilo", type=float, help="Peso aplicado a função de perda do estilo", default=5e5)
    parser.add_argument("--peso_conteudo", type=float, help="Peso aplicado a função de perda do estilo", default=1)
    parser.add_argument("--epocas", type=int, help="Numero de iterações do processo de transferência", default=500)
    parser.add_argument("--otimizador", type=str, help="Numero de iterações do processo de transferência", default="LBFGS")

    args = parser.parse_args()


    # Wrapping inference configuration into a dictionary
    argumentos = dict()
    for arg in vars(args):
        argumentos[arg] = getattr(args, arg)

    argumentos['caminho_conteudo'] = caminho_conteudo
    argumentos['caminho_saida'] = caminho_saida
    argumentos['caminho_estilo'] = caminho_estilo

    estilizar_imagem(argumentos)