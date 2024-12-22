import numpy as np
import cv2
from math import pow, sqrt, atan2, degrees, radians, cos, sin
import matplotlib.pyplot as plt
import os

#Parâmetros de entrada
ksize = 3
nome = 'Imagem_1.png'
t1 = 31
t2 = 127
#𝜃 = 𝑎𝑡𝑎𝑛2(𝜕𝑓/𝜕𝑦, 𝜕𝑓/𝜕𝑥)

#Função para aplicar o filtro da mediana
def median_filter(img, ksize):
    """
    Aplica um filtro de mediana em uma imagem para redução de ruído.

    Entradas:
        img: Imagem de entrada em escala de cinza.
        ksize: Tamanho do kernel (deve ser ímpar).

    Retorno:
        filtered_img: Imagem filtrada após a aplicação da mediana.
    """
    pad_size = ksize // 2
    padded_img = np.pad(img, pad_size, mode='constant', constant_values = 0)
    filtered_img = np.zeros_like(img)

    #Percorre a imagem aplicando a mediana em cada janela
    for i in range(filtered_img.shape[0]):
        for j in range(filtered_img.shape[1]):
            #Extrai a janela ao redor do pixel atual
            window = padded_img[i:i + ksize, j:j + ksize]
            #Calcula a mediana da janela e define no pixel central
            filtered_img[i, j] = np.median(window)
    return filtered_img

#Função para aplicar o filtro gaussiano
def gaussian_filter(img, ksize, sigma = 1.0):
    """
    Aplica um filtro gaussiano em uma imagem para suavização.

    Entradas:
        img: Imagem de entrada em escala de cinza.
        ksize: Tamanho do kernel (deve ser ímpar).
        sigma: Desvio padrão da função gaussiana.

    Retorno:
        filtered_img: Imagem suavizada após a aplicação do filtro gaussiano.
    """
    pad_size = ksize // 2
    padded_img = np.pad(img, pad_size, mode = 'constant', constant_values = 0)
    filtered_img = np.zeros_like(img)
    kernel = gaussian_kernel(ksize, sigma)

#Percorre a imagem aplicando convolução
    for i in range(filtered_img.shape[0]):
        for j in range(filtered_img.shape[1]):
            #Extrai a janela ao redor do pixel atual
            window = padded_img[i:i + ksize, j:j + ksize]
            #Calcula a convolução da janela com o kernel gaussiano
            filtered_img[i, j] = np.sum(window * kernel)
    return filtered_img

#Função para gerar o kernel gaussiano
def gaussian_kernel(ksize, sigma):
    """
    Gera um kernel gaussiano para uso em operações de suavização.

    Entradas:
        ksize: Tamanho do kernel (deve ser ímpar).
        sigma: Desvio padrão da função gaussiana.

    Retorno:
        kernel: Kernel gaussiano normalizado.
    """
    ax = np.linspace(-(ksize // 2), ksize // 2, ksize)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / np.sum(kernel)

#FUnção para calcular o Thresholding duplo:
def double_thresholding(suppressed_image):
    """
    Aplica o limiar duplo em uma imagem, identificando bordas fortes e fracas.

    Entradas:
        suppressed_image: Imagem resultante da supressão não máxima.

    Retorno:
        thresholding_image: Imagem com bordas fortes e fracas marcadas.
    """
    thresholding_image = np.zeros_like(suppressed_image)

    #Marca apenas as bordas fortes e fracas por enquanto:
    for i in range(suppressed_image.shape[0]):
        for j in range(suppressed_image.shape[1]):
            #Bordas fortes
            if suppressed_image[i, j] >= t2:
                thresholding_image[i, j] = 255
            #Bordas fracas
            elif (suppressed_image[i, j] < t2) and (suppressed_image[i, j] >= t1):
                thresholding_image[i, j] = 128
            elif suppressed_image[i, j] < t1:
                thresholding_image[i, j] = 0

    return thresholding_image

#Função para aplicar convolução 2D
def convolve2d(img, kernel):
    """
    Realiza uma convolução 2D entre a imagem e o kernel fornecido.

    Entradas:
        img: Imagem de entrada.
        kernel: Kernel a ser utilizado na convolução.

    Retorno:
        result: Resultado da convolução 2D.
    """
    pad_size = kernel.shape[0] // 2

    result = np.zeros_like(img, dtype = np.float32)

    #Percorre a imagem aplicando a convolução
    for i in range(pad_size, img.shape[0] - pad_size):
        for j in range(pad_size, img.shape[1] - pad_size):
            
            #Extrai a região da janela
            window = img[i - pad_size:i + pad_size + 1, j - pad_size:j + pad_size + 1]
            
            #Multiplica a janela pelo kernel e soma
            result[i, j] = np.sum(window * kernel)
    return result

#Função para calcular o Sobel
def sobel_filter(img, ksize):
    #Define os kernels Sobel para X e Y
    sobel_x = np.array([[-1, -2, -1],
                        [0,   0,  0],
                        [1,   2,  1]])
    
    sobel_y = (sobel_x.T)

    #Aplica a convolução com os kernels Sobel
    grad_x = convolve2d(img, sobel_x)
    grad_y = convolve2d(img, sobel_y)
    Gradiente_final = np.zeros_like(img, dtype = np.float32)
    Direcao_Sobel = np.zeros_like(img, dtype = np.float32)

    Imagem_Colorida = np.zeros((img.shape[0],img.shape[1],3), dtype = np.float32)

    for i in range (Gradiente_final.shape[0]):
        for j in range (Gradiente_final.shape[1]):
            Gradiente_final[i, j] = sqrt(pow(grad_x[i, j], 2) + pow(grad_y[i, j], 2))
            Direcao_Sobel[i, j] = degrees(atan2(grad_y[i, j], grad_x[i, j]))
            if Direcao_Sobel[i, j] < 0:
                Direcao_Sobel[i, j] += 180
            
            if Gradiente_final[i, j] > 0:

                if Direcao_Sobel[i, j] <= 22.5 or 157.5 < Direcao_Sobel[i, j] <= 180:
                    Direcao_Sobel[i, j] = 0
                    Imagem_Colorida[i, j] = [0, 255, 255]

                elif 22.5 < Direcao_Sobel[i, j] <= 67.5:
                    Direcao_Sobel[i, j] = 45
                    Imagem_Colorida[i, j] = [0, 255, 0]

                elif 67.5 < Direcao_Sobel[i, j] <= 112.5:
                    Direcao_Sobel[i, j] = 90
                    Imagem_Colorida[i, j] = [255, 0, 0]

                elif 112.5 < Direcao_Sobel[i, j] <= 157.5:
                    Direcao_Sobel[i, j] = 135
                    Imagem_Colorida[i, j] = [ 0, 0, 255]

    #Normaliza os gradientes para visualização (0-255), preferido em relação às funções cv2.normalize() ou cv2.convertScaleAbs()
    grad_x = np.clip(np.abs(grad_x), 0, 255).astype(np.uint8)
    grad_y = np.clip(np.abs(grad_y), 0, 255).astype(np.uint8)
    Gradiente_final = np.clip(np.abs(Gradiente_final), 0, 255).astype(np.uint8)
    Direcao_Sobel = np.clip(np.abs(Direcao_Sobel), 0, 255).astype(np.uint8)

    return grad_x, grad_y, Gradiente_final, Direcao_Sobel, Imagem_Colorida

def non_maximum_suppression(gradient_magnitude, gradient_direction):

    #Cria uma cópia da magnitude para modificar
    suppressed_image = gradient_magnitude.copy()
    
    #Converte direções para o intervalo [0, 180]
    gradient_direction = gradient_direction % 180

    #Itera pixel a pixels ignorando as bordas:
    for i in range(1, gradient_direction.shape[0] - 1):
        for j in range(1, gradient_direction.shape[1] - 1):
            if gradient_magnitude[i, j] > 0:
                theta = gradient_direction[i, j]

                if 0 <= theta <= 22.5 or 157.5 <= theta <= 180:
                    if gradient_magnitude[i, j] <= max(gradient_magnitude[i - 1, j], gradient_magnitude[i + 1, j]):
                        suppressed_image[i, j] = 0

                elif 22.5 < theta <= 67.5:
                    if gradient_magnitude[i, j] <= max(gradient_magnitude[i - 1, j - 1], gradient_magnitude[i + 1, j + 1]):
                        suppressed_image[i, j] = 0

                elif 67.5 < theta <= 112.5:
                    if gradient_magnitude[i, j] <= max(gradient_magnitude[i, j - 1], gradient_magnitude[i, j + 1]):
                        suppressed_image[i, j] = 0

                elif 112.5 < theta < 157.5:
                    if gradient_magnitude[i, j] <= max(gradient_magnitude[i + 1, j - 1], gradient_magnitude[i - 1, j + 1]):
                        suppressed_image[i, j] = 0

    return suppressed_image

def hysteresis(thresholded_image):

    #Inicializa uma cópia da imagem para modificar
    hysteresis_imgage = thresholded_image.copy()

    #Variável para rastrear alterações na imagem durante a iteração
    altered = True

    #Laço iterativo que só para quando nenhuma mudança adicional ocorrer
    while altered:
        altered = False  #Reseta a variável a cada iteração
        imgage_copy = hysteresis_imgage.copy()  #Copia a imagem atual para evitar modificações diretas durante a iteração

        #Percorre todos os pixels da imagem (ignorando bordas)
        for i in range(1, hysteresis_imgage.shape[0] - 1):
            for j in range(1, hysteresis_imgage.shape[1] - 1):
                #Verifica se o pixel atual é uma borda fraca (valor 128)
                if hysteresis_imgage[i, j] == 128:
                    #Extrai a janela 3x3 centrada no pixel atual
                    I_r = hysteresis_imgage[i - 1:i + 2, j - 1:j + 2]

                    #Verifica se há bordas fortes (valor 255) na janela 3x3
                    if (I_r == 255).any():
                        imgage_copy[i, j] = 255  #Promove o pixel atual a borda forte
                        altered = True  # Marca que houve uma alteração

        #Atualiza a imagem para a próxima iteração
        hysteresis_imgage = imgage_copy.copy()

    #Após o laço, descarta todos os pixels fracos restantes (converte 128 para 0)
    hysteresis_imgage[hysteresis_imgage == 128] = 0

    #Retorna a imagem refinada
    return hysteresis_imgage

def hough_transform(edges, rho_res=1, theta_res=1, threshold=100):

    rows, cols = edges.shape
    diag_len = int(np.sqrt(rows**2 + cols**2))  # Comprimento da diagonal
    rho_max = diag_len
    rho_range = int(2 * rho_max / rho_res) + 1  # Resolução de rho
    theta_range = int(180 / theta_res)  # Resolução de theta

    #Inicializa o acumulador
    accumulator = np.zeros((rho_range, theta_range), dtype=np.int32)

    #Pré-calcula seno e cosseno para os ângulos possíveis
    thetas = np.arange(0, 180, theta_res)
    cos_theta = np.cos(np.deg2rad(thetas))
    sin_theta = np.sin(np.deg2rad(thetas))

    #Varre os pixels de borda
    y_idxs, x_idxs = np.nonzero(edges)  # Encontra os pixels com bordas
    for x, y in zip(x_idxs, y_idxs):
        for theta_idx, (ct, st) in enumerate(zip(cos_theta, sin_theta)):
            rho = int(round((x * ct + y * st) / rho_res)) + rho_max
            accumulator[rho, theta_idx] += 1

    #Lista para armazenar as linhas detectadas
    lines = []

    #Verifica o acumulador e extrai as linhas que excedem o limiar
    for rho_idx in range(rho_range):
        for theta_idx in range(theta_range):
            if accumulator[rho_idx, theta_idx] >= threshold:
                rho = (rho_idx - rho_max) * rho_res
                theta = theta_idx * theta_res
                lines.append((rho, theta))

    return lines

def draw_hough_lines(image_path, lines):
    """
    Desenha as linhas detectadas na imagem original.

    Entradas:
        image_path: Arquivo da imagem original.
        lines: Lista de linhas detectadas no formato [(rho, theta), ...].

    Retorno:
        img: Imagem com as linhas sobrepostas.
    """
    #Carrega a imagem original
    img = image_path.copy()
    if img is None:
        raise ValueError("Erro ao carregar a imagem. Verifique o caminho.")

    #Itera pelas linhas detectadas
    for rho, theta in lines:
        theta_rad = np.deg2rad(theta)
        a = np.cos(theta_rad)
        b = np.sin(theta_rad)
        x0 = a * rho
        y0 = b * rho
        #Define dois pontos para desenhar a linha
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        #Desenha a linha em vermelho
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return img

#Carrega a imagem em escala de cinza
imagem = cv2.imread(nome, 0)

#Remove a extensão do nome do arquivo (name without extension)
nwe = os.path.splitext(nome)[0]

#Cria as pasta caso ela não exista
output_Canny = "Resultados Canny"
if not os.path.exists(output_Canny):
    os.makedirs(output_Canny)

output_Hough = "Resultados Hough"
if not os.path.exists(output_Hough):
    os.makedirs(output_Hough)

if imagem is None:
    print(f'Arquivo de imagem não encontrado')
else:
    print(f'Arquivo {nome} carregado')

    #Os próximos passos definem a implementação do Canny:
    print(f'Linhas: {imagem.shape[0]}, Colunas: {imagem.shape[1]}\n')
    # 1 - Redução de ruído
    imagem = median_filter(imagem, ksize)
    # 2 - Aplica o filtro Sobel
    sobelx, sobely, final, Direcao_Sobel, Imagem_Colorida = sobel_filter(imagem, ksize)
    # 3 - Aplica o "Nom Maximum Supression"
    NMS = non_maximum_suppression(final, Direcao_Sobel)
    # 4 - Aplica o Duplo Threshold
    thresholding_image = double_thresholding(NMS)
    # 5 - Aplica a Função de Histerese
    hysteresis_image = hysteresis(thresholding_image)
    # Após todos os passos, hysteresis_image será o arquivo com o resultado do Canny!

    linhas_hough = hough_transform(hysteresis_image)
    imagem_hough = draw_hough_lines(imagem, linhas_hough)

    #Exibe as imagens
    cv2.imshow('Imagem Original', imagem)
    cv2.imshow('Sobel X', sobelx)
    cv2.imwrite(os.path.join(output_Canny, f'{nwe}_Sobel x.png'), sobelx)
    cv2.imshow('Sobel Y', sobely)
    cv2.imwrite(os.path.join(output_Canny, f'{nwe}_Sobel y.png'), sobely)
    cv2.imshow('Gradiente', final)
    cv2.imwrite(os.path.join(output_Canny, f'{nwe}_Gradiente.png'), final)

    cv2.imshow('Direcao_Sobel', Direcao_Sobel)
    cv2.imwrite(os.path.join(output_Canny, f'{nwe}_Direcao Sobel.png'), Direcao_Sobel)
    
    cv2.imshow('Imagem Colorida', Imagem_Colorida)
    cv2.imwrite(os.path.join(output_Canny, f'{nwe}_Imagem Colorida.png'), Imagem_Colorida)
    
    cv2.imshow('Non Maximum Suppression', NMS)
    cv2.imwrite(os.path.join(output_Canny, f'{nwe}_Non Maximum Suppression.png'), NMS)
    
    cv2.imshow('Duplo thresholding', thresholding_image)
    cv2.imwrite(os.path.join(output_Canny, f'{nwe}_Duplo thresholding.png'), thresholding_image)
    
    cv2.imshow('Canny Completo', hysteresis_image)
    cv2.imwrite(os.path.join(output_Canny, f'{nwe}_Canny Completo.png'), hysteresis_image)

    cv2.imshow('Transformada de Hough', imagem_hough)
    cv2.imwrite(os.path.join(output_Hough, 'Transformada de Hough.png'), imagem_hough)

    cv2.waitKey(0)
    cv2.destroyAllWindows()