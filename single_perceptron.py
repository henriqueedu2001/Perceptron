import numpy as np

class Perceptron:
    def __init__(self, dimension) -> None:
        self.weights = np.zeros(dimension)
        self.bias = 0
        self.dimension = dimension 
        
    def output(self, x: np.array):
        """Saída do perceptron

        Args:
            x (np.array): vetor de entrada

        Returns:
            float: valor da saída y = f(x), sendo f a função de ativação
        """
        w_sum = self.weighted_sum(x)
        
        return self.logit(w_sum)
    
    
    def gradient_descent_fit(self, train_set: np.array, learning_rate: float, epochs: int) -> None:
        """Treinamento por gradiente descendente

        Args:
            train_set (np.array): dataset de treino
            learning_rate (float): taxa de aprendizado
            epochs (int): quantidade de épocas
        """
        # pontos do dataset de treino
        n_points = len(train_set)
        
        for i in range(epochs):
            loss_grad = self.loss_grad(train_set)
            
            # w^(t+1) = w^(t) - eta * grad(L(w^(t)))
            self.weights = self.weights - learning_rate*loss_grad[1:]
            self.bias = self.bias - learning_rate*loss_grad[0]
            
            print(self.bias, self.weights)
    
    
    def loss_grad(self, train_set: np.array) -> np.array:
        """Gradiente da Loss

        Args:
            train_set (np.array): dataset de treino

        Returns:
            np.array: gradiente da loss grad L = (grad_w_0, grad_w_1, 
            grad_w_2, grad_w_3, ..., grad_w_n)
        """
        # grad L = (grad_w_0, grad_w_1, grad_w_2, ..., grad_w_n)
        grad = np.zeros(self.dimension)
        
        n_points = len(train_set)
        
        # para cada ponto, somar parcela do gradiente
        for point in train_set:
                x, y = point[:-1], point[-1]
                y_hat = self.output(x)
                
                logit_grad = self.logit_gradient(x)
                
                grad = grad + (2/n_points)*(y_hat - y)*logit_grad
        
        return grad
    
    def weighted_sum(self, x: np.array):
        """Soma da entrada, ponderada pelos pesos do perceptron
        S(x) = w_0 + w^Tx = w_0 + w_1*x_1 + w_2*x_2 + ... + w_n*x_n
        Args:
            x (np.array): vetor de entrada

        Returns:
            float: valor da soma ponderada pelos pesos
        """
        # w_sum = w_0 + w^Tx = w_0 + w_1*x_1 + w_2*x_2 + ... + w_n*x_n
        return self.bias + np.dot(self.weights, x)
    
    
    def logit(self, x: float) -> float:
        """Função logística
        \sigma (x) = 1/(1 + e^{-x})

        Args:
            x (float): entrada

        Returns:
            float: saída da função logística y = f(x)
        """
        # logit(x) = 1/(1 + e^{-x})
        return 1/(1 + np.exp(-x))
    
    
    def logit_gradient(self, x:np.array) -> np.array:
        """Gradiente da função logística
        grad logit(x) = (grad_bias, grad_w_1, grad_w_2, ..., grad_w_n)

        Args:
            x (np.array): vetor no qual se deseja calcular o gradiente

        Returns:
            np.array: gradiente do vetor de pesos grad logit(x)
        """
        w_sum = self.weighted_sum(x)
        logit_x = self.logit(w_sum)
        logit_derivative = logit_x * (1 - logit_x)
        
        # grad_w_k = \sigma(x) * (1 - \sigma(x)) * w_k
        grad_weights = logit_derivative*self.weights
        
        # grad_w_k = \sigma(x) * (1 - \sigma(x))
        grad_bias = np.array([logit_derivative])
        
        return np.concatenate([grad_bias, grad_weights])


def generate_data(n):
    for i in range(n):
        x = 10*(np.random.rand() - 0.5)
        print(f'[{x}, {Perceptron.logit(Perceptron, x)}], ')
    
def test():
    p = Perceptron(1)
    p.weights = np.array([2.5])
    p.bias = -1.3
    
    # f(x) = logit(0 + 1x)
    train_set = np.array([
        [3.0345, 0.954], 
        [2.8760, 0.94], 
        [-2.666, 0.064], 
        [1.840, 0.862], 
        [-3.018, 0.0465], 
        [-3.844, 0.0209], 
        [-2.26, 0.0940], 
        [2.986, 0.951], 
        [2.482, 0.922], 
        [3.0026, 0.952]
    ])
    
    p.gradient_descent_fit(train_set, 0.10, 400)
    
    x = np.array([5])
    print(p.output(x))

# generate_data(10)
test()