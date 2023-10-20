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
        
    
def test():
    p = Perceptron(2)
    p.weights = np.array([2, 3])
    p.bias = 7
    
    x = np.array([3,1])
    p.weighted_sum(x)
    print(p.logit_gradient(x))
    
test()