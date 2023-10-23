import numpy as np

class Perceptron:
    def __init__(self, dimension, activation_function_type: str = 'logit') -> None:
        self.weights = np.zeros(dimension + 1)
        self.dimension = dimension
        self.activation_function_type = activation_function_type
        self.activation_function_dict = {
            'logit': self.logit,
            'relu': self.ReLU,
            'tanh': self.hiperbolic_tan,
            'linear': self.linear_identity
        }
        
        
    def output(self, x: np.array):
        """Saída y do perceptron, para um dado x = (x_1, x_2, x_3, ..., x_n)

        Args:
            x (np.array): vetor de entrada x = (x_1, x_2, x_3, ..., x_n)

        Returns:
            float: valor da saída y = f(w_sum(x)), sendo f a função de ativação e w_sum(x) =
            <w,x> = w^Tx = w_0 + w_1*x_1 + w_*2x_2 + w_3*x_3 + ... + w_n*x_n
        """
        w_sum = self.weighted_sum(x)
        
        return self.activation_function(w_sum)
    
    
    def activation_function(self, x: np.array) -> float:
        """Função de ativação f

        Args:
            x (np.array): vetor de entrada x = (x_1, x_2, x_3, ..., x_n)

        Returns:
            float: resultado de f(x) 
        """
        
        return self.activation_function_dict[self.activation_function_type](x)
    
    
    def gradient_descent_fit(self, train_set: np.array, learning_rate: float, epochs: int) -> None:
        """Treinamento por gradiente descendente

        Args:
            train_set (np.array): dataset de treino
            learning_rate (float): taxa de aprendizado
            epochs (int): quantidade de épocas
        """
        
        for i in range(epochs):
            loss_grad = self.loss_grad(train_set)
            
            # w^(t+1) = w^(t) - eta * grad(L(w^(t)))
            self.weights = self.weights - learning_rate*loss_grad
            
            print(self.weights)
    
    
    def loss_grad(self, train_set: np.array) -> np.array:
        """Gradiente da Loss

        Args:
            train_set (np.array): dataset de treino

        Returns:
            np.array: gradiente da loss grad L = (grad_w_0, grad_w_1, 
            grad_w_2, grad_w_3, ..., grad_w_n)
        """
        
        # grad L = (grad_w_0, grad_w_1, grad_w_2, ..., grad_w_n)
        grad = np.zeros(self.dimension + 1)
        
        for i in range(0, self.dimension + 1):
            grad[i] = self.loss_partial_derivative(train_set, i)
        
        return grad
    
    
    def loss_partial_derivative(self, train_set: np.array, weigth_index: int) -> float:
        """Derivada parcial da Loss del L/del w_k, em relação ao peso w_k

        Args:
            train_set (np.array): dataset de treino
            weigth_index (int): índice k do peso w_k, em relação ao qual se deseja calcular
            del L/del w_k

        Returns:
            float: derivada parcial da loss em relação a del L/del w_k
        """
        n_points = len(train_set)
        
        loss_par_der = 0
        
        for point in train_set:
            x, y = point[:-1], point[-1]
            y_hat = self.output(x)    
            w_k = x[weigth_index - 1] if weigth_index != 0 else 1
            
            loss_par_der = loss_par_der + ((y_hat - y)*y_hat*(1 - y_hat)*w_k/n_points)
            
        return loss_par_der
    
    
    def weighted_sum(self, x: np.array):
        """Soma da entrada, ponderada pelos pesos do perceptron
        S(x) = <w,x> = w_0 + w^Tx = w_0 + w_1*x_1 + w_2*x_2 + ... + w_n*x_n,
        para x = (1, x_1, x_2, ..., x_n) e w = (w_0, w_1, w_2, ..., w_n)
        Args:
            x (np.array): vetor de entrada

        Returns:
            float: valor da soma ponderada pelos pesos
        """
        
        # w_sum = <w,x> = w^Tx = w_0 + w_1*x_1 + w_2*x_2 + ... + w_n*x_n
        return self.weights[0] + np.dot(self.weights[1:], x)
    
    
    def logit(self, x: float) -> float:
        """Função logística
        \sigma (x) = 1/(1 + e^{-x})

        Args:
            x (float): entrada

        Returns:
            float: saída da função logística f(x) = \sigma(x) = 1/(1 + e^{-x})
        """
        
        return 1/(1 + np.exp(-x))
    
    
    def hiperbolic_tan(self, x: float) -> float:
        """Função tangente hiperbólica
        f(x) = (e^x - e^{-x})/(e^x + e^{-x})

        Args:
            x (float): entrada x

        Returns:
            float: saída da função tangente hiperbólica f(x) = tanh(x) = 
            (e^x - e^{-x})/(e^x + e^{-x})
        """
        
        return np.tanh(x)
    
    
    def ReLU(self, x: float) -> float:
        """Função ReLU (Rectified Linear Unit) f(x) = max(0, x)

        Args:
            x (float): entrada x

        Returns:
            float: saída da função ReLU f(x) = max(0, x)
        """
        
        return x if x > 0 else 0
    
    
    def step(self, x: float) -> float:
        """Função degrau, de Heaviside H(x) = 1, se x >= 0, e H(x) = 
        0, se x < 0

        Args:
            x (float): entrada x

        Returns:
            float: saída da função degrau de Heaviside H(x); H(x) = 1, 
            se x >= 0, e H(x) = 0, se x < 0
        """
        
        return 1 if x >= 0 else 0
    
    
    def linear_identity(self, x: float) -> float:
        """Função identidade I(x) = x

        Args:
            x (float): entrada x

        Returns:
            float: saída da função identidade I(x) = x
        """
        
        return x