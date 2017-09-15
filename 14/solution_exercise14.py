import numpy as np

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    # TODO: Compute and return softmax(x)
    numerador = np.exp(x)
    suma = sum(numerador)
    return (1/suma) * numerador

logits = [3.0, 1.0, 0.2]
print(softmax(logits))

