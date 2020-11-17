import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as lin

# Create a helper function - mathematical defenition of norm

p_norm_helper = lambda v, p: np.power((sum(abs(np.power(v, p)))), 1 / p)


# Python has a problem when comes to take a root of large numbers
# So let's normalize our vector so maximal value Python has take a root will be 1
# Divide it by maximum of a vector and then call to helper


def norm(v, p=2):
    max_val = max(abs(v))
    return p_norm_helper(v / max_val, p) * max_val


v = np.array([1, -2, 3, 1, 5])
norm(v, 2)

for i in range(1, 11):
    print("%4d-norm = %6.3f" % (i, norm(v, i)))
print("infinity norm is: %.3f" % (norm(v, np.inf)))

v = np.array([1, 5, 2, -2, -1, 7])
normalized = v / norm(v, np.inf)
print(normalized)

# norm(normalized, np.inf) = 1; check


v1 = np.array([0, 7, -15, 2, 7])
v2 = np.array([1, 3, -2, -3, 5])
d = norm(v1 - v2, np.inf)
print(d)

def generate_tamplate():
    X = np.zeros([8, 8])
    X[1, 1:3] = 1
    X[1, 5:7] = 1
    X[3, 3:5] = 1
    X[5:7, 1::5] = 1
    X[6, 2:6] = 1
    #A = np.logical_not(X)
    return X;




def similarity(A, B):
    inprod = lambda A, B: sum(sum(A * B))  # represents frobenious inner product
    res = inprod(A, B) / (lin.norm(A)*lin.norm(B))
    return res  # inprod/(norm_A*norm_B)




def random_face():
    t = generate_tamplate()
    X = np.zeros((8,8))
    X[1:7, 1:7] = np.random.randint(0,2, size=(6,6))
    while sum(sum(X)) < 0.7*sum(sum(t)):
        X[1:7, 1:7] = np.random.randint(0, 2, size=(6, 6))
    return X

def display(tamplate, agent, coef):
    plt.subplot(1,2,1)
    plt.imshow(tamplate, cmap='gray')
    plt.subplot(1,2,2)
    plt.imshow(agent, cmap='gray')
    plt.title("Similarity coefficient is %7.5f" %coef)
    plt.show()


def gate_check():
    coef = 0 #similarity coefficient
    threshold = 0.7 #thresh hold
    tamplate = generate_tamplate()
    counter = 0

    while(coef <= threshold):
        agent = random_face()
        coef = similarity(tamplate, agent)
        display(tamplate,agent, coef)
        counter+=1
        print("Access denied")

    plt.title('Access permitted')
    print("Access permitted after %d times" %counter)
    return

gate_check()
