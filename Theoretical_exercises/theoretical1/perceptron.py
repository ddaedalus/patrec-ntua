import numpy as np
import matplotlib.pyplot as plt

X_train = np.array([[1,-1,4],[1,1,2],[1,2,-2],[1,1,-4],[1,4,-1],
                    [1,-4, 2],[1,-2,1],[1,-2,-1],[1,-1,-3],[1,-1,-6]])

y_train = np.array([1,1,1,1,1,2,2,2,2,2])

def perceptron(w_cur, X, r, y):
    while (True):
        changed = False
        for i in range(len(X_train)):
            if (y[i] == 1 and np.dot(w_cur, X[i]) <= 0):
                w_next = w_cur + r * X[i]                
                changed = True

            elif (y[i] == 2 and np.dot(w_cur, X[i]) >= 0):
                w_next = w_cur - r * X[i]  
                changed = True

            else:
                w_next = w_cur        

            w_cur = w_next
            print(w_cur)
        
        if (not changed):
            break
        
    return w_cur

w = np.array([0,0,0])
print(perceptron(w, X_train, 1, y_train))


X_train = np.array([[-1,4],[1,2],[2,-2],[1,-4],[4,-1],
                    [-4, 2],[-2,1],[-2,-1],[-1,-3],[-1,-6]])
X0, X1 = X_train[:,0], X_train[:,1]
colors = ['red', 'yellow']
fig, ax = plt.subplots()
for label in range(1,3):
    ax.scatter(
        X0[y_train == label], X1[y_train == label],
        c=(colors[int(label)-1]), label=int(label-1),
        s=60, alpha=0.9, edgecolors='k'
    )
    
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_title('Linear Discriminant Analysis')
plt.show()