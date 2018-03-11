import numpy as np
from sklearn.metrics import confusion_matrix

def distance(a,b):
    return(np.linalg.norm(a-b))

def check_neighbors(neigh,full_list):
    n = len(neigh)
    vivo = 0
    dead = 0
    for i in range(n):
        x = full_list[neigh[i],0]
        if x == 0:
            dead += 1
        else:
            vivo += 1
    if vivo > dead:
        return(1)
    else:
        return(0)



### Archivo de entrada ###
data = np.genfromtxt('train_data_1.csv',delimiter=',', skip_header=1,dtype=float)

### Ordenar al azar los datos ###
random = list(range(np.shape(data)[0]))
np.random.shuffle(random)
data = data[random,:]

### Dividir en train/test 80:20 ###
split = int(len(data)*0.8)

train = data[:split,:]
test = data[split:,:]
test_in = test[:,1:]
test_out = test[:,0]


### Parámetros ###
k = 3
n = len(train)
distancias = np.arange(n).reshape(n,1)
zeros = np.zeros(n).reshape(n,1)
distancias = np.concatenate((distancias,zeros),axis=1)
predicted_out = np.zeros(len(test_in)).astype(int)


### Calcular todas las distancias ###

for i in range(len(test_in)):
    #print('{}/{}'.format(i,len(test_in)))
    for j in range(n):
        distancias[j,1] = distance(test_in[i],train[j,1:])
    distancias = distancias[distancias[:,1].argsort()]
    neighbors = distancias[:k,0].astype(int)
    predicted = check_neighbors(neighbors,train)
    predicted_out[i] = predicted
conf = confusion_matrix(test_out,predicted_out)
print(conf)
print(len(test))
#print('Error k={}: {:.2f}%'.format(k,acc))