# -*- coding: utf-8 -*-

#Initialize Learning rate, bias and weights
lr = 1
bias = 1
weights = [-50, 20, 20]

#Perceptron (x_1, x_2, output_A)
def Perceptron(x_1, x_2, output_A):
    output_P = bias*weights[0] + x_1*weights[1] + x_2*weights[2]
    if output_P > 4:
        output_P = 1
    else:
        output_P = 0
    
    error = 1/2*(output_A - output_P)**2
    weights[0] = weights[0] + error*bias*lr
    weights[1] = weights[1] + error*x_1*lr
    weights[2] = weights[2] + error*x_2*lr
    
def Predict(x_1, x_2):
    output_P =  bias*weights[0] + x_1*weights[1] + x_2*weights[2]
    if output_P > 4:
        output_P = 1
    else:
        output_P = 0
    return output_P

for i in range(20):
    Perceptron(0,0,0)
    Perceptron(0,1,1)
    Perceptron(1,0,1)
    Perceptron(1,1,1)
    print("Weights:", weights)

print("Weights:", weights)
x_1 = int(input("Enter first input"))  
x_2 = int(input("Enter second input"))
output_pred = Predict(x_1, x_2)
print(x_1, "or", x_2, "-->:", output_pred)



