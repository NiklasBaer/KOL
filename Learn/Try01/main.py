
inputs = [1, 2, 3, 2.5]

wieghts1 = [0.2, 0.8, -0.5, 1.0]
wieghts2 = [0.5, -0.91, 0.26, -0.5]
wieghts3 = [-0.26, -0.27, 0.17, 0.87]

bias1 = 2
bias2 = 3
bias3 = 0.5

output = [inputs[0] * wieghts1[0] + inputs[1] * wieghts1[1]+ inputs[2] * wieghts1[2] + inputs[3] * wieghts1[3] + bias1, 
          inputs[0] * wieghts2[0] + inputs[1] * wieghts2[1]+ inputs[2] * wieghts2[2] + inputs[3] * wieghts2[3] + bias2,
          inputs[0] * wieghts3[0] + inputs[1] * wieghts3[1]+ inputs[2] * wieghts3[2] + inputs[3] * wieghts3[3] + bias3]

print(output)