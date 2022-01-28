
f = open('label.txt', 'r')

labelBuffer = (f.read()).split('\n')

f.close()
print(labelBuffer)
