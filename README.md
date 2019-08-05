{% include mathjax.html %}
# Introducción al modelo de aprendizaje PAC
Ejercicio 3 del capítulo 2 de Understanding machine learning. [Ver el código del programa](https://github.com/EstuarDiaz/MachineLearning/blob/master/PAC.ipynb)

### Especificamos una parametrización del espacio de hipótesis.

### ¿Cuál es el tamaño del espacio de hipótesis?

### Calculamos la probabilidad de extraer un ejemplo en cada una de las clases.

### En python, definimos un conjunto de métodos que realizan las siguientes tareas: 
a) Formar conjuntos de entrenamiento $$S$$ con $$m$$ elementos
```markdown
# Obtener una muestra de m elementos
def getSample(m):
    return np.random.uniform(0,2,(m,2))
```
b) Implementar una hipótesis $$h:X \rightarrow \{0,1\}$$ para cada $$h$$ en $$X$$
```markdown
# Obtener la parametrizacion de la clase de hipotesis
def getHypothesisClass(l):
    return np.linspace(0,1,round(1/l))
```
c) Calcular errores empíricos y de generalización y 
```markdown
# Error empirico
def LS(h,S):
    error = 0
    d = 1/np.sqrt(2)
    for i in range(len(S)):
        # Si h lo calsifica como elemento de X1
        inX0 = ((S[i][0] < 1-d) or (S[i][1] < 1-d) or (S[i][0] > 1+d) or (S[i][1] > 1+d)) 
        if ((S[i][0] > h) and (S[i][1] > h) and (S[i][0] < 2-h) and (S[i][1] < 2-h)):
            # Y no pertenece a X1
            if inX0:
                error = error+1
        # Si h lo clasifica como elemento de X0
        else:
            # Pero no pertenece a X0
            if not inX0:
                error = error+1
    return error/len(S)
# Error de generalizacion
def LD(h):
    lim = 1-1/np.sqrt(2)
    if h < lim:
        return 1/2-2*h+h**2
    elif h > lim:
        return -1/2+2*h-h**2
    else:
        return 0
```
d) Seleccionar al rectángulo ERMH.
```markdown
# Obtener una hipotesis que minimize el error empirico
def ERM(H,S):
    min_error = 1
    EMRh = 0
    for h in np.nditer(H):
        Ls_error = LS(h,S) 
        if Ls_error < min_error:
            min_error = Ls_error
            EMRh = h
    return EMRh
```

### Simplifiquemos la clase de hipótesis finitas a rectángulos con lados paralelos a los lados de $$X$$, centrados en $$X$$. 

¿Cuál es el tamaño de $$H$$? 
Formamos un conjunto de entrenamiento con $$m$$ elementos, para $$m=e^0,e^{1/2},e^1,…,mH$$, donde $$mH$$ es la complejidad de $$S$$ para un parámetro de confianza y un error de 0.01. Graficamos los puntos de cada conjunto de entrenamiento. 
Graficamos los errores de generalización $$L(D,f)$$ y empírico $$LS$$ para cada hipótesis en $$H$$. Obtén algún rectángulo hS que minimice $$LS$$ y grafícalo en $$X$$.

### Para cada valor de $$m$$, forma 1000 conjuntos de entrenamiento, selecciona $$hS$$ en cada caso y haz un histograma del error $$L(D,f)$$.

### Si seleccionas un valor $$\epsilon$$ del error de generalización $$L(D,f)$$, ¿cuántos puntos del histograma tienen error de generalización menor a $$\epsilon$$?

### ¿Qué dice en este caso el resultado de que las clases de hipótesis finitas son PAC aprendibles con complejidad de muestra $$log(|H|/\delta)/\epsilon$$?
