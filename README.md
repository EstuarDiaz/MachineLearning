{% include mathjax.html %}
# Introducción al modelo de aprendizaje PAC
Ejercicio 3 del capítulo 2 de Understanding machine learning. [Ver el código del programa](https://github.com/EstuarDiaz/MachineLearning/blob/master/PAC.ipynb)

### Especificamos una parametrización del espacio de hipótesis.
Sea $$H = \{R(x_1,y_1,x_2,y_2) : x_1=y_1, 0 \leq x_1 <x_2 \leq 2,  0 \leq y_1 <y_2 \leq 2\}$$

### ¿Cuál es el tamaño del espacio de hipótesis?
Sea $$\Delta = 0.01$$. Calculemos $$|H|$$:
 - Tenemos que $$x_1 = y_1$$ pueden tomar $$2/\Delta$$ valores
 - Luego, $$x_2,y_2$$ pueden tomar $$\frac{2-x_1}{\Delta}$$ valores
Entonces $$|H| = \Big(\sum_{k=0}^{2/\Delta-1} \frac{2-k\Delta}{\Delta}\Big)^2 =  \Big(\sum_{k=0}^{2/\Delta-1} \frac{2}{\Delta} - k\Big)^2 = \Big(\frac{2}{\Delta}\sum_{k=0}^{2/\Delta-1} 1 - \sum_{k=0}^{2/\Delta-1} k\Big)^2 $$ $$=  \Big(\frac{2}{\Delta}^2  - \frac{2/\Delta(2/\Delta -1)}{2}\Big)^2 = \Big(\frac{4}{\Delta^2} - \frac{2/\Delta - 1}{\Delta}\Big)^2 = \Big(\frac{4-2+\Delta}{\Delta^2} \Big)^2 = \Big(\frac{2+\Delta}{\Delta^2} \Big)^2$$
Por lo tanto, $$|H| = \Big(\frac{2+\Delta}{\Delta^2} \Big)^2 = \Big(\frac{2+0.01}{0.01^2} \Big)^2 = 404,010,000$$

### Calculamos la probabilidad de extraer un ejemplo en cada una de las clases.
Por un lado, $$ P(x \in X_1) = \frac{(\frac{2}{\sqrt 2})^2}{2^2} = \frac{1}{2}$$ Por otro lado, $$P(x \in X_0) = 1 - P(x \in X_1) = 1 - \frac{1}{2} = \frac{1}{2}$$ 

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
### ¿Cuál es el tamaño de $$H$$? 
La nueva clase de hipótesis esta definida por $$H = \{R(x_1,y_1,x_2,y_2) : x_1=y_1,x_2 = y_2, x_2 = 2 - x_1, y_2 = 2-y_1\}$$
Entonces $$|H| = \frac{2}{2\Delta} = \frac{1}{\Delta} = \frac{1}{0.01} = 100$$.

La complejidad de la muestra $$mH$$ para $$\delta = 0.05, \epsilon = 0.01$$ es $$mH = ln(|H|/\delta)/\epsilon = ln(100/0.05)/0.01 \approx 760$$
Formamos un conjunto de entrenamiento con $$m$$ elementos, para $$m=e^0,e^{1/2},e^1,…,mH$$, donde $$mH$$ es la complejidad de $$S$$ para un parámetro de confianza y un error de 0.01. Graficamos los puntos de cada conjunto de entrenamiento. 
Graficamos los errores de generalización $$L(D,f)$$ y empírico $$LS$$ para cada hipótesis en $$H$$. Obtén algún rectángulo hS que minimice $$LS$$ y grafícalo en $$X$$.

### Para cada valor de $$m$$, forma 1000 conjuntos de entrenamiento, selecciona $$hS$$ en cada caso y haz un histograma del error $$L(D,f)$$.

### Si seleccionas un valor $$\epsilon$$ del error de generalización $$L(D,f)$$, ¿cuántos puntos del histograma tienen error de generalización menor a $$\epsilon$$?

### ¿Qué dice en este caso el resultado de que las clases de hipótesis finitas son PAC aprendibles con complejidad de muestra $$log(|H|/\delta)/\epsilon$$?
