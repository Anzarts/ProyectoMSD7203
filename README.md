# ProyectoMSD7203

**Benjamín Farías**
**Sebastián Sanhueza**

En este repo se implementa un modelo HVAE para generación de imágenes.

### Secciones

 - **Dataset:** se descargan las imágenes de CAPTCHA y se construyen un dataset y dataloader de Pytorch.
 - **HVAE**: módulos de Pytorch que implementan el modelo HVAE.
 - **Train Faunction**: función de entrenamiento utilizada para ajustar los modelos.
 - **Generation**: funciones para samplear y visualizar imágenes.
 - **Run Model**: carga un HVAE ya entrenado y genera una muesra de imágenes.


 ### Intrucciones

La clase HVAE construye un modelo recibiendo su arquitectura como
parámetros. El argumento `outer_config` define los bloques de entrada y salida ($q_\phi(z_1|x)$ y $p_\theta(x|z_1)$) por medio de listas de tuplas de la forma:

             [('conv', 32), ('conv-bn', 32), ('pool', 2)]


donde
 - `('conv', 32)` define una capa de convolución con 32 filtros.
 - `('conv-bn', 32)` define una capa de convolución con 32 filtros, seguida de batch normalization.
 - `('pool', 2)` define una capa de maxpooling con kernel 2.


 para el bloque de entrada, apiladas en el mismo orden. El bloque de salida se construye espejado al de entrada, de forma tal que se invierte el orden, por cada convolución de $c_{in}$ a $c_{out}$ canales se añade una convolución traspuesta de $c_{out}$ a $c_{in}$ canales y por cada maxpooling se añade una capa de upsample.

 El argumento `inner_config` recibe una lista de listas como la `outer_config` para definir los bloques internos. Aquí utiliza `'convt'` para crear convoluciones traspuestas y `'upsample'` para upsamples, también se añaden skip connections, `'skip-up'` para hacer un salto y `'skip-down'` para recibirlo, acompañadas de un índice que indica qué capas conectar. Los bloques internos de inferencia y generación se construyen de forma idéntica.

 La clase tiene por defecto los parámetros de la configuración que fue entrenada para el sampleo.