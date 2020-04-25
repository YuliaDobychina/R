# Установка необходимых пакетов
install.packages('devtools')
library('devtools')
devtools::install_github("rstudio/tensorflow",force = TRUE)
devtools::install_github("rstudio/keras",force = TRUE)

library('keras')
library('tensorflow')

install_keras()

# загрузка данных
mnist <- dataset_mnist()

# Разбиваем исходный массив данных на 4 части
train_images <- mnist$train$x
train_labels <- mnist$train$y
test_images <- mnist$test$x
test_labels <- mnist$test$y

# строим архитектуру нейронной сети

network <- keras_model_sequential() %>%
layer_dense(units = 512, activation = 'relu', input_shape = c(28*28)) %>%
layer_dense(units = 10, activation = 'softmax')

# Добавляем для нейронной сети оптимизатор, функцию потерь, какие метрики выводить на экран (в примере выводится только точность)
network %>% compile(
optimizer = 'rmsprop',
loss = 'categorical_crossentropy',
metrics = c('accuracy')
)

# Подготовка данных для обучения нейронной сети (изменение размерности)

train_images <- array_reshape(train_images, c(60000, 28*28))
train_images <- train_images/255
str(train_images)
test_images <- array_reshape(test_images, c(10000, 28*28))
test_images <- test_images/255

# создаем категории для ярлыков

train_labels <- to_categorical(train_labels)
test_labels <- to_categorical(test_labels)

# поле подготовки данных тренируем нейронную сеть

network %>% fit(train_images, train_labels, epochs = 30, batch_size = 128)

# точность модели составила 98,9%

metric <- network %>% evaluate(test_images, test_labels)
metric

# Предсказание определенных значений можно довольно просто:

network %>% predict_classes(test_images[1:15,])
test_labels1 <- mnist$test$y
test_labels1[1:15] # Выводим реальные значения и ссравниваем с предсказанными с помощью глаз)


# обучение сетей с добавлением валидации (появляется интерактивны график с точками каждой эпохи)
history <- network %>% fit(train_images, train_labels,
epochs = 5, batch_size = 128,
validation_split = 0.2)
# простой точечный график процесса обучения
plot(history)

# Рисуем картинку числа из массива
a <- mnist$test$x[5, 1:28, 1:28]
image(as.matrix(a))

# Предсказание одного числа из массива
test_a <- array_reshape(a, c(1, 28*28))
test_a <- test_a/255
network %>% predict_classes(test_a)

# Прогноз с 1 по 10 - с помощью цикла из общего массива данных последовательно берутся первые 10 матриц, а затем дня них расчитывается прогноз.
# Далее пары значений заполняются в таблицу REZULT
for(i in 1:10){
a <- mnist$test$x[i, 1:28, 1:28]
test_a <- array_reshape(a, c(1, 28*28))
test_a <- test_a/255

if(i==1){
test1 <- network %>% predict_classes(test_a)
fact <- test_labels1[1]
REZULT <- data.frame(Прогноз = test1, Факт = fact)
}
else{
temp <- c(network %>% predict_classes(test_a),test_labels1[i])
REZULT <- rbind(REZULT, temp)
}
}
#Результат с 1 по 10
REZULT
temp_size <-dim(test_labels1)-9

# Прогноз последних 10
for(j in temp_size:(temp_size+9)){
a <- mnist$test$x[j, 1:28, 1:28]
test_a <- array_reshape(a, c(1, 28*28))
test_a <- test_a/255
temp <- c(network %>% predict_classes(test_a),test_labels1[j])
REZULT <- rbind(REZULT, temp)
}

#Результат с 1 по 10 + последние 10
REZULT
