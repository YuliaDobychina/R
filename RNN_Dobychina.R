library('keras')
library('tensorflow')


max_features <- 10000


# загрузка данных mnist
mnist <- dataset_mnist()
str(mnist)

# Подготовка данных к анализу
input_train <- mnist$train$x
y_train <- mnist$train$y
input_test <- mnist$test$x
y_test <- mnist$test$y

# Сокращение размерности данных
input_train <- array_reshape(input_train, c(60000, 28*28))
input_train <- input_train/255
str(input_train)


# Построение рекуррентной нейронной сети
model <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_features, output_dim = 32) %>%
  layer_simple_rnn(units = 32) %>%
  layer_dense(units = 1, activation = "sigmoid")
model %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("acc")
)

history <- model %>% fit(
  input_train, y_train,
  epochs = 5,
  batch_size = 128,
  validation_split = 0.2
)

#Оценка результатов
plot(history)
