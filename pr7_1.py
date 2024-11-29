import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import re
import string

# 1. Ініціалізація даних та параметрів

# Зразковий набір даних
texts = [
    "I am very happy with the service.",
    "The support was terrible and unhelpful.",
    "Excellent experience, will come back again!",
    "I'm not satisfied with the product quality.",
    "Great customer service and friendly staff.",
    "Bad experience, the issue was not resolved.",
    "Loved the quick response and professionalism.",
    "Very disappointed with the handling of my complaint.",
    "The team was awesome and very supportive.",
    "Not happy with the delays and poor communication."
]

labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1: Задоволений, 0: Незадоволений

# Функція попередньої обробки тексту
def preprocess_text(text):
    text = text.lower()  # Перетворення тексту в нижній регістр
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)  # Видалення пунктуації
    text = re.sub(r'\d+', '', text)  # Видалення цифр
    return text

# Застосування попередньої обробки до всіх текстів
texts = [preprocess_text(text) for text in texts]

# Векторизація тексту за допомогою TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts).toarray()  # Перетворення тексту в числовий формат
y = np.array(labels)  # Перетворення міток в масив NumPy

# Розділення даних на навчальну та тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Застосування PCA для зменшення розмірності до 2D для візуалізації
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)  # Навчання PCA на навчальних даних та трансформація
X_test_pca = pca.transform(X_test)  # Трансформація тестових даних

# Ініціалізація вагів та зсуву для 2D даних
np.random.seed(42)
weights = np.zeros(X_train_pca.shape[1])  # Ваги ініціалізуються нулями
bias = 0.0  # Зсув ініціалізується нулем

# 2. Функція активації Сигмоїда
def sigmoid(z):
    return 1 / (1 + np.exp(-z))  # Логістична функція

# 3. Функція втрат Логістична втрати з L2 регуляризацією
def log_loss(y_true, y_pred, weights, lambda_reg):
    epsilon = 1e-15  # Маленьке значення для уникнення логарифму нуля
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Обмеження прогнозів
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))  # Основна логістична втрати
    loss += lambda_reg * np.sum(weights ** 2)  # Додавання L2 регуляризації
    return loss

# 4. Градієнтний спуск з L2 регуляризацією
def gradient_descent(X, y, weights, bias, learning_rate, epochs, lambda_reg):
    losses = []  # Список для збереження значень втрат на кожній ітерації
    for epoch in range(epochs):
        linear_model = np.dot(X, weights) + bias  # Лінійна комбінація
        y_pred = sigmoid(linear_model)  # Прогнозування за допомогою сигмоїди
        loss = log_loss(y, y_pred, weights, lambda_reg)  # Обчислення втрат
        losses.append(loss)  # Додавання втрат до списку

        # Обчислення градієнтів
        dw = np.dot(X.T, (y_pred - y)) / len(y) + 2 * lambda_reg * weights
        db = np.mean(y_pred - y)

        # Оновлення параметрів
        weights -= learning_rate * dw
        bias -= learning_rate * db

        # Виведення втрат на певних ітераціях
        if (epoch+1) % 100 == 0 or epoch == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}')
    return weights, bias, losses

# 5. Навчання моделі на PCA-трансформованих даних
learning_rate = 0.1  # Швидкість навчання
epochs = 1000  # Кількість епох
lambda_reg = 0.01  # Параметр регуляризації

weights, bias, losses = gradient_descent(X_train_pca, y_train, weights, bias, learning_rate, epochs, lambda_reg)

# 6. Оцінка моделі
def accuracy(X, y, weights, bias):
    y_pred = sigmoid(np.dot(X, weights) + bias) >= 0.5  # Класифікація за порогом 0.5
    return np.mean(y_pred == y)  # Обчислення точності

accuracy_train = accuracy(X_train_pca, y_train, weights, bias)  # Точність на навчальних даних
accuracy_test = accuracy(X_test_pca, y_test, weights, bias)  # Точність на тестових даних
print(f'Training Accuracy: {accuracy_train * 100:.2f}%')
print(f'Test Accuracy: {accuracy_test * 100:.2f}%')

# Візуалізація функції втрат протягом епох
plt.figure(figsize=(10,6))
plt.plot(range(1, epochs+1), losses, label='Loss')  # Побудова графіка втрат
plt.xlabel('Epochs')  # Підпис осі X
plt.ylabel('Loss')  # Підпис осі Y
plt.title('Loss over Epochs')  # Заголовок графіка
plt.legend()
plt.show()

# 7. Візуалізація межі класифікації
plt.figure(figsize=(10,6))
plt.scatter(X_train_pca[y_train==1, 0], X_train_pca[y_train==1, 1], label='Задоволений')  # Точки класу 1
plt.scatter(X_train_pca[y_train==0, 0], X_train_pca[y_train==0, 1], label='Незадоволений')  # Точки класу 0

# Обчислення межі класифікації
x_values = np.array([X_train_pca[:,0].min()-1, X_train_pca[:,0].max()+1])
y_values = -(weights[0] * x_values + bias) / weights[1]
plt.plot(x_values, y_values, label='Межа класифікації', color='black')  # Лінія межі
plt.xlabel('PCA Component 1')  # Підпис осі X
plt.ylabel('PCA Component 2')  # Підпис осі Y
plt.title('Межа бінарної класифікації')  # Заголовок графіка
plt.legend()
plt.show()

# 8. Аналіз параметрів
learning_rates = [0.01, 0.1, 0.5]  # Різні швидкості навчання
lambda_regs = [0, 0.01, 0.1]  # Різні параметри регуляризації
plt.figure(figsize=(10,6))
for lr in learning_rates:
    for lr_reg in lambda_regs:
        _, _, temp_losses = gradient_descent(X_train_pca, y_train, np.zeros(X_train_pca.shape[1]), 0.0, lr, 100, lr_reg)
        plt.plot(temp_losses, label=f'LR: {lr}, Lambda: {lr_reg}')  # Побудова графіка для кожної конфігурації
plt.xlabel('Epochs')  # Підпис осі X
plt.ylabel('Loss')  # Підпис осі Y
plt.title('Втрати при різних швидкостях навчання та регуляризації')  # Заголовок графіка
plt.legend()
plt.show()

# 9. Стохастичний градієнтний спуск (SGD)
def sgd(X, y, weights, bias, learning_rate, epochs, lambda_reg):
    losses = []
    for epoch in range(epochs):
        indices = np.arange(len(X))
        np.random.shuffle(indices)  # Перемішування індексів
        for i in indices:
            xi = X[i]
            yi = y[i]
            linear_model = np.dot(xi, weights) + bias
            y_pred = sigmoid(linear_model)
            loss = - (yi * np.log(y_pred) + (1 - yi) * np.log(1 - y_pred)) + lambda_reg * np.sum(weights ** 2)
            losses.append(loss)
            # Обчислення градієнтів
            dw = (y_pred - yi) * xi + 2 * lambda_reg * weights
            db = y_pred - yi
            # Оновлення параметрів
            weights -= learning_rate * dw
            bias -= learning_rate * db
        # Виведення втрат на певних епохах
        if (epoch+1) % 100 == 0 or epoch == 0:
            print(f'SGD Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}')
    return weights, bias, losses

# Навчання моделі за допомогою SGD
weights_sgd, bias_sgd, losses_sgd = sgd(X_train_pca, y_train, np.zeros(X_train_pca.shape[1]), 0.0, 0.01, 1000, 0.01)
print(f'SGD Training Accuracy: {accuracy(X_train_pca, y_train, weights_sgd, bias_sgd) * 100:.2f}%')
print(f'SGD Test Accuracy: {accuracy(X_test_pca, y_test, weights_sgd, bias_sgd) * 100:.2f}%')

# 10. Міні-батч градієнтний спуск
def mini_batch_gd(X, y, weights, bias, learning_rate, epochs, batch_size, lambda_reg):
    losses = []
    for epoch in range(epochs):
        indices = np.arange(len(X))
        np.random.shuffle(indices)  # Перемішування індексів
        for start in range(0, len(X), batch_size):
            end = start + batch_size
            xb = X[start:end]  # Вибір міні-батчу
            yb = y[start:end]
            linear_model = np.dot(xb, weights) + bias
            y_pred = sigmoid(linear_model)
            loss = log_loss(yb, y_pred, weights, lambda_reg)
            losses.append(loss)
            # Обчислення градієнтів
            dw = np.dot(xb.T, (y_pred - yb)) / len(yb) + 2 * lambda_reg * weights
            db = np.mean(y_pred - yb)
            # Оновлення параметрів
            weights -= learning_rate * dw
            bias -= learning_rate * db
        # Виведення втрат на певних епохах
        if (epoch+1) % 100 == 0 or epoch == 0:
            print(f'Mini-Batch GD Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}')
    return weights, bias, losses

# Навчання моделі за допомогою міні-батч градієнтного спуску
weights_mbgd, bias_mbgd, losses_mbgd = mini_batch_gd(X_train_pca, y_train, np.zeros(X_train_pca.shape[1]), 0.0, 0.1, 1000, 2, 0.01)
print(f'Mini-Batch GD Training Accuracy: {accuracy(X_train_pca, y_train, weights_mbgd, bias_mbgd) * 100:.2f}%')
print(f'Mini-Batch GD Test Accuracy: {accuracy(X_test_pca, y_test, weights_mbgd, bias_mbgd) * 100:.2f}%')

# 11. Тестування та валідація
def analyze_errors(X, y, weights, bias):
    y_pred = sigmoid(np.dot(X, weights) + bias) >= 0.5  # Класифікація за порогом 0.5
    errors = X[y_pred != y]  # Визначення помилково класифікованих зразків
    return errors

errors = analyze_errors(X_test_pca, y_test, weights, bias)
print(f'Number of Misclassified Samples: {len(errors)}')

# Візуалізація помилково класифікованих зразків
plt.figure(figsize=(10,6))
plt.scatter(X_test_pca[y_test==1, 0], X_test_pca[y_test==1, 1], label='Задоволений')  # Точки класу 1
plt.scatter(X_test_pca[y_test==0, 0], X_test_pca[y_test==0, 1], label='Незадоволений')  # Точки класу 0
if len(errors) > 0:
    plt.scatter(errors[:,0], errors[:,1], facecolors='none', edgecolors='r', label='Помилки')  # Помилково класифіковані точки
plt.xlabel('PCA Component 1')  # Підпис осі X
plt.ylabel('PCA Component 2')  # Підпис осі Y
plt.title('Помилково класифіковані зразки')  # Заголовок графіка
plt.legend()
plt.show()