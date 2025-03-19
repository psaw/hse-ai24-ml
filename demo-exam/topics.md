# Cheat-sheet по темам демо-экзамена
Этот документ предоставляет самые ключевые знания по перечисленным темам. Все утвеждения и формулы приводятся без доказательств. Цель документа - мини-справочник для использования на экзамене.

## 1. Введение в машинное обучение

### Основные понятия
- Объект (instance, example) - $x_i$
- Признаки (features) - $f_j(x)$
- Целевая переменная (target) - $y_i$
- Выборка (dataset) - $\{(x_i, y_i)\}_{i=1}^l$
- Модель (model) - $a(x)$
- Пространство признаков (feature space) - $\mathcal{X}$
- Пространство ответов (target space) - $\mathcal{Y}$

### Типы признаков
- Количественные (numerical)
- Категориальные (categorical)
- Порядковые (ordinal) - имеют естественный порядок, но интервалы между значениями не обязательно равны (в отличие от количественных)
- Бинарные (binary)

### Типы задач машинного обучения

#### Обучение с учителем (supervised learning)
- Регрессия: $\mathcal{Y} = \mathbb{R}$
  - Прогноз числовых значений
  - Примеры: предсказание цен, прогноз температуры
- Бинарная классификация: $\mathcal{Y} = \{0,1\}$ или $\{-1,+1\}$
  - Разделение объектов на два класса
  - Примеры: спам-фильтрация, определение мошенничества
- Многоклассовая классификация: $\mathcal{Y} = \{1,\ldots,K\}$
  - K > 2 непересекающихся классов
  - Примеры: распознавание цифр, классификация товаров
- Многоклассовая классификация с пересечениями: $\mathcal{Y} = \{0,1\}^K$
  - Объект может принадлежать нескольким классам
  - Примеры: теги статей, жанры фильмов

#### Обучение без учителя (unsupervised learning)
- Кластеризация
  - Разбиение выборки на группы схожих объектов
  - Нет целевой переменной
- Понижение размерности
  - Сжатие данных с сохранением структуры
  - Визуализация многомерных данных
- Поиск аномалий
  - Обнаружение нетипичных объектов
- Оценивание плотности
  - Моделирование распределения данных

#### Частичное обучение (semi-supervised learning)
- Часть выборки размечена, часть - нет
- Используется вся доступная информация
- Примеры: автоматическая разметка данных

### Основные принципы
- Принцип максимума правдоподобия
- Принцип минимизации эмпирического риска
- Принцип структурной минимизации риска
- Компромисс между сложностью модели и качеством аппроксимации (bias-variance trade-off)

#### bias-variance trade-off
- Разложение ошибки модели на компоненты:
  - Ошибка = Смещение² + Дисперсия + Шум
  - $\mathbb{E}[(y - \hat{f}(x))^2] = [Bias(\hat{f}(x))]^2 + Var(\hat{f}(x)) + \sigma^2$

- Смещение (bias):
  - Систематическая ошибка модели
  - Насколько в среднем предсказания отклоняются от истинных значений
  - Высокое смещение → недообучение (underfitting)
  - Причины: слишком простая модель или сильные предположения о данных

- Дисперсия (variance):
  - Разброс предсказаний при небольших изменениях в обучающих данных
  - Насколько сильно модель реагирует на шум в данных
  - Высокая дисперсия → переобучение (overfitting)
  - Причины: слишком сложная модель или недостаточно данных

- Компромисс:
  - Нельзя одновременно минимизировать и смещение, и дисперсию
  - Уменьшение одного обычно ведёт к увеличению другого
  - Оптимальная модель находит баланс между ними

- Практические способы управления:
  - Регуляризация (λ↑ => bias↑, variance↓)
  - Размер модели (размер↑ => bias↓, variance↑)
  - Объем данных (объем↑ => variance↓)
  - Количество признаков (кол-во↑ => variance↑)
  - Ансамблирование моделей:
    - Бэггинг (bagging) => variance↓, bias≈const
    - Бустинг (boosting) => bias↓, может увеличить variance
    - Стекинг (stacking) => bias↓, variance↓ при правильной настройке
    - Случайный лес (random forest) => variance↓, bias≈const или небольшое↑

- Признаки проблем:
  - Высокое смещение: большая ошибка на тренировочных данных
  - Высокая дисперсия: большая разница между ошибками на тренировочных и тестовых данных

### Проблемы обучения
- Переобучение (overfitting) - модель слишком хорошо подстраивается под обучающие данные, но плохо обобщается на новые данные
- Недообучение (underfitting) - модель слишком проста и не может уловить закономерности в данных
- Смещение и разброс (bias-variance) - компромисс между систематической ошибкой модели и её чувствительностью к случайным колебаниям в данных
- Проклятие размерности - с ростом числа признаков экспоненциально растёт объём пространства, что требует больше данных для обучения

### Процесс обучения
1. Сбор и подготовка данных
2. Выбор модели и метрики качества
3. Обучение модели
4. Оценка качества и валидация
5. Внедрение модели

## 2. Математические основы

### Линейная алгебра
- Определитель и ранг матриц:
  - Определитель: $\det(A)$ или $|A|$ - скалярная величина, связанная с обратимостью матрицы
  - Ранг матрицы: максимальное число линейно независимых строк или столбцов матрицы
  - Свойства ранга:
    - $0 \leq \text{rank}(A) \leq \min(m,n)$ для матрицы $A$ размера $m \times n$
    - $\text{rank}(A) = \text{rank}(A^T)$
    - $\text{rank}(AB) \leq \min(\text{rank}(A), \text{rank}(B))$
- Обратная матрица: $\mathbf{A}\mathbf{A}^{-1} = \mathbf{I}$
- Собственные значения и векторы:
  - $\mathbf{A}\mathbf{v} = \lambda\mathbf{v}$, где $\lambda$ - собственное значение, $\mathbf{v}$ - собственный вектор
  - Практическое применение:
    - Разложение ковариационной матрицы в PCA (анализ главных компонент)
    - Спектральная кластеризация данных
    - Сжатие данных и понижение размерности
    - Решение систем дифференциальных уравнений
    - Анализ устойчивости динамических систем
    - Вычисление матричных функций (например, $e^A$)

### Оптимизация
- Градиент: $\nabla f = (\frac{\partial f}{\partial x_1}, \ldots, \frac{\partial f}{\partial x_n})^\top$
- Градиентный спуск:
  - $\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \nabla f(\mathbf{w}_t)$
  - $\eta$ - learning rate
- Методы оптимизации:
  - Стохастический градиентный спуск (SGD)
  - Mini-batch градиентный спуск
  - Momentum: $\mathbf{v}_{t+1} = \gamma\mathbf{v}_t + \eta\nabla f(\mathbf{w}_t)$
  - Adam: адаптивная оценка моментов
    - Сочетает идеи Momentum и RMSProp
    - Адаптивно настраивает скорость обучения для каждого параметра
    - Использует экспоненциальное скользящее среднее градиентов (первый момент)
    - Использует экспоненциальное скользящее среднее квадратов градиентов (второй момент)
    - Формулы обновления:
      - $m_t = \beta_1 m_{t-1} + (1-\beta_1) \nabla f(\mathbf{w}_t)$ (первый момент)
      - $v_t = \beta_2 v_{t-1} + (1-\beta_2) (\nabla f(\mathbf{w}_t))^2$ (второй момент)
      - $\hat{m}_t = \frac{m_t}{1-\beta_1^t}$, $\hat{v}_t = \frac{v_t}{1-\beta_2^t}$ (коррекция смещения)
      - $\mathbf{w}_{t+1} = \mathbf{w}_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$
    - Типичные значения гиперпараметров: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$
    - Преимущества: быстрая сходимость, устойчивость к выбору learning rate
- Условия оптимальности:
  - Необходимое: $\nabla f(\mathbf{w}^*) = 0$
  - Достаточное: матрица Гессе положительно определена

### Теория вероятностей
- Случайная величина и её характеристики:
  - Математическое ожидание: $\mathbb{E}[X]$
  - Дисперсия: $\mathbb{D}[X] = \mathbb{E}[(X - \mathbb{E}[X])^2]$
  - Ковариация: $\text{cov}(X,Y) = \mathbb{E}[(X - \mathbb{E}[X])(Y - \mathbb{E}[Y])]$
- Основные распределения:
  - Нормальное: $\mathcal{N}(\mu, \sigma^2)$
  - Бернулли: $\text{Ber}(p)$
  - Биномиальное: $\text{Bin}(n,p)$
- Условная вероятность:
  - Формула Байеса: $P(A|B) = \frac{P(B|A)P(A)}{P(B)}$

### Статистика
- Оценки параметров:
  - Несмещённость: $\mathbb{E}[\hat{\theta}] = \theta$
  - Состоятельность: $\hat{\theta} \xrightarrow{p} \theta$
- Метод максимального правдоподобия:
  - $\hat{\theta} = \arg\max_{\theta} \prod_{i=1}^n p(x_i|\theta)$
  - Log-правдоподобие: $\mathcal{L}(\theta) = \sum_{i=1}^n \log p(x_i|\theta)$
- Доверительные интервалы
- Проверка гипотез:
  - p-value
  - Ошибки I и II рода

### Математический анализ
- Производные и частные производные
- Градиент и матрица Якоби:
  - Градиент: $\nabla f = (\frac{\partial f}{\partial x_1}, \ldots, \frac{\partial f}{\partial x_n})^\top$ - вектор частных производных
  - Матрица Якоби: $J = \begin{pmatrix} 
    \frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\
    \vdots & \ddots & \vdots \\
    \frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n}
    \end{pmatrix}$ - обобщение градиента для векторнозначных функций
  - Градиент показывает направление наибольшего возрастания функции
  - Матрица Якоби содержит все частные производные первого порядка векторной функции
- Матрица Гессе:
  - $H = \begin{pmatrix} 
    \frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\
    \frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots & \frac{\partial^2 f}{\partial x_2 \partial x_n} \\
    \vdots & \vdots & \ddots & \vdots \\
    \frac{\partial^2 f}{\partial x_n \partial x_1} & \frac{\partial^2 f}{\partial x_n \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_n^2}
    \end{pmatrix}$ - матрица вторых частных производных
  - Используется для анализа критических точек функции
  - Если $H$ положительно определена в точке $x^*$, то $x^*$ - точка минимума
  - Если $H$ отрицательно определена в точке $x^*$, то $x^*$ - точка максимума
  - Если $H$ имеет как положительные, так и отрицательные собственные значения, то $x^*$ - седловая точка
  - Для выпуклых функций матрица Гессе положительно полуопределена
- Выпуклые функции:
  - $f(\alpha x + (1-\alpha)y) \leq \alpha f(x) + (1-\alpha)f(y)$
- Ряды Тейлора:
  - $f(x) = f(a) + f'(a)(x-a) + \frac{f''(a)}{2!}(x-a)^2 + \ldots$

## 3. Теория вероятностей и статистика
- Нормальное распределение и его свойства:
  - Плотность вероятности: $f(x) = \frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$
  - Параметры: $\mu$ - математическое ожидание, $\sigma^2$ - дисперсия
  - Свойства:
    - Симметричность относительно $\mu$
    - 68-95-99.7 правило: 68% значений лежат в пределах $\mu \pm \sigma$, 95% в пределах $\mu \pm 2\sigma$, 99.7% в пределах $\mu \pm 3\sigma$
    - Сумма нормальных случайных величин также имеет нормальное распределение
    - Центральная предельная теорема: сумма большого числа независимых случайных величин стремится к нормальному распределению
    - Стандартное нормальное распределение: $\mathcal{N}(0,1)$
- Математическое ожидание случайной величины:
  - Определение: $\mathbb{E}[X] = \sum_{i} x_i p(x_i)$ для дискретных случайных величин
  - Для непрерывных: $\mathbb{E}[X] = \int_{-\infty}^{\infty} x f(x) dx$
  - Свойства:
    - Линейность: $\mathbb{E}[aX + bY] = a\mathbb{E}[X] + b\mathbb{E}[Y]$
    - Для независимых величин: $\mathbb{E}[XY] = \mathbb{E}[X]\mathbb{E}[Y]$
  - Условное математическое ожидание: $\mathbb{E}[X|Y]$ - ожидание $X$ при фиксированном значении $Y$
  - Закон полного математического ожидания: $\mathbb{E}[X] = \mathbb{E}[\mathbb{E}[X|Y]]$
- Корреляция Пирсона:
  - Определение: $\rho_{X,Y} = \frac{\text{cov}(X,Y)}{\sigma_X \sigma_Y} = \frac{\mathbb{E}[(X-\mu_X)(Y-\mu_Y)]}{\sigma_X \sigma_Y}$
  - Измеряет линейную зависимость между случайными величинами
  - Значения от -1 до 1: 
    - 1 означает полную положительную линейную корреляцию
    - -1 означает полную отрицательную линейную корреляцию
    - 0 означает отсутствие линейной корреляции
  - Ограничения: выявляет только линейные зависимости
  - Чувствительна к выбросам
  - Формула выборочной корреляции: $r = \frac{\sum_{i=1}^{n}(x_i-\bar{x})(y_i-\bar{y})}{\sqrt{\sum_{i=1}^{n}(x_i-\bar{x})^2}\sqrt{\sum_{i=1}^{n}(y_i-\bar{y})^2}}$
- Квантили и медианы:
  - Квантиль порядка $p$ - значение, ниже которого находится доля $p$ элементов выборки
  - Медиана - квантиль порядка 0.5 (делит выборку пополам)
  - Квартили - квантили порядка 0.25, 0.5 и 0.75
  - Процентили - квантили, выраженные в процентах (от 0 до 100)
  - Межквартильный размах (IQR) - разница между 3-м и 1-м квартилями, мера разброса данных
  - Используются для:
    - Описания распределения данных
    - Выявления выбросов (значения вне диапазона $[Q_1 - 1.5 \cdot IQR, Q_3 + 1.5 \cdot IQR]$)
    - Построения диаграмм "ящик с усами" (box plot)
  - Эмпирическая функция распределения: $F_n(x) = \frac{1}{n}\sum_{i=1}^{n} I(X_i \leq x)$
- Распределение Бернулли:
  - Описывает случайный эксперимент с двумя возможными исходами (успех/неудача)
  - Вероятностная функция: $P(X=k) = p^k(1-p)^{1-k}$ для $k \in \{0,1\}$
  - Параметр $p$ - вероятность успеха
  - Математическое ожидание: $\mathbb{E}[X] = p$
  - Дисперсия: $\text{Var}(X) = p(1-p)$
  - Является частным случаем биномиального распределения с $n=1$
  - Применяется для моделирования бинарных событий (подбрасывание монеты, успех/неудача эксперимента)
- Биномиальное распределение:
  - Описывает число успехов в последовательности из $n$ независимых испытаний Бернулли
  - Вероятностная функция: $P(X=k) = \binom{n}{k} p^k (1-p)^{n-k}$, где $k \in \{0,1,2,...,n\}$
    - где $\binom{n}{k} = \frac{n!}{k!(n-k)!}$
  - Параметры: $n$ - число испытаний, $p$ - вероятность успеха в каждом испытании
  - Математическое ожидание: $\mathbb{E}[X] = np$
  - Дисперсия: $\text{Var}(X) = np(1-p)$
  - Свойства:
    - Сумма независимых биномиальных случайных величин с одинаковым $p$ также имеет биномиальное распределение
    - При больших $n$ и не экстремальных $p$ аппроксимируется нормальным распределением $\mathcal{N}(np, np(1-p))$
    - При малых $np$ аппроксимируется распределением Пуассона с параметром $\lambda = np$
  - Примеры применения: подсчет числа успешных испытаний, моделирование количества дефектных изделий, анализ результатов опросов
- Оценивание плотности распределения:
  - Параметрические методы:
    - Предположение о виде распределения (нормальное, экспоненциальное и т.д.)
    - Оценка параметров методом максимального правдоподобия
    - Преимущества: эффективность при правильном выборе модели
    - Недостатки: чувствительность к ошибкам спецификации модели
  - Непараметрические методы:
    - Гистограммы: разбиение диапазона данных на интервалы и подсчет частот
    - Ядерные оценки плотности (KDE): $\hat{f}(x) = \frac{1}{nh}\sum_{i=1}^{n}K\left(\frac{x-X_i}{h}\right)$
    - Выбор ширины окна (bandwidth) $h$ и функции ядра $K$
    - Метод k-ближайших соседей для оценки плотности
  - Смешанные модели (Mixture Models):
    - Представление плотности как взвешенной суммы базовых распределений
    - Алгоритм EM (Expectation-Maximization) для оценки параметров
  - Применения:
    - Визуализация распределения данных
    - Обнаружение аномалий и выбросов
    - Генерация новых данных
    - Байесовский вывод и принятие решений
- Обнаружение аномалий:
  - Определение: выявление наблюдений, значительно отклоняющихся от основного распределения данных
  - Методы обнаружения:
    - Статистические методы (z-score, модифицированный z-score, тест Граббса)
    - Методы на основе расстояний (LOF - Local Outlier Factor, DBSCAN)
    - Методы на основе плотности (Isolation Forest, One-Class SVM)
    - Методы на основе кластеризации (K-means с анализом размера кластеров)
  - Подходы к обработке аномалий:
    - Удаление аномальных значений
    - Замена аномалий (средними, медианами, предсказанными значениями)
    - Отдельная обработка аномалий как особого класса данных
  - Применения:
    - Обнаружение мошенничества в финансовых транзакциях
    - Выявление неисправностей в промышленных системах
    - Обнаружение вторжений в компьютерные сети
    - Выявление аномального поведения пользователей
  - Метрики оценки качества:
    - Precision, Recall, F1-score для задач с размеченными аномалиями
    - AUC-ROC и AUC-PR для оценки ранжирования аномалий

## 4. Предобработка данных
- Обработка пропущенных значений
  - Удаление строк или столбцов с пропусками
    - `dropna()` в pandas
    - ```python
      import pandas as pd

      df = pd.read_csv('your_data.csv')
      df_dropped_rows = df.dropna(axis=0)  # Удаление строк с пропусками
      df_dropped_cols = df.dropna(axis=1)  # Удаление столбцов с пропусками
      ```
  - Заполнение пропущенных значений (импутация)
    - Заполнение средним, медианой или модой
      - `fillna()` в pandas с `mean()`, `median()`, `mode()`
      - ```python
        mean_value = df['column_name'].mean()
        df['column_name'].fillna(mean_value, inplace=True)

        median_value = df['column_name'].median()
        df['column_name'].fillna(median_value, inplace=True)

        mode_value = df['column_name'].mode()[0] # mode() возвращает Series
        df['column_name'].fillna(mode_value, inplace=True)
        ```
    - Заполнение константным значением
      - `fillna()` в pandas с константой
      - ```python
        df['column_name'].fillna(0, inplace=True) # Заполнение нулем
        df['column_name'].fillna('missing', inplace=True) # Заполнение строкой
        ```
    - Импутация с использованием машинного обучения (например, KNNImputer)
      - `KNNImputer` из `sklearn.impute`
      - ```python
        from sklearn.impute import KNNImputer
        import numpy as np

        imputer = KNNImputer(n_neighbors=2) # 2 ближайших соседа
        df[['col1', 'col2']] = imputer.fit_transform(df[['col1', 'col2']])
        ```
  - Игнорирование пропущенных значений
    - Некоторые модели (например, CatBoost, LightGBM) могут обрабатывать пропуски нативно.
  - Создание индикатора пропущенных значений
    - Создание нового бинарного признака, указывающего на наличие пропуска
    - ```python
      df['column_name_missing'] = df['column_name'].isnull().astype(int)
      ```
- Кодирование категориальных признаков
  - Label Encoding (порядковое кодирование)
    - Принцип: каждой категории присваивается уникальное целое число. Подходит для порядковых признаков.
    - Python (scikit-learn):
      ```python
      from sklearn.preprocessing import LabelEncoder

      le = LabelEncoder()
      df['column_encoded'] = le.fit_transform(df['column'])
      ```
  - One-Hot Encoding (однократное кодирование)
    - Принцип: для каждой категории создается бинарный столбец. Подходит для номинальных признаков. Может привести к увеличению размерности данных.
    - Python (pandas):
      ```python
      df_encoded = pd.get_dummies(df, columns=['column'], prefix='column')
      ```
  - Binary Encoding (бинарное кодирование)
    - Принцип: категории кодируются в двоичный код. Уменьшает размерность по сравнению с One-Hot Encoding.
    - Python (category_encoders):
      ```python
      import category_encoders as ce

      encoder = ce.BinaryEncoder(cols=['column'])
      df_encoded = encoder.fit_transform(df)
      ```
  - Frequency Encoding (частотное кодирование)
    - Принцип: категории заменяются их частотой в наборе данных. Полезно, когда частота является информативной.
    - Python (pandas):
      ```python
      frequency_map = df['column'].value_counts(normalize=True).to_dict()
      df['column_encoded'] = df['column'].map(frequency_map)
      ```
  - Target Encoding (целевое кодирование)
    - Принцип: категории заменяются средним значением целевой переменной для этой категории. Может быть мощным, но склонен к переобучению.
    - Python (category_encoders):
      ```python
      import category_encoders as ce

      encoder = ce.TargetEncoder(cols=['column'])
      df_encoded = encoder.fit_transform(df, df['target']) # df['target'] - целевая переменная
      ```
- One-hot encoding (однократное кодирование)
    - Принцип: для каждой категории создается бинарный столбец. Подходит для номинальных признаков. Может привести к увеличению размерности данных.
    - Python (pandas):
      ```python
      df_encoded = pd.get_dummies(df, columns=['column'], prefix='column')
      ```
    - Python (scikit-learn):
      ```python
      from sklearn.preprocessing import OneHotEncoder

      ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore') # sparse=False для numpy array, handle_unknown='ignore' для новых категорий
      df_encoded = ohe.fit_transform(df[['column']])
      ```
- Корреляционный анализ
  - Принцип: оценка статистической взаимосвязи между двумя или более переменными. Коэффициент корреляции Пирсона измеряет линейную зависимость между двумя переменными. Значения коэффициента варьируются от -1 до 1, где 1 означает полную положительную корреляцию, -1 - полную отрицательную, а 0 - отсутствие линейной корреляции.
  - Python (pandas):
    ```python
    correlation_matrix = df.corr(method='pearson') # 'pearson', 'kendall', 'spearman'
    correlation_with_target = df.corr(method='pearson')['target_column']
    ```
  - Python (numpy):
    ```python
    import numpy as np

    correlation_coefficient = np.corrcoef(df['column1'], df['column2'])[0, 1]
    ```
- Создание новых признаков (Feature Engineering)
  - Принцип: создание новых признаков на основе существующих. Новые признаки могут быть получены путем преобразования, комбинирования или извлечения информации из исходных признаков. Цель - улучшить качество модели, предоставив ей более релевантные и информативные данные.
  - Примеры:
    - Полиномиальные признаки: создание новых признаков путем возведения существующих признаков в степень или перемножения их друг с другом. Полезно для улавливания нелинейных зависимостей.
      - Python (scikit-learn):
        ```python
        from sklearn.preprocessing import PolynomialFeatures
        import pandas as pd

        data = {'feature1': [1, 2, 3, 4, 5], 'feature2': [2, 3, 4, 5, 6]}
        df = pd.DataFrame(data)

        poly = PolynomialFeatures(degree=2, include_bias=False)
        poly_features = poly.fit_transform(df)
        df_poly = pd.DataFrame(poly_features, columns=poly.get_feature_names_out(df.columns))
        ```
    - Комбинация признаков (Interaction features): создание новых признаков путем комбинирования двух или более существующих признаков. Например, перемножение двух признаков для учета их взаимодействия.
      - Python (pandas):
        ```python
        df['interaction_feature'] = df['feature1'] * df['feature2']
        ```
    - Извлечение признаков из даты и времени: создание новых признаков, таких как день недели, месяц, год, час и т.д., из признаков даты и времени.
      - Python (pandas):
        ```python
        df['date_column'] = pd.to_datetime(df['date_column'])
        df['day_of_week'] = df['date_column'].dt.dayofweek
        df['month'] = df['date_column'].dt.month
        ```
- Масштабирование признаков
  - Принцип: приведение числовых признаков к одному масштабу.  Необходимо, когда признаки имеют разные диапазоны значений, что может негативно сказаться на работе некоторых алгоритмов машинного обучения (например, градиентного спуска, k-ближайших соседей, SVM). Масштабирование помогает алгоритмам быстрее сходиться и улучшает качество моделей.
  - Методы:
    - StandardScaler (стандартизация):
      - Преобразование: $x_{scaled} = \frac{x - \mu}{\sigma}$, где $\mu$ - среднее, $\sigma$ - стандартное отклонение.
      - Принцип: приводит данные к нулевому среднему и единичной дисперсии.
      - Python (scikit-learn):
        ```python
        from sklearn.preprocessing import StandardScaler
        import pandas as pd

        data = {'feature1': [1, 2, 3, 4, 5], 'feature2': [10, 20, 30, 40, 50]}
        df = pd.DataFrame(data)

        scaler = StandardScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
        ```
    - MinMaxScaler (нормализация):
      - Преобразование: $x_{scaled} = \frac{x - x_{min}}{x_{max} - x_{min}}$
      - Принцип: приводит данные к диапазону [0, 1].
      - Python (scikit-learn):
        ```python
        from sklearn.preprocessing import MinMaxScaler
        import pandas as pd

        data = {'feature1': [1, 2, 3, 4, 5], 'feature2': [10, 20, 30, 40, 50]}
        df = pd.DataFrame(data)

        scaler = MinMaxScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
        ```

## 5. Визуализация данных
- Основные виды визуализации датасетов:
  - **Гистограмма (Histogram)**
    - Принцип: Показывает распределение частот значений числовой переменной. Разбивает данные на интервалы (бины) и отображает количество значений, попадающих в каждый интервал.
    - Python (matplotlib):
      ```python
      import matplotlib.pyplot as plt
      plt.hist(df['column_name'])
      plt.show()
      ```
  - **Диаграмма рассеяния (Scatter plot)**
    - Принцип: Отображает взаимосвязь между двумя числовыми переменными. Каждая точка на графике представляет собой пару значений из двух переменных.
    - Python (matplotlib):
      ```python
      import matplotlib.pyplot as plt
      plt.scatter(df['column_name_x'], df['column_name_y'])
      plt.xlabel('column_name_x')
      plt.ylabel('column_name_y')
      plt.show()
      ```
  - **Коробчатая диаграмма (Box plot)**
    - Принцип: Показывает распределение числовой переменной для разных категорий. Отображает медиану, квартили, выбросы.
    - Python (seaborn):
      ```python
      import seaborn as sns
      sns.boxplot(x='categorical_column', y='numerical_column', data=df)
      plt.show()
      ```
  - **Столбчатая диаграмма (Bar chart)**
    - Принцип: Сравнивает значения категорий. Высота столбца пропорциональна значению категории.
    - Python (matplotlib):
      ```python
      import matplotlib.pyplot as plt
      plt.bar(df['categorical_column'].value_counts().index, df['categorical_column'].value_counts().values)
      plt.xlabel('categorical_column')
      plt.ylabel('Count')
      plt.show()
      ```
  - **Линейный график (Line plot)**
    - Принцип: Показывает изменение значения переменной с течением времени или в зависимости от другой упорядоченной переменной.
    - Python (matplotlib):
      ```python
      import matplotlib.pyplot as plt
      plt.plot(df['time_column'], df['value_column'])
      plt.xlabel('Time')
      plt.ylabel('Value')
      plt.show()
      ```
  - **Тепловая карта (Heatmap)**
    - Принцип: Визуализирует матрицу данных, где значения представлены цветом. Часто используется для отображения корреляционных матриц или матриц расстояний.
    - Python (seaborn):
      ```python
      import seaborn as sns
      corr_matrix = df.corr()
      sns.heatmap(corr_matrix, annot=True)
      plt.show()
      ```
- Интерпретация визуализаций плотности распределения
  - Гистограммы и KDE (Kernel Density Estimation)
    - Принцип: Гистограмма показывает частоту значений, попадающих в определенные интервалы (бины). KDE сглаживает гистограмму, представляя оценку плотности вероятности в виде непрерывной кривой.
    - Python (matplotlib/seaborn):
      ```python
      import matplotlib.pyplot as plt
      import seaborn as sns
      plt.hist(df['column_name']) # Гистограмма
      sns.kdeplot(df['column_name']) # KDE
      plt.show()
      ```
  - Форма распределения (симметричность, скошенность, мультимодальность)
    - Принцип: Анализ формы распределения позволяет понять характеристики данных. Симметричное распределение означает, что данные равномерно распределены относительно центра. Скошенное распределение указывает на асимметрию, а мультимодальное - на наличие нескольких пиков, что может свидетельствовать о наличии нескольких групп данных.
    - Python (scipy/seaborn - визуальная оценка и численные характеристики):
      ```python
      import matplotlib.pyplot as plt
      import seaborn as sns
      sns.histplot(df['column_name'], kde=True) # Гистограмма и KDE для визуального анализа формы
      plt.show()
      from scipy.stats import skew, kurtosis # Численные характеристики
      print(f"Skewness: {skew(df['column_name'])}")
      print(f"Kurtosis: {kurtosis(df['column_name'])}")
      ```
  - Выбросы и аномалии на графиках плотности
    - Принцип: Выбросы - это значения, которые значительно отличаются от основной массы данных. На графиках плотности они могут проявляться как отдельные пики на большом расстоянии от основного распределения или как длинные "хвосты".
    - Python (matplotlib/seaborn - визуальная оценка):
      ```python
      import matplotlib.pyplot as plt
      import seaborn as sns
      sns.boxplot(df['column_name']) # Boxplot для выявления выбросов
      plt.show()
      sns.histplot(df['column_name'], kde=True) # Гистограмма/KDE для визуального поиска выбросов
      plt.show()
      ```
  - Сравнение распределений разных признаков
    - Принцип: Сравнение распределений различных признаков позволяет выявить различия в их статистических свойствах и формах. Это помогает понять, как признаки отличаются друг от друга и какие закономерности могут существовать между ними.
    - Python (matplotlib/seaborn):
      ```python
      import matplotlib.pyplot as plt
      import seaborn as sns
      sns.kdeplot(df['column_name_1'], label='column_name_1')
      sns.kdeplot(df['column_name_2'], label='column_name_2')
      plt.legend()
      plt.show()
      ```
  - Связь формы распределения с природой данных
    - Принцип: Форма распределения часто отражает природу данных. Например, нормальное распределение может возникать в результате действия множества случайных факторов, экспоненциальное распределение может описывать время ожидания событий, а бимодальное распределение может указывать на смесь двух различных групп наблюдений.
- Анализ зависимостей между признаками
  - Принцип: Анализ зависимостей между признаками позволяет выявить взаимосвязи в данных, что помогает лучше понять структуру данных и может быть полезно при построении моделей. Сильные зависимости могут указывать на мультиколлинеарность, которая может негативно влиять на некоторые модели.
  - Корреляционный анализ
    - Принцип: Измерение статистической взаимосвязи между двумя переменными. Коэффициент корреляции Пирсона измеряет линейную зависимость, коэффициент Спирмена - монотонную зависимость.
    - Python (pandas/seaborn):
      ```python
      import pandas as pd
      import seaborn as sns
      import matplotlib.pyplot as plt
      
      # Корреляция Пирсона
      corr_matrix = df.corr(method='pearson')
      print(corr_matrix)
      
      # Корреляция Спирмена (для нелинейных зависимостей)
      corr_spearman = df.corr(method='spearman')
      print(corr_spearman)
      
      # Визуализация корреляционной матрицы
      plt.figure(figsize=(10, 8))
      sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
      plt.title('Корреляционная матрица признаков')
      plt.show()
      ```
  - Диаграммы рассеяния (Scatter plots)
    - Принцип: Визуальное представление взаимосвязи между двумя числовыми переменными. Позволяет обнаружить линейные и нелинейные зависимости, кластеры и выбросы.
    - Python (matplotlib/seaborn):
      ```python
      import matplotlib.pyplot as plt
      import seaborn as sns
      
      # Простая диаграмма рассеяния
      plt.scatter(df['feature1'], df['feature2'])
      plt.xlabel('Feature 1')
      plt.ylabel('Feature 2')
      plt.title('Зависимость между признаками')
      plt.show()
      
      # Матрица диаграмм рассеяния для нескольких признаков
      sns.pairplot(df[['feature1', 'feature2', 'feature3', 'feature4']])
      plt.suptitle('Матрица диаграмм рассеяния', y=1.02)
      plt.show()
      
      # Диаграмма рассеяния с регрессионной линией
      sns.regplot(x='feature1', y='feature2', data=df)
      plt.title('Диаграмма рассеяния с линией регрессии')
      plt.show()
      ```
  - Анализ взаимной информации (Mutual Information)
    - Принцип: Измерение количества информации, которое одна переменная содержит о другой. Позволяет обнаружить нелинейные зависимости, которые могут быть не видны при корреляционном анализе.
    - Python (sklearn):
      ```python
      from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
      
      # Для регрессионных задач
      mi_reg = mutual_info_regression(df[['feature1', 'feature2']], df['target'])
      print(f"Взаимная информация для регрессии: {mi_reg}")
      
      # Для задач классификации
      mi_class = mutual_info_classif(df[['feature1', 'feature2']], df['target_class'])
      print(f"Взаимная информация для классификации: {mi_class}")
      
      # Визуализация
      plt.bar(['feature1', 'feature2'], mi_reg)
      plt.title('Взаимная информация признаков с целевой переменной')
      plt.ylabel('Взаимная информация')
      plt.show()
      ```
  - Анализ условных зависимостей
    - Принцип: Изучение зависимости между переменными при фиксированных значениях других переменных. Позволяет выявить скрытые зависимости и причинно-следственные связи.
    - Python (seaborn):
      ```python
      import seaborn as sns
      import matplotlib.pyplot as plt
      
      # Диаграмма рассеяния с разделением по категориальному признаку
      sns.lmplot(x='feature1', y='feature2', hue='category', data=df)
      plt.title('Условная зависимость признаков по категориям')
      plt.show()
      
      # Факторный график для анализа взаимодействия признаков
      sns.catplot(x='feature1', y='feature2', col='category', kind='box', data=df)
      plt.suptitle('Распределение feature2 в зависимости от feature1 и category', y=1.02)
      plt.show()
      ```
- Визуализация многомерных данных
  - Принцип: Представление данных с большим числом измерений в более понятной и интерпретируемой форме. Позволяет выявить скрытые структуры, кластеры и выбросы в многомерных данных.
  - Python (matplotlib, seaborn, plotly):
    ```python
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D
    import plotly.express as px
    
    # 3D-график рассеяния для трех признаков
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df['feature1'], df['feature2'], df['feature3'], c=df['target'], cmap='viridis')
    ax.set_xlabel('Признак 1')
    ax.set_ylabel('Признак 2')
    ax.set_zlabel('Признак 3')
    plt.title('3D визуализация данных')
    plt.show()
    
    # Тепловая карта корреляций
    plt.figure(figsize=(12, 10))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Тепловая карта корреляций признаков')
    plt.show()
    
    # Параллельные координаты для многомерных данных
    # Каждая вертикальная линия представляет один признак, а каждая ломаная линия - одно наблюдение
    fig = px.parallel_coordinates(
        df, 
        color="target",
        dimensions=['feature1', 'feature2', 'feature3', 'feature4', 'feature5'],
        color_continuous_scale=px.colors.diverging.Tealrose,
        title='Визуализация с помощью параллельных координат'
    )
    fig.show()
    
    # Радарная диаграмма (для сравнения нескольких наблюдений по множеству признаков)
    # Выбираем несколько наблюдений для сравнения
    samples = df.iloc[[0, 10, 20]].copy()
    features = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5']
    
    # Нормализуем данные для радарной диаграммы
    for feature in features:
        samples[feature] = (samples[feature] - df[feature].min()) / (df[feature].max() - df[feature].min())
    
    # Создаем радарную диаграмму
    angles = np.linspace(0, 2*np.pi, len(features), endpoint=False).tolist()
    angles += angles[:1]  # замыкаем круг
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    for i, row in samples.iterrows():
        values = row[features].tolist()
        values += values[:1]  # замыкаем значения
        ax.plot(angles, values, linewidth=2, label=f'Наблюдение {i}')
        ax.fill(angles, values, alpha=0.1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features)
    plt.legend(loc='upper right')
    plt.title('Радарная диаграмма для сравнения наблюдений')
    plt.show()
    ```
- Методы понижения размерности для визуализации:
  - PCA (метод главных компонент):
    - Принцип: линейное преобразование, проецирующее данные в пространство с меньшей размерностью, сохраняя максимум дисперсии
    - Находит направления максимальной вариации данных
    - Используется для визуализации многомерных данных в 2D/3D
    ```python
    from sklearn.decomposition import PCA
    
    # Применение PCA для визуализации в 2D
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # Визуализация результатов
    plt.figure(figsize=(10, 8))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.8)
    plt.colorbar(label='Класс')
    plt.xlabel('Первая главная компонента')
    plt.ylabel('Вторая главная компонента')
    plt.title('PCA: визуализация данных в 2D')
    plt.show()
    
    # Объяснённая дисперсия
    print(f"Объяснённая дисперсия: {pca.explained_variance_ratio_}")
    ```
  
  - t-SNE (t-distributed Stochastic Neighbor Embedding):
    - Принцип: нелинейный метод, сохраняющий локальную структуру данных
    - Хорошо визуализирует кластеры в данных
    - Сохраняет близость точек в исходном пространстве
    ```python
    from sklearn.manifold import TSNE
    
    # Применение t-SNE для визуализации
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X)
    
    # Визуализация результатов
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', alpha=0.8)
    plt.colorbar(scatter, label='Класс')
    plt.xlabel('t-SNE компонента 1')
    plt.ylabel('t-SNE компонента 2')
    plt.title('t-SNE: визуализация данных в 2D')
    plt.show()
    ```
  
  - UMAP (Uniform Manifold Approximation and Projection):
    - Принцип: сохраняет как локальную, так и глобальную структуру данных
    - Быстрее t-SNE и лучше сохраняет глобальную структуру
    - Хорошо работает с большими наборами данных
    ```python
    import umap
    
    # Применение UMAP для визуализации
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    X_umap = reducer.fit_transform(X)
    
    # Визуализация результатов
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='viridis', alpha=0.8)
    plt.colorbar(scatter, label='Класс')
    plt.xlabel('UMAP компонента 1')
    plt.ylabel('UMAP компонента 2')
    plt.title('UMAP: визуализация данных в 2D')
    plt.show()
    ```
  
  - Сравнение методов понижения размерности:
    ```python
    # Визуализация сравнения методов
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # PCA
    axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.8)
    axes[0].set_title('PCA')
    
    # t-SNE
    axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', alpha=0.8)
    axes[1].set_title('t-SNE')
    
    # UMAP
    axes[2].scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='viridis', alpha=0.8)
    axes[2].set_title('UMAP')
    
    plt.tight_layout()
    plt.show()
    ```
- Визуализация результатов работы моделей:
  - Принцип: графическое представление предсказаний модели для оценки её эффективности и понимания ошибок
  - Матрица ошибок (confusion matrix):
    ```python
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    # Построение матрицы ошибок
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Предсказанные значения')
    plt.ylabel('Истинные значения')
    plt.title('Матрица ошибок')
    plt.show()
    ```
  - Кривая обучения (learning curve):
    ```python
    from sklearn.model_selection import learning_curve
    
    # Построение кривой обучения
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, scoring='accuracy', 
        train_sizes=np.linspace(0.1, 1.0, 10))
    
    # Средние значения и стандартные отклонения
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    # Визуализация
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', color='r', label='Обучающая выборка')
    plt.plot(train_sizes, test_mean, 'o-', color='g', label='Тестовая выборка')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='g')
    plt.xlabel('Размер обучающей выборки')
    plt.ylabel('Точность')
    plt.title('Кривая обучения')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()
    ```
  - Визуализация предсказаний регрессии:
    ```python
    # Визуализация предсказаний регрессионной модели
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.7)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2)
    plt.xlabel('Истинные значения')
    plt.ylabel('Предсказанные значения')
    plt.title('Сравнение истинных и предсказанных значений')
    plt.grid(True)
    plt.show()
    
    # Визуализация остатков
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.xlabel('Предсказанные значения')
    plt.ylabel('Остатки')
    plt.title('График остатков')
    plt.grid(True)
    plt.show()
    ```
  - Визуализация вероятностей классификации:
    ```python
    # Получение вероятностей классов
    y_proba = model.predict_proba(X_test)
    
    # Визуализация вероятностей для первых N примеров
    n_samples = 10
    plt.figure(figsize=(12, 6))
    for i in range(n_samples):
        plt.bar(np.arange(len(class_names)), y_proba[i], alpha=0.7)
        plt.xticks(np.arange(len(class_names)), class_names, rotation=45)
        plt.xlabel('Классы')
        plt.ylabel('Вероятность')
        plt.title(f'Распределение вероятностей для примера {i+1} (истинный класс: {class_names[y_test[i]]})')
        plt.grid(True)
        plt.show()
    ```
- Визуализация важности признаков:
  - Принцип: отображение относительной важности каждого признака в модели, помогает понять, какие признаки наиболее влияют на предсказания
  - Для деревьев и ансамблей (feature_importances_):
    ```python
    # Визуализация важности признаков для моделей на основе деревьев
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(X.shape[1]), importances[indices], align='center')
    plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=90)
    plt.xlabel('Признаки')
    plt.ylabel('Важность')
    plt.title('Важность признаков')
    plt.tight_layout()
    plt.show()
    ```
  - Для линейных моделей (коэффициенты):
    ```python
    # Визуализация коэффициентов линейной модели
    coefs = pd.Series(model.coef_[0], index=feature_names)
    coefs_abs = coefs.abs().sort_values(ascending=False)
    
    plt.figure(figsize=(10, 6))
    coefs_abs.head(20).plot(kind='bar')
    plt.xlabel('Признаки')
    plt.ylabel('Абсолютное значение коэффициента')
    plt.title('Топ-20 важных признаков')
    plt.tight_layout()
    plt.show()
    ```
  - Permutation importance (для любых моделей):
    ```python
    from sklearn.inspection import permutation_importance
    
    # Вычисление важности признаков методом перестановок
    result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
    
    # Сортировка признаков по важности
    sorted_idx = result.importances_mean.argsort()[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.boxplot(result.importances[sorted_idx].T, vert=False, labels=[feature_names[i] for i in sorted_idx])
    plt.xlabel('Уменьшение точности')
    plt.title('Важность признаков (метод перестановок)')
    plt.tight_layout()
    plt.show()
    ```
  - SHAP values (для объяснения предсказаний):
    ```python
    import shap
    
    # Создание объяснителя
    explainer = shap.TreeExplainer(model)
    # Вычисление SHAP значений
    shap_values = explainer.shap_values(X_test)
    
    # Сводная диаграмма важности признаков
    shap.summary_plot(shap_values, X_test, feature_names=feature_names)
    
    # Зависимость SHAP значений от значений признака
    shap.dependence_plot("feature_index", shap_values, X_test, feature_names=feature_names)
    ```
- Визуализация деревьев решений
  - Принцип: графическое представление структуры дерева решений, позволяющее понять логику принятия решений моделью. Визуализация показывает узлы дерева, условия разделения, значения в листьях и другие характеристики.
  - Базовая визуализация с использованием scikit-learn:
    ```python
    from sklearn import tree
    import matplotlib.pyplot as plt
    
    # Обучение дерева решений
    clf = tree.DecisionTreeClassifier(max_depth=3)
    clf = clf.fit(X_train, y_train)
    
    # Визуализация дерева
    plt.figure(figsize=(15, 10))
    tree.plot_tree(clf, feature_names=feature_names, class_names=class_names, filled=True)
    plt.show()
    ```
  - Экспорт дерева в формате DOT и визуализация с помощью Graphviz:
    ```python
    import graphviz
    
    # Экспорт дерева в формате DOT
    dot_data = tree.export_graphviz(clf, out_file=None, 
                                    feature_names=feature_names,  
                                    class_names=class_names,
                                    filled=True, rounded=True,  
                                    special_characters=True)
    
    # Создание графика с помощью Graphviz
    graph = graphviz.Source(dot_data)
    graph.render("decision_tree", format="png")  # Сохранение в файл
    graph  # Отображение в Jupyter Notebook
    ```
  - Визуализация с помощью библиотеки dtreeviz для более детального анализа:
    ```python
    from dtreeviz.trees import dtreeviz
    
    # Создание визуализации
    viz = dtreeviz(clf, X_train, y_train,
                  target_name="target",
                  feature_names=feature_names,
                  class_names=list(class_names))
    
    # Отображение визуализации
    viz.view()
    
    # Сохранение в файл
    viz.save("decision_tree_detailed.svg")
    ```
- Визуализация процесса обучения
  - Принцип: отслеживание и визуализация изменения метрик и параметров модели в процессе обучения для анализа сходимости, выявления переобучения и оценки эффективности обучения.
  - Визуализация кривых обучения (learning curves):
    ```python
    import matplotlib.pyplot as plt
    from sklearn.model_selection import learning_curve
    
    # Построение кривых обучения
    train_sizes, train_scores, test_scores = learning_curve(
        estimator=model, X=X, y=y, cv=5, scoring='accuracy',
        train_sizes=np.linspace(0.1, 1.0, 10))
    
    # Расчет средних значений и стандартных отклонений
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    # Визуализация
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, color='blue', marker='o', 
             label='Точность на обучающей выборке')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                     alpha=0.15, color='blue')
    plt.plot(train_sizes, test_mean, color='green', marker='s', 
             label='Точность на тестовой выборке')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, 
                     alpha=0.15, color='green')
    plt.title('Кривые обучения')
    plt.xlabel('Размер обучающей выборки')
    plt.ylabel('Точность')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()
    ```
  - Визуализация истории обучения с TensorFlow/Keras:
    ```python
    import matplotlib.pyplot as plt
    
    # Обучение модели с сохранением истории
    history = model.fit(X_train, y_train, 
                        validation_data=(X_val, y_val),
                        epochs=100, batch_size=32, verbose=1)
    
    # Визуализация истории обучения
    plt.figure(figsize=(12, 5))
    
    # График функции потерь
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Обучающая выборка')
    plt.plot(history.history['val_loss'], label='Валидационная выборка')
    plt.title('Функция потерь')
    plt.xlabel('Эпоха')
    plt.ylabel('Значение функции потерь')
    plt.legend()
    
    # График точности
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Обучающая выборка')
    plt.plot(history.history['val_accuracy'], label='Валидационная выборка')
    plt.title('Точность модели')
    plt.xlabel('Эпоха')
    plt.ylabel('Точность')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    ```
  - Визуализация процесса обучения в реальном времени с использованием callbacks:
    ```python
    from tensorflow.keras.callbacks import TensorBoard
    import datetime
    
    # Создание директории для логов
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Создание callback для TensorBoard
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    # Обучение модели с использованием callback
    model.fit(X_train, y_train, 
              validation_data=(X_val, y_val),
              epochs=100, batch_size=32,
              callbacks=[tensorboard_callback])
    
    # Запуск TensorBoard в командной строке:
    # tensorboard --logdir=logs/fit
    ```

## 6. Линейная регрессия
- Интерпретация коэффициентов модели:
  - Для простой линейной регрессии: $y = \beta_0 + \beta_1 x$
    - $\beta_0$ - свободный член (intercept), значение $y$ при $x = 0$
    - $\beta_1$ - наклон прямой, показывает изменение $y$ при увеличении $x$ на единицу
  - Для множественной линейной регрессии: $y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_n x_n$
    - $\beta_i$ - показывает изменение $y$ при увеличении $x_i$ на единицу при фиксированных остальных признаках
  - Стандартизованные коэффициенты:
    - Позволяют сравнивать влияние признаков с разными единицами измерения
    - Вычисляются на стандартизованных данных или по формуле: $\beta_i^* = \beta_i \cdot \frac{\sigma_{x_i}}{\sigma_y}$
  - Ограничения интерпретации:
    - Корректна только для линейных зависимостей
    - Затруднена при мультиколлинеарности признаков
    - Не отражает причинно-следственные связи
  - Доверительные интервалы для коэффициентов:
    - Позволяют оценить неопределенность коэффициентов
    - Широкие интервалы указывают на низкую статистическую значимость
- Масштабирование признаков
  - Необходимость масштабирования:
    - Многие алгоритмы машинного обучения чувствительны к масштабу признаков
    - Признаки с большими значениями могут доминировать в функции потерь
    - Ускоряет сходимость градиентных методов оптимизации
  - Методы масштабирования:
    - Стандартизация (Z-нормализация):
      - Преобразует признаки к распределению со средним 0 и стандартным отклонением 1
      - Формула: $x_{scaled} = \frac{x - \mu}{\sigma}$
      - Python (scikit-learn):
        ```python
        from sklearn.preprocessing import StandardScaler
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        ```
    - Минимакс-нормализация:
      - Масштабирует признаки в диапазон [0, 1] или [-1, 1]
      - Формула: $x_{scaled} = \frac{x - x_{min}}{x_{max} - x_{min}}$
      - Python (scikit-learn):
        ```python
        from sklearn.preprocessing import MinMaxScaler
        
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_scaled = scaler.fit_transform(X)
        ```
    - Робастное масштабирование:
      - Использует медиану и межквартильный размах вместо среднего и стандартного отклонения
      - Устойчиво к выбросам
      - Формула: $x_{scaled} = \frac{x - median(x)}{IQR(x)}$
      - Python (scikit-learn):
        ```python
        from sklearn.preprocessing import RobustScaler
        
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        ```
  - Важные практические аспекты:
    - Применять одинаковое масштабирование к обучающей и тестовой выборкам
    - Обучать scaler только на обучающих данных
    - Сохранять параметры масштабирования для последующего применения к новым данным
- Функции потерь (MSE, MAE):
  - Mean Squared Error (MSE):
    - Формула: $MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$
    - Квадратичная функция потерь, сильно штрафует за большие ошибки
    - Дифференцируема, что удобно для градиентных методов оптимизации
    - Чувствительна к выбросам из-за возведения в квадрат
    - Python:
      ```python
      from sklearn.metrics import mean_squared_error
      
      mse = mean_squared_error(y_true, y_pred)
      ```
  - Mean Absolute Error (MAE):
    - Формула: $MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$
    - Линейная функция потерь, менее чувствительна к выбросам
    - Не дифференцируема в нуле, что может создавать сложности при оптимизации
    - Медиана минимизирует MAE (в отличие от среднего для MSE)
    - Python:
      ```python
      from sklearn.metrics import mean_absolute_error
      
      mae = mean_absolute_error(y_true, y_pred)
      ```
  - Huber Loss:
    - Комбинирует преимущества MSE и MAE
    - Менее чувствительна к выбросам, чем MSE, но сохраняет дифференцируемость
    - Формула: $L_\delta(y, \hat{y}) = \begin{cases}
      \frac{1}{2}(y - \hat{y})^2, & \text{если } |y - \hat{y}| \leq \delta \\
      \delta|y - \hat{y}| - \frac{1}{2}\delta^2, & \text{иначе}
      \end{cases}$
    - Параметр $\delta$ определяет порог перехода от квадратичной к линейной функции
  - Root Mean Squared Error (RMSE):
    - Формула: $RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$
    - Корень из MSE, имеет те же единицы измерения, что и целевая переменная
    - Часто используется для интерпретации результатов
- Полиномиальные признаки
  - Принцип: создание новых признаков путем возведения исходных признаков в степень или их перемножения. Позволяет моделировать нелинейные зависимости с помощью линейных моделей.
  - Формула: для признаков $x_1, x_2, ..., x_n$ и степени $d$ создаются все возможные комбинации вида $x_1^{a_1} \cdot x_2^{a_2} \cdot ... \cdot x_n^{a_n}$, где $a_1 + a_2 + ... + a_n \leq d$
  - Примеры: для $d=2$ и признаков $x_1, x_2$ получаем $x_1, x_2, x_1^2, x_1x_2, x_2^2$
  - Python (scikit-learn):
    ```python
    from sklearn.preprocessing import PolynomialFeatures
    
    # Создание полиномиальных признаков степени 2
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)
    
    # Получение имен новых признаков
    feature_names = poly.get_feature_names_out(input_features=X.columns)
    ```
  - Преимущества:
    - Позволяет моделировать нелинейные зависимости с помощью линейных алгоритмов
    - Интерпретируемость результатов (в отличие от более сложных нелинейных моделей)
  - Недостатки:
    - Быстрый рост числа признаков с увеличением степени полинома и количества исходных признаков
    - Склонность к переобучению при высоких степенях полинома
    - Чувствительность к масштабу признаков (рекомендуется предварительное масштабирование)
- Факторизационные машины (Factorization Machines, FM):
  - Принцип: расширение линейных моделей для моделирования взаимодействий между признаками через факторизацию
  - Формула: $\hat{y}(x) = w_0 + \sum_{i=1}^{n} w_i x_i + \sum_{i=1}^{n} \sum_{j=i+1}^{n} \langle v_i, v_j \rangle x_i x_j$
    - $w_0$ - глобальное смещение
    - $w_i$ - вес для i-го признака
    - $v_i$ - вектор факторов для i-го признака
    - $\langle v_i, v_j \rangle$ - скалярное произведение векторов факторов
  - Преимущества:
    - Эффективное моделирование взаимодействий между признаками даже при разреженных данных
    - Работает хорошо с категориальными признаками после one-hot кодирования
    - Вычислительная сложность линейна относительно числа признаков и факторов
    - Обобщает матричную факторизацию для рекомендательных систем
  - Применения:
    - Рекомендательные системы
    - Задачи с разреженными данными высокой размерности
    - CTR-предсказание (Click-Through Rate) в онлайн-рекламе
  - Реализации:
    - Python (fastFM, pyFM):
      ```python
      from fastFM.mcmc import FMClassification
      
      fm = FMClassification(n_iter=100, rank=2)
      fm.fit(X_train, y_train)
      y_pred = fm.predict(X_test)
      ```
  - Расширения:
    - Field-aware Factorization Machines (FFM)
    - Higher-Order Factorization Machines
    - Deep Factorization Machines (DeepFM)
- Регуляризация
  - Принцип: метод предотвращения переобучения модели путем добавления штрафа за сложность модели
  - Основные типы регуляризации:
    - L1-регуляризация (Lasso):
      - Добавляет штраф, пропорциональный сумме абсолютных значений весов: $\lambda \sum_{i=1}^{n} |w_i|$
      - Способствует разреженности модели (обнуляет некоторые веса)
      - Выполняет неявный отбор признаков
      - Формула для линейной регрессии с L1: $J(w) = MSE(w) + \lambda \sum_{i=1}^{n} |w_i|$
    - L2-регуляризация (Ridge):
      - Добавляет штраф, пропорциональный сумме квадратов весов: $\lambda \sum_{i=1}^{n} w_i^2$
      - Стремится уменьшить все веса, но не обнуляет их
      - Особенно полезна при мультиколлинеарности признаков
      - Формула для линейной регрессии с L2: $J(w) = MSE(w) + \lambda \sum_{i=1}^{n} w_i^2$
    - Elastic Net:
      - Комбинирует L1 и L2 регуляризацию: $\lambda_1 \sum_{i=1}^{n} |w_i| + \lambda_2 \sum_{i=1}^{n} w_i^2$
      - Сочетает преимущества обоих подходов
      - Формула: $J(w) = MSE(w) + \lambda_1 \sum_{i=1}^{n} |w_i| + \lambda_2 \sum_{i=1}^{n} w_i^2$
  - Реализация в Python:
    ```python
    # L1-регуляризация (Lasso)
    from sklearn.linear_model import Lasso
    lasso = Lasso(alpha=0.1)  # alpha - коэффициент регуляризации
    lasso.fit(X_train, y_train)
    
    # L2-регуляризация (Ridge)
    from sklearn.linear_model import Ridge
    ridge = Ridge(alpha=0.1)
    ridge.fit(X_train, y_train)
    
    # Elastic Net
    from sklearn.linear_model import ElasticNet
    elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)  # l1_ratio - соотношение L1 и L2
    elastic.fit(X_train, y_train)
    ```
  - Выбор параметра регуляризации:
    - Обычно определяется с помощью кросс-валидации
    - Слишком маленькое значение может привести к переобучению
    - Слишком большое значение может привести к недообучению
  - Преимущества регуляризации:
    - Снижение переобучения
    - Улучшение обобщающей способности модели
    - Стабилизация решения при мультиколлинеарности
    - Возможность неявного отбора признаков (для L1)
- Градиентный спуск
  - Принцип: итеративный алгоритм оптимизации, который находит минимум функции потерь путем движения в направлении, противоположном градиенту
  - Формула обновления весов: $w_{t+1} = w_t - \eta \nabla J(w_t)$, где $\eta$ - скорость обучения (learning rate)
  - Разновидности:
    - Пакетный градиентный спуск (Batch Gradient Descent):
      - Использует все обучающие примеры для вычисления градиента
      - Медленный, но стабильный
      - Гарантированно сходится к глобальному минимуму для выпуклых функций
    - Стохастический градиентный спуск (SGD):
      - Использует один случайный пример для вычисления градиента на каждой итерации
      - Быстрее, но менее стабильный
      - Может "перепрыгнуть" локальные минимумы
      - Формула: $w_{t+1} = w_t - \eta \nabla J_i(w_t)$, где $J_i$ - функция потерь для i-го примера
    - Мини-пакетный градиентный спуск (Mini-batch Gradient Descent):
      - Компромисс между пакетным и стохастическим подходами
      - Использует небольшую случайную выборку примеров (мини-пакет)
      - Обычно размер мини-пакета от 32 до 256 примеров
  - Проблемы и их решения:
    - Выбор скорости обучения:
      - Слишком большая - расходимость
      - Слишком маленькая - медленная сходимость
      - Решения: графики обучения, адаптивные методы
    - Застревание в локальных минимумах:
      - Решения: случайная инициализация, стохастический градиентный спуск
    - Плато и седловые точки:
      - Решения: моментум, адаптивные методы
  - Реализация в Python:
    ```python
    # Пример простой реализации градиентного спуска для линейной регрессии
    def gradient_descent(X, y, learning_rate=0.01, n_iterations=1000):
        m = len(y)
        theta = np.zeros(X.shape[1])
        cost_history = []
        
        for i in range(n_iterations):
            # Вычисление предсказаний
            predictions = X.dot(theta)
            
            # Вычисление ошибки
            errors = predictions - y
            
            # Вычисление градиента
            gradient = (1/m) * X.T.dot(errors)
            
            # Обновление весов
            theta = theta - learning_rate * gradient
            
            # Вычисление функции потерь
            cost = (1/(2*m)) * np.sum(errors**2)
            cost_history.append(cost)
            
        return theta, cost_history
    
    # Использование с scikit-learn
    from sklearn.linear_model import SGDRegressor
    
    sgd_reg = SGDRegressor(max_iter=1000, tol=1e-3, penalty=None, eta0=0.1)
    sgd_reg.fit(X_train, y_train)
    ```
  - Продвинутые методы оптимизации:
    - Моментум: добавляет "инерцию" к движению, помогая преодолевать плато и локальные минимумы
      - $v_t = \gamma v_{t-1} + \eta \nabla J(w_t)$
      - $w_{t+1} = w_t - v_t$
    - Nesterov Accelerated Gradient (NAG): улучшенная версия моментума
    - AdaGrad: адаптивно настраивает скорость обучения для каждого параметра
    - RMSProp: решает проблему уменьшающейся скорости обучения в AdaGrad
    - Adam: комбинирует преимущества моментума и RMSProp
- Нормальные уравнения
  - Принцип: аналитическое решение задачи линейной регрессии путем приравнивания градиента функции потерь к нулю
  - Формула: $\mathbf{w} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$
  - Преимущества:
    - Точное решение за один шаг (без итераций)
    - Не требует настройки гиперпараметров (скорости обучения)
  - Недостатки:
    - Вычислительно затратно для больших наборов данных (сложность $O(n^3)$)
    - Проблемы с обращением матрицы при мультиколлинеарности
    - Неустойчивость при плохо обусловленных матрицах
  - Реализация в Python:
    ```python
    # Решение с помощью нормальных уравнений
    def normal_equation(X, y):
        return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    
    # Альтернативная реализация с использованием numpy.linalg.solve
    def normal_equation_solve(X, y):
        return np.linalg.solve(X.T.dot(X), X.T.dot(y))
    
    # Использование с scikit-learn
    from sklearn.linear_model import LinearRegression
    
    lr = LinearRegression(fit_intercept=True)
    lr.fit(X_train, y_train)
    # sklearn использует более стабильные методы, чем прямое обращение матрицы
    ```
  - Регуляризованные версии:
    - Ridge (L2): $\mathbf{w} = (\mathbf{X}^T\mathbf{X} + \lambda \mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}$
    - Решает проблему мультиколлинеарности и улучшает обусловленность
- Вероятностный подход к линейной регрессии
  - Принцип: рассматривает линейную регрессию как вероятностную модель, где целевая переменная имеет нормальное распределение с математическим ожиданием, зависящим от входных данных
  - Модель: $y = \mathbf{w}^T\mathbf{x} + \varepsilon$, где $\varepsilon \sim \mathcal{N}(0, \sigma^2)$
  - Вероятностная интерпретация: $p(y|\mathbf{x}, \mathbf{w}, \sigma^2) = \mathcal{N}(y|\mathbf{w}^T\mathbf{x}, \sigma^2)$
  - Преимущества:
    - Позволяет оценивать неопределенность предсказаний
    - Обеспечивает теоретическое обоснование для методов регуляризации
    - Дает возможность использовать байесовские методы для оценки параметров
  - Связь с методом наименьших квадратов: максимизация правдоподобия при нормальном распределении ошибок эквивалентна минимизации суммы квадратов ошибок
  - Реализация в Python:
    ```python
    import pymc3 as pm
    import numpy as np
    
    # Создание вероятностной модели
    with pm.Model() as linear_model:
        # Приоры для параметров
        weights = pm.Normal('weights', mu=0, sigma=1, shape=X.shape[1])
        sigma = pm.HalfNormal('sigma', sigma=1)
        
        # Ожидаемое значение
        mu = pm.math.dot(X, weights)
        
        # Правдоподобие (наблюдаемые данные)
        y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)
        
        # Выборка из апостериорного распределения
        trace = pm.sample(1000, tune=1000)
    
    # Анализ результатов
    pm.plot_trace(trace)
    pm.summary(trace)
    ```
- Байесовский подход
  - Принцип: использует теорему Байеса для обновления вероятностных оценок параметров модели с учетом наблюдаемых данных
  - Формулировка: $p(\mathbf{w}|X, \mathbf{y}) \propto p(\mathbf{y}|X, \mathbf{w}) \cdot p(\mathbf{w})$
  - Компоненты:
    - Априорное распределение $p(\mathbf{w})$ - начальные предположения о параметрах
    - Функция правдоподобия $p(\mathbf{y}|X, \mathbf{w})$ - как данные зависят от параметров
    - Апостериорное распределение $p(\mathbf{w}|X, \mathbf{y})$ - обновленные знания о параметрах
  - Преимущества:
    - Учет неопределенности в оценках параметров
    - Естественная регуляризация через априорные распределения
    - Возможность инкрементального обучения
    - Оценка интервалов доверия для предсказаний
  - Байесовская линейная регрессия:
    - Априорное распределение: $\mathbf{w} \sim \mathcal{N}(0, \alpha^{-1}I)$
    - Апостериорное распределение: $p(\mathbf{w}|X, \mathbf{y}) \sim \mathcal{N}(\mathbf{m}_N, \mathbf{S}_N)$
    - $\mathbf{S}_N = (\alpha I + \beta X^TX)^{-1}$
    - $\mathbf{m}_N = \beta \mathbf{S}_N X^T \mathbf{y}$
  - Реализация с помощью PyMC3:
    ```python
    import pymc3 as pm
    
    with pm.Model() as model:
        # Априорные распределения
        alpha = pm.HalfCauchy('alpha', beta=5)
        weights = pm.Normal('weights', mu=0, sigma=1/alpha, shape=X.shape[1])
        sigma = pm.HalfCauchy('sigma', beta=5)
        
        # Функция правдоподобия
        mu = pm.math.dot(X, weights)
        y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)
        
        # Выборка из апостериорного распределения
        trace = pm.sample(2000, tune=1000)
    
    # Предсказания с учетом неопределенности
    with model:
        ppc = pm.sample_posterior_predictive(trace, samples=500)
    ```
- Метод максимального правдоподобия
  - Принцип: оценка параметров модели путем максимизации функции правдоподобия - вероятности наблюдения данных при заданных параметрах
  - Формулировка: $\hat{\theta} = \arg\max_{\theta} L(\theta|X) = \arg\max_{\theta} \prod_{i=1}^n p(x_i|\theta)$
  - Логарифмическая функция правдоподобия: $\ell(\theta|X) = \log L(\theta|X) = \sum_{i=1}^n \log p(x_i|\theta)$
  - Преимущества:
    - Состоятельность оценок (сходимость к истинным значениям при увеличении выборки)
    - Асимптотическая эффективность (минимальная дисперсия оценок при больших выборках)
    - Инвариантность относительно параметризации
  - Применение в линейной регрессии:
    - Предположение о нормальном распределении ошибок: $\varepsilon_i \sim \mathcal{N}(0, \sigma^2)$
    - Функция правдоподобия: $L(\mathbf{w}, \sigma^2|X, \mathbf{y}) = \prod_{i=1}^n \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y_i - \mathbf{w}^T\mathbf{x}_i)^2}{2\sigma^2}\right)$
    - Логарифмическая функция правдоподобия: $\ell(\mathbf{w}, \sigma^2|X, \mathbf{y}) = -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^n (y_i - \mathbf{w}^T\mathbf{x}_i)^2$
    - Максимизация по $\mathbf{w}$ эквивалентна минимизации суммы квадратов ошибок
  - Реализация в Python:
    ```python
    from scipy.optimize import minimize
    
    # Определение отрицательной логарифмической функции правдоподобия
    def negative_log_likelihood(params, X, y):
        w = params[:-1]  # Коэффициенты модели
        sigma = params[-1]  # Стандартное отклонение ошибки
        
        # Предсказания модели
        y_pred = np.dot(X, w)
        
        # Вычисление отрицательного логарифмического правдоподобия
        nll = 0.5 * len(y) * np.log(2 * np.pi * sigma**2)
        nll += 0.5 * np.sum((y - y_pred)**2) / (sigma**2)
        
        return nll
    
    # Начальные значения параметров
    initial_params = np.zeros(X.shape[1] + 1)
    initial_params[-1] = 1.0  # Начальное значение для sigma
    
    # Оптимизация
    result = minimize(negative_log_likelihood, initial_params, args=(X, y))
    
    # Извлечение оптимальных параметров
    optimal_w = result.x[:-1]
    optimal_sigma = result.x[-1]
    ```
- Связь МНК и метода максимального правдоподобия
  - Принцип: при предположении о нормальном распределении ошибок с нулевым средним и постоянной дисперсией, метод наименьших квадратов (МНК) эквивалентен методу максимального правдоподобия (ММП)
  - Математическое обоснование:
    - В линейной регрессии с моделью $y = X\beta + \varepsilon$, где $\varepsilon \sim \mathcal{N}(0, \sigma^2I)$
    - Функция правдоподобия: $L(\beta, \sigma^2|X, y) = \prod_{i=1}^n \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y_i - x_i^T\beta)^2}{2\sigma^2}\right)$
    - Логарифмическая функция правдоподобия: $\ell(\beta, \sigma^2|X, y) = -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^n (y_i - x_i^T\beta)^2$
    - Максимизация $\ell$ по $\beta$ эквивалентна минимизации суммы квадратов ошибок $\sum_{i=1}^n (y_i - x_i^T\beta)^2$, что и является целью МНК
  - Следствия:
    - Оценки МНК являются оценками максимального правдоподобия при нормальном распределении ошибок
    - Оценки МНК наследуют свойства оценок максимального правдоподобия: состоятельность, асимптотическую нормальность и эффективность
    - При нарушении предположения о нормальности ошибок, МНК и ММП могут давать разные результаты
  - Практическое значение:
    - Обоснование использования МНК в статистическом анализе
    - Возможность построения доверительных интервалов и проведения статистических тестов для коэффициентов регрессии
- Методы оптимизации для линейной регрессии
  - Принцип: алгоритмы для нахождения оптимальных параметров модели путем минимизации функции потерь
  - Аналитическое решение (Normal Equation):
    - Формула: $\hat{\beta} = (X^TX)^{-1}X^Ty$
    - Python:
      ```python
      import numpy as np
      
      # Аналитическое решение
      beta = np.linalg.inv(X.T @ X) @ X.T @ y
      ```
    - Преимущества: точное решение за один шаг
    - Недостатки: вычислительно затратно для больших наборов данных (O(n³))
  - Градиентный спуск (Gradient Descent):
    - Принцип: итеративное обновление параметров в направлении антиградиента функции потерь
    - Формула обновления: $\beta^{(t+1)} = \beta^{(t)} - \alpha \nabla J(\beta^{(t)})$
    - Python:
      ```python
      # Градиентный спуск для линейной регрессии
      def gradient_descent(X, y, learning_rate=0.01, n_iterations=1000):
          m, n = X.shape
          beta = np.zeros(n)
          
          for i in range(n_iterations):
              predictions = X @ beta
              errors = predictions - y
              gradient = (1/m) * X.T @ errors
              beta = beta - learning_rate * gradient
              
          return beta
      ```
    - Разновидности:
      - Пакетный (Batch) градиентный спуск: использует все данные для каждого обновления
      - Стохастический (Stochastic) градиентный спуск: использует один случайный пример для каждого обновления
      - Мини-пакетный (Mini-batch) градиентный спуск: использует подмножество данных для каждого обновления
  - Метод наименьших квадратов с регуляризацией:
    - Ridge-регрессия (L2-регуляризация): $J(\beta) = \|y - X\beta\|^2 + \lambda\|\beta\|^2$
    - Lasso-регрессия (L1-регуляризация): $J(\beta) = \|y - X\beta\|^2 + \lambda\|\beta\|_1$
    - ElasticNet (комбинация L1 и L2): $J(\beta) = \|y - X\beta\|^2 + \lambda_1\|\beta\|_1 + \lambda_2\|\beta\|^2$
  - Метод сопряженных градиентов (Conjugate Gradient Method):
    - Принцип: итеративный метод для решения систем линейных уравнений, более эффективный чем обычный градиентный спуск
    - Применение: решение нормального уравнения $X^TX\beta = X^Ty$ без явного вычисления $(X^TX)^{-1}$
  - Реализация в scikit-learn:
    ```python
    from sklearn.linear_model import LinearRegression, SGDRegressor
    
    # Аналитическое решение
    model = LinearRegression()
    model.fit(X, y)
    
    # Градиентный спуск
    sgd_model = SGDRegressor(loss='squared_error', max_iter=1000, tol=1e-3)
    sgd_model.fit(X, y)
    ```

## 7. Линейная классификация
- Логистическая регрессия:
  - Принцип: моделирование вероятности принадлежности к классу с помощью логистической функции
  - Математическая формула: $P(y=1|x) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + ... + \beta_n x_n)}} = \sigma(\beta^T x)$
  - Логистическая (сигмоидная) функция: $\sigma(z) = \frac{1}{1 + e^{-z}}$
  - Функция потерь: $J(\beta) = -\frac{1}{m} \sum_{i=1}^{m} [y_i \log(h_\beta(x_i)) + (1-y_i) \log(1-h_\beta(x_i))]$
  - Обучение: максимизация функции правдоподобия или минимизация функции потерь с помощью градиентного спуска
  - Реализация в scikit-learn:
    ```python
    from sklearn.linear_model import LogisticRegression
    
    model = LogisticRegression(C=1.0, penalty='l2', solver='lbfgs')
    model.fit(X_train, y_train)
    
    # Получение вероятностей
    probabilities = model.predict_proba(X_test)
    
    # Получение предсказаний
    predictions = model.predict(X_test)
    ```
  - Регуляризация: L1 (Lasso) и L2 (Ridge) для предотвращения переобучения
  - Интерпретация коэффициентов: $\beta_i$ показывает изменение логарифма шансов при увеличении $x_i$ на единицу
  - Преимущества:
    - Вероятностная интерпретация результатов
    - Эффективность для линейно разделимых данных
    - Низкая вычислительная сложность
    - Хорошая интерпретируемость
  - Недостатки:
    - Предположение о линейной разделимости классов
    - Чувствительность к мультиколлинеарности
    - Ограниченная выразительность для сложных зависимостей
- Метод максимального правдоподобия для классификации
  - Принцип: оценка параметров модели путем максимизации вероятности наблюдения обучающих данных
  - Математическая формула: $L(\theta) = P(D|\theta) = \prod_{i=1}^{n} P(y_i|x_i, \theta)$
  - Логарифмическая функция правдоподобия: $\log L(\theta) = \sum_{i=1}^{n} \log P(y_i|x_i, \theta)$
  - Связь с функцией потерь: минимизация отрицательного логарифма правдоподобия эквивалентна минимизации кросс-энтропийной функции потерь
  - Преимущества:
    - Статистически обоснованный подход
    - Асимптотическая эффективность оценок
    - Инвариантность к параметризации
  - Недостатки:
    - Чувствительность к выбросам
    - Может приводить к переобучению при малых выборках
- Функции потерь в задачах классификации:
  - Бинарная кросс-энтропия (Binary Cross-Entropy):
    - Используется для бинарной классификации
    - Формула: $L = -\frac{1}{N} \sum_{i=1}^{N} [y_i \log(p_i) + (1-y_i) \log(1-p_i)]$
    - Где $y_i$ - истинная метка класса, $p_i$ - предсказанная вероятность
    - Реализация в Python:
      ```python
      from sklearn.metrics import log_loss
      
      loss = log_loss(y_true, y_pred)
      ```
  - Категориальная кросс-энтропия (Categorical Cross-Entropy):
    - Используется для многоклассовой классификации
    - Формула: $L = -\frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{C} y_{ij} \log(p_{ij})$
    - Где $y_{ij}$ - индикатор принадлежности i-го примера к классу j, $p_{ij}$ - предсказанная вероятность
    - Требует one-hot кодирования меток классов
  - Функция потерь Хинджа (Hinge Loss):
    - Используется в SVM для максимизации отступа
    - Формула: $L = \frac{1}{N} \sum_{i=1}^{N} \max(0, 1 - y_i \cdot f(x_i))$
    - Где $y_i \in \{-1, 1\}$ - метка класса, $f(x_i)$ - предсказанное значение
  - Функция потерь Хубера (Huber Loss):
    - Комбинирует MSE и MAE, менее чувствительна к выбросам
    - Формула: 
    $$L_\delta(y, f(x)) = \begin{cases}
      \frac{1}{2}(y - f(x))^2, & \text{если } |y - f(x)| \leq \delta \\
      \delta|y - f(x)| - \frac{1}{2}\delta^2, & \text{иначе}
      \end{cases}$$
  - Focal Loss:
    - Модификация кросс-энтропии для несбалансированных данных
    - Формула: $L = -\frac{1}{N} \sum_{i=1}^{N} (1-p_i)^\gamma \log(p_i)$ для положительных примеров
    - Параметр $\gamma$ уменьшает вклад легко классифицируемых примеров
- Многоклассовая классификация
  - Принцип: расширение методов бинарной классификации для задач с более чем двумя классами
  - Основные подходы:
    - One-vs-All (One-vs-Rest): обучение отдельного классификатора для каждого класса против всех остальных
    - One-vs-One: обучение отдельного классификатора для каждой пары классов
    - Прямые многоклассовые методы: модели, изначально поддерживающие многоклассовую классификацию (например, softmax-регрессия)
  - Softmax-регрессия (многоклассовая логистическая регрессия):
    - Обобщение логистической регрессии на случай нескольких классов
    - Функция softmax: $P(y=j|x) = \frac{e^{w_j^T x}}{\sum_{k=1}^{K} e^{w_k^T x}}$
    - Функция потерь: категориальная кросс-энтропия
  - Метрики качества для многоклассовой классификации:
    - Accuracy: доля правильно классифицированных примеров
      ```python
      from sklearn.metrics import accuracy_score
      
      accuracy = accuracy_score(y_true, y_pred)
      ```
    - Macro/Micro/Weighted F1-score: различные способы усреднения F1-меры по классам
      ```python
      from sklearn.metrics import f1_score
      
      # Macro F1 - среднее F1 по всем классам с равным весом
      macro_f1 = f1_score(y_true, y_pred, average='macro')
      
      # Micro F1 - вычисляет метрики глобально по всем классам
      micro_f1 = f1_score(y_true, y_pred, average='micro')
      
      # Weighted F1 - учитывает дисбаланс классов
      weighted_f1 = f1_score(y_true, y_pred, average='weighted')
      ```
    - Confusion matrix: матрица ошибок, показывающая распределение предсказаний по классам
      ```python
      import numpy as np
      import matplotlib.pyplot as plt
      from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
      
      # Построение матрицы ошибок
      cm = confusion_matrix(y_true, y_pred)
      
      # Визуализация
      disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
      disp.plot(cmap=plt.cm.Blues)
      plt.title('Матрица ошибок')
      plt.show()
      ```
  - Проблемы и особенности:
    - Несбалансированность классов: некоторые классы могут быть представлены значительно меньшим числом примеров
    - Вычислительная сложность: для One-vs-One подхода требуется обучить $\frac{K(K-1)}{2}$ классификаторов
    - Интерпретация вероятностей: калибровка вероятностных оценок для получения надежных предсказаний
- One-vs-All и One-vs-One подходы
  - One-vs-All (One-vs-Rest):
    - Принцип: для каждого класса обучается отдельный бинарный классификатор, который отличает этот класс от всех остальных
    - Для K классов требуется обучить K классификаторов
    - При предсказании выбирается класс с наибольшей уверенностью (вероятностью или расстоянием до разделяющей гиперплоскости)
    - Реализация в scikit-learn:
      ```python
      from sklearn.multiclass import OneVsRestClassifier
      from sklearn.svm import SVC
      
      # Создание мультиклассового классификатора
      clf = OneVsRestClassifier(SVC(kernel='linear'))
      clf.fit(X_train, y_train)
      ```
    - Преимущества: простота реализации, меньшее количество моделей по сравнению с One-vs-One
    - Недостатки: может страдать от несбалансированности данных, так как один класс сравнивается со всеми остальными
  - One-vs-One:
    - Принцип: обучение бинарного классификатора для каждой пары классов
    - Для K классов требуется обучить K(K-1)/2 классификаторов
    - При предсказании используется "голосование" - каждый классификатор "голосует" за один из двух классов, побеждает класс с наибольшим количеством голосов
    - Реализация в scikit-learn:
      ```python
      from sklearn.multiclass import OneVsOneClassifier
      from sklearn.svm import SVC
      
      # Создание мультиклассового классификатора
      clf = OneVsOneClassifier(SVC(kernel='linear'))
      clf.fit(X_train, y_train)
      ```
    - Преимущества: более устойчив к несбалансированности данных, каждый классификатор решает более простую задачу
    - Недостатки: требует обучения большего количества моделей, что может быть вычислительно затратно при большом количестве классов
- Softmax-регрессия
  - Принцип: обобщение логистической регрессии для многоклассовой классификации. Вместо предсказания вероятности принадлежности к одному классу, softmax-регрессия предсказывает распределение вероятностей по всем классам.
  - Функция softmax: преобразует вектор логитов в вектор вероятностей, сумма которых равна 1
    ```python
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Вычитание максимума для численной стабильности
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    ```
  - Реализация в scikit-learn:
    ```python
    from sklearn.linear_model import LogisticRegression
    
    # multi_class='multinomial' указывает на использование softmax-регрессии
    # solver='lbfgs' или 'newton-cg' необходимы для многоклассовой задачи
    clf = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    clf.fit(X_train, y_train)
    
    # Получение вероятностей принадлежности к каждому классу
    probabilities = clf.predict_proba(X_test)
    ```
  - Функция потерь: категориальная кросс-энтропия (categorical cross-entropy)
    ```python
    def categorical_crossentropy(y_true, y_pred):
        # y_true - one-hot encoded метки классов
        # y_pred - предсказанные вероятности
        return -np.sum(y_true * np.log(y_pred + 1e-10)) / len(y_true)
    ```
  - Преимущества:
    - Естественное обобщение логистической регрессии на многоклассовый случай
    - Предоставляет вероятностные оценки для всех классов
    - Простая интерпретация результатов
  - Недостатки:
    - Предполагает линейную разделимость классов
    - Может страдать от мультиколлинеарности признаков
    - Требует больше вычислительных ресурсов по сравнению с бинарной логистической регрессией
- Методы оптимизации для задач классификации
  - Градиентный спуск (Gradient Descent)
    - Принцип: итеративное обновление параметров модели в направлении, противоположном градиенту функции потерь
    - Варианты:
      - Пакетный градиентный спуск (Batch Gradient Descent): использует всю обучающую выборку для вычисления градиента
      - Стохастический градиентный спуск (SGD): использует один случайный пример для вычисления градиента
      - Мини-пакетный градиентный спуск (Mini-batch Gradient Descent): использует небольшую случайную подвыборку для вычисления градиента
    - Реализация в scikit-learn:
      ```python
      from sklearn.linear_model import SGDClassifier
      
      clf = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3)
      clf.fit(X_train, y_train)
      ```
  - Методы второго порядка
    - Метод Ньютона: использует вторые производные (матрицу Гессе) для более точного определения направления оптимизации
    - L-BFGS: аппроксимирует матрицу Гессе, требуя меньше памяти
    - Реализация в scikit-learn:
      ```python
      from sklearn.linear_model import LogisticRegression
      
      clf = LogisticRegression(solver='newton-cg')  # или 'lbfgs'
      clf.fit(X_train, y_train)
      ```
  - Адаптивные методы оптимизации
    - AdaGrad: адаптивно настраивает скорость обучения для каждого параметра
    - RMSProp: улучшает AdaGrad, используя экспоненциальное скользящее среднее
    - Adam: комбинирует идеи RMSProp и моментума, адаптивно настраивая скорость обучения и используя моментум
  - Регуляризация в оптимизации
    - L1-регуляризация (Lasso): добавляет штраф, пропорциональный абсолютной величине коэффициентов, способствует разреженности модели
    - L2-регуляризация (Ridge): добавляет штраф, пропорциональный квадрату коэффициентов, предотвращает переобучение
    - Elastic Net: комбинация L1 и L2 регуляризации
    - Реализация в scikit-learn:
      ```python
      from sklearn.linear_model import LogisticRegression
      
      # L1-регуляризация
      clf_l1 = LogisticRegression(penalty='l1', solver='liblinear', C=0.1)
      
      # L2-регуляризация
      clf_l2 = LogisticRegression(penalty='l2', C=0.1)
      
      # Elastic Net
      clf_elastic = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, C=0.1)
      ```
  - Оптимизация гиперпараметров
    - Grid Search: перебор всех комбинаций гиперпараметров из заданного набора
    - Random Search: случайный выбор комбинаций гиперпараметров
    - Байесовская оптимизация: последовательный выбор гиперпараметров на основе предыдущих результатов
    - Реализация в scikit-learn:
      ```python
      from sklearn.model_selection import GridSearchCV
      
      param_grid = {
          'C': [0.001, 0.01, 0.1, 1, 10, 100],
          'penalty': ['l1', 'l2']
      }
      
      grid_search = GridSearchCV(LogisticRegression(solver='liblinear'), param_grid, cv=5)
      grid_search.fit(X_train, y_train)
      
      best_params = grid_search.best_params_
      best_model = grid_search.best_estimator_
      ```

## 8. Метрики качества
- Метрики качества для задач классификации:
  - Confusion Matrix (матрица ошибок): таблица, показывающая соотношение между истинными и предсказанными классами
    - Структура для бинарной классификации:
      | Real \ Predicted       | Pred +       | Pred -       |
      |------------|--------------|--------------|
      | **Positive** | True + (TP)  | False - (FN) |
      | **Negative** | False + (FP) | True - (TN)  |
    - Смысл: позволяет детально анализировать ошибки классификации и понимать, какие типы ошибок совершает модель
    - Python (scikit-learn):
      ```python
      from sklearn.metrics import confusion_matrix

      # Построение матрицы ошибок
      cm = confusion_matrix(y_true, y_pred)
      ```
  - Accuracy (точность): доля правильных предсказаний среди всех предсказаний
    $$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$$
    - Смысл: показывает общую долю верных предсказаний, но может быть обманчивой при несбалансированных классах
  - Precision (точность): доля правильных положительных предсказаний среди всех положительных предсказаний
    $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$
    - Смысл: показывает, насколько можно доверять положительным предсказаниям модели (сколько из предсказанных положительных примеров действительно положительные)
  - Recall (полнота): доля правильных положительных предсказаний среди всех фактически положительных примеров
    $$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
    - Смысл: показывает, какую долю положительных примеров модель смогла обнаружить (сколько положительных примеров было правильно классифицировано)
  - F1-score: гармоническое среднее между Precision и Recall
    $$\text{F1} = \frac{2}{\frac{1}{Recall} + \frac{1}{Prescision}} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{TP}{TP + \frac{FP+FN}{2}}$$
    - Смысл: сбалансированная метрика, учитывающая как точность, так и полноту предсказаний
  - TPR (true positive rate): это то же что полнота (Recall), т.е. доля положительных объектов, правильно предсказанных положительными
    $$TPR = \frac{TP}{P} = \frac{TP}{TP+FN} $$
  - FPR (false positive rate): доля отрицательных объектов, неправильно предсказанных положительными
    $$FPR = \frac{FP}{N} = \frac{FP}{FP + TN} $$
- Порог классификации (threshold):
  - Определение: значение вероятности, при котором классификатор меняет решение с `-` на `+`
  - По умолчанию обычно равен 0.5
  - Изменение порога влияет на баланс между Precision и Recall:
    - Увеличение порога - ↑ Precision, но ↓ Recall, т.к. в `+` будут попадать только "действительно положительные" (с высокой вероятностью)
    - Уменьшение порога - ↑ Recall, но ↓ Precision
  - Выбор оптимального порога зависит от конкретной задачи и стоимости ошибок разного типа
  - Python (scikit-learn):
    ```python
    # Получение вероятностей классов
    y_proba = model.predict_proba(X_test)[:, 1]  # Вероятности положительного класса
    
    # Применение пользовательского порога
    custom_threshold = 0.7
    y_pred_custom = (y_proba >= custom_threshold).astype(int)
    ```
- ROC-AUC (Area Under the Receiver Operating Characteristic Curve):
  - ROC (Receiver Operating Characteristic)
    - кривая в осях [x=FPR, y=TPR], которая строится путем отображения FPR против TPR при различных порогах классификации. Т.е беря все значения порога из [0,1], можно вычислить все возможные TPR, FPR и построить кривую.
    - для задачи с конечной выборкой (а это все), ROC - ступенчатая, т.к. различных значений порогов (которые что-то меняют) не больше чем наблюдений в выборке.
  - AUC (Area Under Curve): площадь под кривой ROC
  - Смысл: измеряет способность модели различать классы независимо от выбранного порога. Значение 0.5 соответствует случайному угадыванию, 1.0 - идеальной классификации
  - Преимущества: нечувствительность к несбалансированным классам, оценка качества ранжирования
  - Недостатки: может быть оптимистичной при сильно несбалансированных данных
  - Python (scikit-learn):
    ```python
    from sklearn.metrics import roc_curve, auc
    
    # Получение вероятностей положительного класса
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Построение ROC-кривой
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    # дальше строим график
    ```
- PR-AUC (Area Under the Precision-Recall Curve):
  - Определение: площадь под кривой Precision-Recall, которая строится путем отображения Precision против Recall при различных порогах классификации
  - Смысл: оценивает компромисс между точностью и полнотой, особенно полезна при несбалансированных классах, когда положительные примеры редки
  - Преимущества: более информативна, чем ROC-AUC при несбалансированных данных с акцентом на положительный класс
  - Недостатки: значения сильно зависят от соотношения классов в данных
  - Python (scikit-learn):
    ```python
    from sklearn.metrics import precision_recall_curve, auc
    
    # Получение вероятностей положительного класса
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Построение PR-кривой
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)
    ```
- Hitrate@k (Hit Rate at k):
  - Определение: доля запросов, для которых релевантный элемент находится среди первых k результатов
  - Формула: $\text{Hitrate@k} = \frac{1}{|Q|} \sum_{q \in Q} \mathbb{1}(\text{релевантный элемент в топ-}k)$, где Q - множество запросов
  - Применение: рекомендательные системы, поисковые системы
  - Ограничения: не учитывает порядок элементов в топ-k, подходит только когда есть один релевантный элемент
  - Python:
    ```python
    def hitrate_at_k(recommended_items, relevant_items, k=10):
        hits = 0
        for user_id, items in recommended_items.items():
            # Проверяем, есть ли релевантные элементы в топ-k рекомендациях
            if any(item in relevant_items[user_id] for item in items[:k]):
                hits += 1
        return hits / len(recommended_items)
    ```
- DCG@k (Discounted Cumulative Gain at k):
  - Определение: метрика, оценивающая качество ранжирования с учетом позиции релевантных элементов
  - Формула: $\text{DCG@k} = \sum_{i=1}^{k} \frac{rel_i}{\log_2(i+1)}$, где $rel_i$ - релевантность элемента на позиции i
  - Учитывает как релевантность элементов, так и их позицию в списке
  - Более релевантные элементы на более высоких позициях дают больший вклад
  - Python:
    ```python
    def dcg_at_k(relevances, k=10):
        """Вычисляет DCG@k для списка релевантностей."""
        relevances = np.asarray(relevances)[:k]
        if len(relevances) > 0:
            return np.sum(relevances / np.log2(np.arange(2, len(relevances) + 2)))
        return 0.0
    ```
- nDCG@k (normalized Discounted Cumulative Gain at k):
  - Определение: нормализованная версия DCG@k, значения от 0 до 1
  - Формула: $\text{nDCG@k} = \frac{\text{DCG@k}}{\text{IDCG@k}}$, где IDCG@k - идеальный DCG@k (при оптимальном ранжировании)
  - Нормализация позволяет сравнивать результаты для разных запросов
  - Значение 1.0 означает идеальное ранжирование
  - Python:
    ```python
    def ndcg_at_k(relevances, k=10):
        """Вычисляет nDCG@k для списка релевантностей."""
        dcg = dcg_at_k(relevances, k)
        # Идеальный DCG - когда релевантности отсортированы по убыванию
        ideal_relevances = sorted(relevances, reverse=True)
        idcg = dcg_at_k(ideal_relevances, k)
        if idcg > 0:
            return dcg / idcg
        return 0.0
    ```
- Коэффициент детерминации (R²):
  - Определение: метрика, показывающая долю дисперсии зависимой переменной, объясняемую моделью
  - Формула: $R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}$, где $\hat{y}_i$ - предсказанные значения, $\bar{y}$ - среднее значение зависимой переменной
  - Интерпретация:
    - R² = 1: модель идеально описывает данные
    - R² = 0: модель не лучше, чем простое предсказание среднего значения
    - R² < 0: модель хуже, чем предсказание среднего значения
  - Ограничения: может увеличиваться при добавлении новых признаков, даже если они не улучшают модель
  - Скорректированный R² (adjusted R²): учитывает количество признаков в модели
    - Формула: $R^2_{adj} = 1 - \frac{(1-R^2)(n-1)}{n-p-1}$, где n - количество наблюдений, p - количество признаков
  - Python:
    ```python
    from sklearn.metrics import r2_score
    
    r2 = r2_score(y_true, y_pred)
    
    # Вычисление вручную
    def r2_manual(y_true, y_pred):
        ss_total = np.sum((y_true - np.mean(y_true))**2)
        ss_residual = np.sum((y_true - y_pred)**2)
        return 1 - (ss_residual / ss_total)
    ```
- Функционалы качества и Функции потерь
  - Определение и отличия:
    - Функция потерь (Loss Function): математическая функция, которая измеряет ошибку между предсказаниями модели и фактическими значениями для отдельных примеров. Используется непосредственно в процессе обучения модели для оптимизации параметров.
    - Функционал качества (Metric): метрика, которая оценивает общую производительность модели на наборе данных. Используется для оценки и сравнения моделей, но не всегда напрямую оптимизируется в процессе обучения.
    - Некоторые математические конструкции используются и как функции потерь, и как метрики, например MSE.
  - Ключевые отличия:
    - Цель использования: функции потерь оптимизируются в процессе обучения, функционалы качества используются для оценки результатов
    - Дифференцируемость: функции потерь обычно должны быть дифференцируемыми для использования в градиентных методах, функционалы качества не обязательно дифференцируемы (например, Accuracy, F1-score)
    - Интерпретируемость: функционалы качества часто более интерпретируемы для конечных пользователей (например, точность классификации в процентах)
    - Соответствие бизнес-задаче: функционалы качества обычно ближе к реальным бизнес-метрикам, функции потерь выбираются с учетом математических свойств и удобства оптимизации
  - Функции потерь для регрессии:
    - MSE (Mean Squared Error): $L(y, \hat{y}) = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$
      - Сильно штрафует за большие ошибки из-за квадратичной зависимости
      - Чувствительна к выбросам
      - Дифференцируема, что удобно для градиентных методов оптимизации
    - MAE (Mean Absolute Error): $L(y, \hat{y}) = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$
      - Более устойчива к выбросам, чем MSE
      - Штрафует все ошибки пропорционально их величине
      - Недифференцируема в нуле, что может создавать проблемы при оптимизации
    - RMSE (Root Mean Squared Error): $L(y, \hat{y}) = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$
      - Корень из MSE, имеет те же единицы измерения, что и целевая переменная
      - Сохраняет свойства MSE, но более интерпретируема
    - Huber Loss: комбинирует MSE и MAE
      - $L_\delta(y, \hat{y}) = \begin{cases}
        \frac{1}{2}(y - \hat{y})^2, & \text{если } |y - \hat{y}| \leq \delta \\
        \delta|y - \hat{y}| - \frac{1}{2}\delta^2, & \text{иначе}
        \end{cases}$
      - Менее чувствительна к выбросам, чем MSE, но сохраняет дифференцируемость
  - Функции потерь для классификации:
    - Log Loss (Binary Cross-Entropy): $L(y, \hat{p}) = -\frac{1}{n}\sum_{i=1}^{n}[y_i\log(\hat{p}_i) + (1-y_i)\log(1-\hat{p}_i)]$
      - Для бинарной классификации, где $\hat{p}_i$ - вероятность принадлежности к положительному классу
      - Сильно штрафует за уверенные, но неправильные предсказания
    - Categorical Cross-Entropy: $L(y, \hat{p}) = -\frac{1}{n}\sum_{i=1}^{n}\sum_{j=1}^{m}y_{ij}\log(\hat{p}_{ij})$
      - Обобщение Log Loss для многоклассовой классификации
      - $y_{ij}$ - индикатор принадлежности i-го объекта к j-му классу
    - Hinge Loss: $L(y, f) = \max(0, 1 - y \cdot f)$
      - Используется в SVM
      - $y \in \{-1, 1\}$, $f$ - выход модели
      - Штрафует не только за неправильные предсказания, но и за недостаточно уверенные правильные
    - Focal Loss: $L(p_t) = -\alpha_t(1-p_t)^\gamma\log(p_t)$
      - Модификация Cross-Entropy для несбалансированных данных
      - Уменьшает вес легко классифицируемых примеров
  - Функции потерь для ранжирования:
    - Pairwise Ranking Loss: оценивает правильность относительного порядка пар объектов
    - ListNet Loss: оптимизирует вероятность перестановки всего списка
    - LambdaRank: учитывает изменение метрик ранжирования при перестановке объектов
  - Реализация в Python:
    ```python
    import numpy as np
    from sklearn.metrics import mean_squared_error, mean_absolute_error, log_loss
    
    # MSE
    mse = mean_squared_error(y_true, y_pred)
    # или вручную
    mse_manual = np.mean((y_true - y_pred)**2)
    
    # MAE
    mae = mean_absolute_error(y_true, y_pred)
    # или вручную
    mae_manual = np.mean(np.abs(y_true - y_pred))
    
    # RMSE
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Log Loss для бинарной классификации
    logloss = log_loss(y_true, y_pred)
    ```
- Выбор метрик для конкретных задач:
  - Для задач регрессии:
    - RMSE: когда важно сильнее штрафовать большие ошибки и данные относительно чистые
    - MAE: когда в данных есть выбросы, которые могут сильно искажать MSE/RMSE
    - R²: для оценки доли объясненной дисперсии и сравнения моделей
    - MAPE: когда важна относительная ошибка в процентах
  - Для задач классификации:
    - Accuracy: только для сбалансированных данных, когда классы равнозначны
    - Precision: когда важно минимизировать ложноположительные срабатывания
    - Recall: когда важно минимизировать ложноотрицательные срабатывания
    - F1-score: когда нужен баланс между precision и recall
    - AUC-ROC: для оценки качества ранжирования и работы с несбалансированными данными
    - AUC-PR: для сильно несбалансированных данных, когда положительный класс редкий
  - Для задач ранжирования:
    - NDCG (Normalized Discounted Cumulative Gain): учитывает порядок и релевантность
    - MAP (Mean Average Precision): оценивает точность на разных уровнях полноты
    - MRR (Mean Reciprocal Rank): фокусируется на позиции первого релевантного результата
  - Для рекомендательных систем:
    - Precision@k и Recall@k: оценка точности и полноты для топ-k рекомендаций
    - Hit Rate: доля пользователей, получивших хотя бы одну релевантную рекомендацию
    - Diversity: разнообразие рекомендаций
    - Coverage: охват каталога рекомендациями

## 9. Кросс-валидация и подбор гиперпараметров
- GridSearchCV
- K-fold cross-validation
- Метрики качества для подбора параметров
- Переобучение и недообучение
- Байесовская оптимизация
- Ранняя остановка

## 10. Деревья решений и ансамбли
- Гиперпараметры деревьев решений
  - `max_depth`: максимальная глубина дерева
    - Ограничивает количество уровней в дереве
    - Помогает бороться с переобучением
    - Типичные значения: 3-10 для простых моделей, больше для сложных задач
  - `min_samples_split`: минимальное количество образцов для разделения узла
    - Определяет, сколько образцов должно быть в узле, чтобы его можно было разделить
    - Увеличение значения помогает предотвратить переобучение
    - По умолчанию: 2 (в scikit-learn)
  - `min_samples_leaf`: минимальное количество образцов в листовом узле
    - Гарантирует, что каждый лист будет содержать не менее указанного числа образцов
    - Важный параметр для контроля переобучения
    - Типичные значения: 1-10, зависит от размера датасета
  - `max_features`: максимальное количество признаков для поиска оптимального разделения
    - Ограничивает число признаков, рассматриваемых при каждом разделении
    - Варианты: 'sqrt', 'log2', число или доля от общего количества признаков
    - Помогает увеличить разнообразие в ансамблях (например, в случайном лесе)
  - `criterion`: критерий для оценки качества разделения
    - Для классификации: 'gini' (индекс Джини) или 'entropy' (информационная энтропия)
    - Для регрессии: 'mse' (среднеквадратичная ошибка) или 'mae' (средняя абсолютная ошибка)
  - `class_weight`: веса классов
    - Полезно для несбалансированных данных
    - Можно задать вручную или использовать 'balanced'
  - `random_state`: начальное значение для генератора случайных чисел
    - Обеспечивает воспроизводимость результатов
  - `ccp_alpha`: параметр сложности для пост-отсечения ветвей (cost-complexity pruning)
    - Контролирует отсечение ветвей после построения дерева
    - Более высокие значения приводят к более сильной регуляризации
- Случайный лес (Random Forest):
  - Принцип работы: ансамбль деревьев решений, где каждое дерево обучается на случайной подвыборке данных с использованием случайного подмножества признаков
  - Особенности:
    - Использует бэггинг (bootstrap aggregating) для создания разнообразных деревьев
    - Каждое дерево "голосует" за класс (в классификации) или предсказывает значение (в регрессии)
    - Итоговое предсказание - результат агрегации (большинство голосов или среднее значение)
  - Гиперпараметры:
    - `n_estimators`: количество деревьев в лесу (обычно 100-500)
    - `max_features`: количество признаков для рассмотрения при поиске лучшего разделения (обычно 'sqrt' или 'log2')
    - Параметры отдельных деревьев (max_depth, min_samples_split и т.д.)
  - Преимущества:
    - Устойчивость к переобучению
    - Хорошая точность на многих задачах
    - Встроенная оценка важности признаков
    - Параллельное обучение деревьев
  - Недостатки:
    - Менее интерпретируемый, чем одиночное дерево
    - Требует больше памяти и вычислительных ресурсов
  - Реализация в scikit-learn:
    ```python
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    
    # Для классификации
    rf_clf = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, random_state=42)
    rf_clf.fit(X_train, y_train)
    
    # Для регрессии
    rf_reg = RandomForestRegressor(n_estimators=100, max_depth=None, min_samples_split=2, random_state=42)
    rf_reg.fit(X_train, y_train)
    ```
- Out-of-bag error (OOB error)
  - Принцип: метод оценки ошибки модели, использующий образцы, не попавшие в обучающую выборку при бутстрэппинге
  - Особенности:
    - Каждое дерево в случайном лесу обучается на подвыборке данных (около 63% от исходных)
    - Оставшиеся ~37% образцов (out-of-bag samples) используются для оценки качества модели
    - Не требует отдельной тестовой выборки или кросс-валидации
  - Применение:
    - Оценка обобщающей способности модели
    - Настройка гиперпараметров
    - Мониторинг переобучения
  - Реализация в scikit-learn:
    ```python
    from sklearn.ensemble import RandomForestClassifier
    
    # Включение расчета OOB ошибки
    rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=42)
    rf.fit(X, y)
    
    # Получение OOB оценки
    oob_score = rf.oob_score_
    print(f"Out-of-bag score: {oob_score:.4f}")
    
    # Для получения OOB предсказаний
    oob_prediction = rf.oob_decision_function_
    ```
  - Преимущества:
    - Эффективное использование данных
    - Хорошая оценка обобщающей способности модели
    - Экономия вычислительных ресурсов (не требуется отдельная валидация)
- Bagging (Bootstrap Aggregating)
  - Принцип: создание множества моделей, обученных на разных подвыборках исходных данных, и объединение их предсказаний
  - Особенности:
    - Подвыборки создаются с помощью бутстрэпа (случайное семплирование с возвращением)
    - Каждая модель обучается независимо от других
    - Предсказания объединяются путем голосования (для классификации) или усреднения (для регрессии)
  - Преимущества:
    - Снижение дисперсии и переобучения
    - Повышение стабильности модели
    - Параллельное обучение моделей
  - Недостатки:
    - Не снижает смещение (bias) модели
    - Требует больше вычислительных ресурсов
  - Реализация в scikit-learn:
    ```python
    from sklearn.ensemble import BaggingClassifier, BaggingRegressor
    from sklearn.tree import DecisionTreeClassifier
    
    # Для классификации
    bagging_clf = BaggingClassifier(
        base_estimator=DecisionTreeClassifier(),
        n_estimators=100,
        max_samples=0.8,
        bootstrap=True,
        random_state=42
    )
    bagging_clf.fit(X_train, y_train)
    
    # Для регрессии
    bagging_reg = BaggingRegressor(
        base_estimator=None,  # По умолчанию используется DecisionTreeRegressor
        n_estimators=100,
        max_samples=0.8,
        bootstrap=True,
        random_state=42
    )
    bagging_reg.fit(X_train, y_train)
    ```
- Boosting
  - Принцип: последовательное обучение моделей, где каждая следующая модель уделяет больше внимания ошибкам предыдущих
  - Особенности:
    - Модели обучаются последовательно, а не параллельно как в bagging
    - Каждая новая модель фокусируется на сложных примерах, которые предыдущие модели классифицировали неверно
    - Итоговое предсказание формируется как взвешенная сумма предсказаний всех моделей
  - Преимущества:
    - **Снижает как дисперсию, так и смещение модели**
    - Часто достигает лучших результатов, чем bagging
    - Эффективно работает даже со слабыми моделями
  - Недостатки:
    - Склонность к переобучению при большом числе итераций
    - Чувствительность к шуму и выбросам
    - Последовательное обучение (нельзя распараллелить)
  - Реализация AdaBoost в scikit-learn:
    ```python
    from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
    from sklearn.tree import DecisionTreeClassifier
    
    # Для классификации
    ada_clf = AdaBoostClassifier(
        base_estimator=DecisionTreeClassifier(max_depth=1),  # "пень" (stump)
        n_estimators=50,
        learning_rate=1.0,
        random_state=42
    )
    ada_clf.fit(X_train, y_train)
    
    # Для регрессии
    ada_reg = AdaBoostRegressor(
        base_estimator=None,  # По умолчанию используется DecisionTreeRegressor
        n_estimators=50,
        learning_rate=1.0,
        random_state=42
    )
    ada_reg.fit(X_train, y_train)
    ```
- Stacking
  - Принцип: обучение мета-модели на предсказаниях базовых моделей
  - Особенности:
    - Базовые модели обучаются на обучающей выборке и делают предсказания на валидационной
    - Мета-модель обучается на предсказаниях базовых моделей для валидационной выборки
    - Для финального предсказания базовые модели переобучаются на всех данных
  - Преимущества:
    - Может достигать лучших результатов, чем отдельные модели
    - Позволяет комбинировать разнородные модели
    - Снижает дисперсию и смещение
  - Недостатки:
    - Сложность реализации и интерпретации
    - Требует больше вычислительных ресурсов
    - Риск переобучения мета-модели
  - Реализация в scikit-learn:
    ```python
    from sklearn.ensemble import StackingClassifier, StackingRegressor
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    
    # Для классификации
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('svm', SVC(probability=True, random_state=42)),
        ('knn', KNeighborsClassifier(n_neighbors=5))
    ]
    
    stacking_clf = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(),
        cv=5  # Кросс-валидация для обучения базовых моделей
    )
    stacking_clf.fit(X_train, y_train)
    
    # Для регрессии
    estimators = [
        ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
        ('gbr', GradientBoostingRegressor(random_state=42)),
        ('lr', LinearRegression())
    ]
    
    stacking_reg = StackingRegressor(
        estimators=estimators,
        final_estimator=LinearRegression(),
        cv=5
    )
    stacking_reg.fit(X_train, y_train)
    ```
- Влияние ансамблевых методов на смещение и разброс
  - Принцип: ансамблевые методы позволяют уменьшить ошибку модели за счет снижения дисперсии (разброса) и/или смещения
  - Бэггинг и случайные леса:
    - Основное влияние: снижение дисперсии при сохранении смещения
    - Усреднение предсказаний нескольких моделей уменьшает разброс
    - Особенно эффективны для моделей с высокой дисперсией (например, деревья решений)
  - Бустинг:
    - Основное влияние: снижение смещения и дисперсии
    - Последовательное обучение моделей на ошибках предыдущих уменьшает смещение
    - При правильной настройке может снижать и дисперсию
  - Стекинг:
    - Комбинирует преимущества разных моделей
    - Может снижать как смещение, так и дисперсию в зависимости от базовых моделей
  - Визуализация влияния на смещение и разброс:
    ```python
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.datasets import make_regression
    
    # Генерация данных
    X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)
    
    # Создание моделей
    tree = DecisionTreeRegressor(max_depth=3)
    rf = RandomForestRegressor(n_estimators=100, max_depth=3)
    gbr = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1)
    
    # Обучение моделей
    tree.fit(X, y)
    rf.fit(X, y)
    gbr.fit(X, y)
    
    # Создание сетки для визуализации
    X_grid = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    
    # Предсказания
    y_tree = tree.predict(X_grid)
    y_rf = rf.predict(X_grid)
    y_gbr = gbr.predict(X_grid)
    
    # Визуализация
    plt.figure(figsize=(12, 6))
    plt.scatter(X, y, color='black', label='Данные')
    plt.plot(X_grid, y_tree, color='blue', label='Дерево решений', linewidth=2)
    plt.plot(X_grid, y_rf, color='red', label='Случайный лес', linewidth=2)
    plt.plot(X_grid, y_gbr, color='green', label='Градиентный бустинг', linewidth=2)
    plt.title('Сравнение моделей: влияние на смещение и разброс')
    plt.xlabel('Признак')
    plt.ylabel('Целевая переменная')
    plt.legend()
    plt.grid(True)
    plt.show()
    ```
- Градиентный бустинг
- Случайные подпространства (Random Subspaces)
- Разные виды бустинга (AdaBoost, Gradient Boosting)
- Особенности реализации градиентного бустинга
- Регуляризация в градиентном бустинге
- XGBoost, LightGBM и их особенности
- Интерпретация ансамблевых моделей
- Feature importance в ансамблях

## 11. Ядерные методы
- Ядерные функции (kernel functions)
- Kernel trick
- Радиальные базисные функции (RBF)
- Полиномиальные ядра
- Ядерная регрессия
- Ядерная классификация
- Метод опорных векторов (SVM)
- Двойственная задача оптимизации

## 12. Методы снижения размерности
- PCA (метод главных компонент)
  - Принцип: метод снижения размерности, который преобразует исходные признаки в новые, некоррелированные переменные (главные компоненты), упорядоченные по убыванию дисперсии. Первые компоненты сохраняют наибольшую часть информации из исходных данных.
  - Python (scikit-learn):
    ```python
    from sklearn.decomposition import PCA
    
    # Создание и обучение модели PCA
    pca = PCA(n_components=2)  # Снижение до 2 компонент
    X_pca = pca.fit_transform(X)
    
    # Объясненная дисперсия
    explained_variance = pca.explained_variance_ratio_
    print(f"Объясненная дисперсия: {explained_variance}")
    print(f"Суммарная объясненная дисперсия: {sum(explained_variance)}")
    
    # Визуализация результатов
    plt.figure(figsize=(10, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
    plt.xlabel('Первая главная компонента')
    plt.ylabel('Вторая главная компонента')
    plt.title('Проекция данных на первые две главные компоненты')
    plt.colorbar(label='Целевая переменная')
    plt.grid(True)
    plt.show()
    ```
  - Выбор количества компонент:
    ```python
    # Визуализация объясненной дисперсии
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
             np.cumsum(pca.explained_variance_ratio_), marker='o')
    plt.xlabel('Количество компонент')
    plt.ylabel('Суммарная объясненная дисперсия')
    plt.axhline(y=0.9, color='r', linestyle='-')
    plt.title('Выбор количества главных компонент')
    plt.grid(True)
    plt.show()
    ```
- Собственные значения и векторы
  - Принцип: в контексте PCA, собственные векторы ковариационной матрицы определяют направления главных компонент, а соответствующие собственные значения указывают на величину дисперсии вдоль этих направлений. Чем больше собственное значение, тем больше информации содержится в соответствующей главной компоненте.
  - Python (numpy):
    ```python
    import numpy as np
    
    # Вычисление ковариационной матрицы
    X_centered = X - X.mean(axis=0)
    cov_matrix = np.cov(X_centered, rowvar=False)
    
    # Вычисление собственных значений и векторов
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # Сортировка собственных значений и векторов в порядке убывания
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Визуализация собственных значений
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(eigenvalues)), eigenvalues)
    plt.xlabel('Компонента')
    plt.ylabel('Собственное значение')
    plt.title('Собственные значения ковариационной матрицы')
    plt.grid(True)
    plt.show()
    ```
- Центрирование и нормирование данных
  - Принцип: подготовка данных путем центрирования (вычитание среднего значения) и нормирования (деление на стандартное отклонение) для улучшения работы алгоритмов машинного обучения. Это особенно важно для методов, чувствительных к масштабу признаков.
  - Центрирование данных (Python):
    ```python
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    
    # Центрирование данных вручную
    X_centered = X - np.mean(X, axis=0)
    
    # Использование StandardScaler для центрирования
    scaler = StandardScaler(with_std=False)
    X_centered = scaler.fit_transform(X)
    ```
  - Стандартизация данных (Z-нормализация):
    ```python
    # Стандартизация вручную
    X_standardized = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    
    # Использование StandardScaler
    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(X)
    ```
  - Нормализация данных (масштабирование в диапазон [0, 1]):
    ```python
    from sklearn.preprocessing import MinMaxScaler
    
    # Нормализация вручную
    X_normalized = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    
    # Использование MinMaxScaler
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X)
    ```
  - Визуализация эффекта нормализации:
    ```python
    import matplotlib.pyplot as plt
    
    # Визуализация распределения до и после нормализации
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(X[:, 0], bins=30)
    plt.title('Исходные данные')
    plt.xlabel('Значение признака')
    plt.ylabel('Частота')
    
    plt.subplot(1, 2, 2)
    plt.hist(X_standardized[:, 0], bins=30)
    plt.title('Стандартизированные данные')
    plt.xlabel('Значение признака')
    plt.ylabel('Частота')
    
    plt.tight_layout()
    plt.show()
    ```
- Латентные модели
- Тематическое моделирование
- Рекомендательные системы

## 13. Кластеризация
- K-means
  - Принцип: алгоритм разделяет данные на K кластеров, минимизируя сумму квадратов расстояний от точек до центров кластеров. Итеративно назначает точки ближайшим центрам и пересчитывает центры.
  - Базовая реализация с использованием scikit-learn:
    ```python
    from sklearn.cluster import KMeans
    
    # Создание и обучение модели
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X)
    
    # Получение меток кластеров и центроидов
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    ```
  - Визуализация результатов кластеризации:
    ```python
    import matplotlib.pyplot as plt
    
    # Визуализация кластеров (для 2D данных)
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, alpha=0.8)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, marker='X')
    plt.title('Результаты кластеризации K-means')
    plt.xlabel('Признак 1')
    plt.ylabel('Признак 2')
    plt.show()
    ```
  - Определение оптимального числа кластеров с помощью метода локтя:
    ```python
    # Метод локтя для определения оптимального числа кластеров
    inertia = []
    k_range = range(1, 11)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertia.append(kmeans.inertia_)
    
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, inertia, marker='o')
    plt.title('Метод локтя для определения оптимального числа кластеров')
    plt.xlabel('Число кластеров')
    plt.ylabel('Инерция (сумма квадратов расстояний)')
    plt.grid(True)
    plt.show()
    ```
  - Силуэтный анализ для оценки качества кластеризации:
    ```python
    from sklearn.metrics import silhouette_score, silhouette_samples
    
    # Расчет среднего силуэтного коэффициента
    silhouette_avg = silhouette_score(X, labels)
    print(f"Средний силуэтный коэффициент: {silhouette_avg:.3f}")
    
    # Детальный силуэтный анализ
    sample_silhouette_values = silhouette_samples(X, labels)
    
    # Визуализация силуэтных коэффициентов
    plt.figure(figsize=(10, 6))
    y_lower = 10
    
    for i in range(kmeans.n_clusters):
        ith_cluster_values = sample_silhouette_values[labels == i]
        ith_cluster_values.sort()
        
        size_cluster_i = ith_cluster_values.shape[0]
        y_upper = y_lower + size_cluster_i
        
        plt.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_values,
                         alpha=0.7, color=plt.cm.viridis(i / kmeans.n_clusters))
        
        plt.text(-0.05, y_lower + 0.5 * size_cluster_i, f'Кластер {i}')
        y_lower = y_upper + 10
    
    plt.title("Силуэтный анализ кластеризации")
    plt.xlabel("Силуэтный коэффициент")
    plt.ylabel("Кластер")
    plt.axvline(x=silhouette_avg, color="red", linestyle="--")
    plt.show()
    ```
- Метод DBSCAN (Density-Based Spatial Clustering of Applications with Noise):
  - Принципы работы:
    - Кластеризация на основе плотности точек
    - Не требует предварительного задания числа кластеров
    - Способен находить кластеры произвольной формы
    - Выявляет и отмечает наблюдения-выбросы (не включает ни в один класс)
    - Устойчив к выбросам (шуму)
    - Использует два ключевых параметра: eps (радиус окрестности) и min_samples (минимальное число точек в окрестности)
  - Пример реализации:
    ```python
    from sklearn.cluster import DBSCAN
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Создание модели DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    
    # Обучение модели
    clusters = dbscan.fit_predict(X)
    
    # Визуализация результатов
    plt.figure(figsize=(10, 6))
    
    # Отображение точек по кластерам
    unique_labels = np.unique(clusters)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    
    for label, color in zip(unique_labels, colors):
        if label == -1:  # Шум
            color = 'gray'
        
        mask = clusters == label
        plt.scatter(X[mask, 0], X[mask, 1], c=[color], label=f'Кластер {label}' if label != -1 else 'Шум')
    
    plt.title('Результаты кластеризации DBSCAN')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Анализ результатов
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = list(clusters).count(-1)
    print(f'Количество кластеров: {n_clusters}')
    print(f'Количество выбросов: {n_noise} ({n_noise/len(X)*100:.2f}%)')
    ```
- Иерархическая кластеризация (agglomerative):
  - Принципы работы:
    - Объединение близких кластеров снизу вверх
    - Начинает с отдельных точек как кластеров
    - Постепенно объединяет ближайшие кластеры
    - Не требует предварительного задания числа кластеров
    - Результат представляется в виде дендрограммы
    - Итоговое число классов определяется "уровнем отсечки" на дендрограмме
    - Различные метрики расстояния (евклидово, манхэттенское и др.)
    - Различные методы связи (одиночная, полная, средняя, метод Уорда)
  - Пример реализации:
    ```python
    from sklearn.cluster import AgglomerativeClustering
    from scipy.cluster.hierarchy import dendrogram
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Создание модели иерархической кластеризации
    model = AgglomerativeClustering(n_clusters=3, linkage='ward')
    
    # Обучение модели
    clusters = model.fit_predict(X)
    
    # Визуализация результатов кластеризации
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis', s=50)
    plt.title('Результаты иерархической кластеризации')
    plt.grid(True)
    
    # Построение дендрограммы
    def plot_dendrogram(model, **kwargs):
        # Создаем связи для построения дендрограммы
        counts = np.zeros(model.children_.shape[0])
        n_samples = len(model.labels_)
        for i, merge in enumerate(model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        linkage_matrix = np.column_stack([model.children_, model.distances_,
                                          counts]).astype(float)
        
        # Строим дендрограмму
        dendrogram(linkage_matrix, **kwargs)
    
    # Модель для дендрограммы (с сохранением расстояний)
    model_dendrogram = AgglomerativeClustering(distance_threshold=0, n_clusters=None, linkage='ward')
    model_dendrogram = model_dendrogram.fit(X)
    
    plt.figure(figsize=(12, 6))
    plot_dendrogram(model_dendrogram, truncate_mode='level', p=3)
    plt.title('Дендрограмма иерархической кластеризации')
    plt.xlabel('Индексы образцов или (количество кластеров)')
    plt.ylabel('Расстояние')
    plt.show()
    
    # Анализ результатов
    print(f'Количество кластеров: {len(np.unique(clusters))}')
    for i in range(len(np.unique(clusters))):
        print(f'Размер кластера {i}: {np.sum(clusters == i)} образцов')
    ```
- Сравнение алгоритмов кластеризации
  - K-means vs иерархическая кластеризация vs DBSCAN
    - K-means: центроидный метод, требует заранее заданного числа кластеров, хорошо работает с выпуклыми кластерами
    - Иерархическая кластеризация: строит дерево кластеров, не требует предварительного указания числа кластеров, может визуализироваться через дендрограмму
    - DBSCAN: основан на плотности, автоматически определяет число кластеров, хорошо находит кластеры произвольной формы и устойчив к выбросам
    - Различия в вычислительной сложности: K-means O(n), иерархическая O(n²), DBSCAN O(n log n)
    - Различия в параметрах: K-means (k), иерархическая (метрика связи), DBSCAN (eps, min_samples)
  - Спектральная кластеризация vs другие методы
    - Спектральная кластеризация: использует собственные векторы матрицы подобия для снижения размерности перед кластеризацией
    - Хорошо работает с нелинейно разделимыми данными и сложными структурами
    - Может находить кластеры произвольной формы, подобно DBSCAN
    - Требует вычисления матрицы подобия, что может быть вычислительно затратно для больших наборов данных
    - Чувствительна к выбору параметров: функции подобия и количеству кластеров
    - В отличие от K-means, лучше справляется с некомпактными кластерами
    - По сравнению с иерархической кластеризацией, более эффективна для больших наборов данных
  - Сравнение по скорости работы
    - K-means: обычно самый быстрый из популярных алгоритмов, линейная сложность O(n)
    - DBSCAN: средняя скорость, сложность O(n log n) при использовании индексных структур
    - Иерархическая кластеризация: медленнее для больших наборов данных, квадратичная сложность O(n²)
    - Спектральная кластеризация: наиболее вычислительно затратная из-за необходимости вычисления собственных векторов, O(n³) в худшем случае
    - Время выполнения зависит от размерности данных, количества образцов и реализации алгоритма
    - На практике K-means может быть в десятки раз быстрее иерархической кластеризации для больших наборов данных
  - Сравнение по устойчивости к шумам
    - DBSCAN: наиболее устойчив к шумам и выбросам, так как явно определяет шумовые точки
    - K-means: чувствителен к выбросам, которые могут значительно смещать центроиды
    - Иерархическая кластеризация: средняя устойчивость, зависит от выбранного метода связи (метод полной связи более устойчив)
    - Спектральная кластеризация: умеренно устойчива, но может быть чувствительна к шуму в данных
    - Выбросы могут существенно влиять на результаты K-means, но меньше влияют на DBSCAN
  - Сравнение по способности находить кластеры разной формы
    - K-means: хорошо работает только с выпуклыми, сферическими кластерами примерно одинакового размера
    - DBSCAN: отлично находит кластеры произвольной формы, включая концентрические и нелинейные структуры
    - Иерархическая кластеризация: может находить кластеры разной формы, но результат зависит от выбранной метрики и метода связи
    - Спектральная кластеризация: эффективна для нелинейно разделимых данных и кластеров сложной формы
    - Для данных с кластерами неправильной формы DBSCAN и спектральная кластеризация обычно превосходят K-means
  - Применимость к большим данным
    - K-means: хорошо масштабируется для больших наборов данных, существуют эффективные реализации (Mini-Batch K-means)
    - DBSCAN: может работать с большими данными при использовании пространственных индексов, но требует оптимизации
    - Иерархическая кластеризация: плохо подходит для больших данных из-за квадратичной сложности и требований к памяти
    - Спектральная кластеризация: ограничена для очень больших наборов данных из-за вычислительной сложности
    - Для больших данных часто используют аппроксимации или выборки для более сложных алгоритмов
  - Масштабируемость алгоритмов
    - K-means: легко распараллеливается, существуют распределенные реализации (Spark K-means)
    - DBSCAN: умеренно масштабируемый, существуют параллельные версии, но сложнее распараллелить полностью
    - Иерархическая кластеризация: плохо масштабируется, хотя существуют приближенные версии (BIRCH)
    - Спектральная кластеризация: сложно масштабируется из-за необходимости вычисления собственных векторов
    - Для очень больших наборов данных часто используют инкрементальные или онлайн-версии алгоритмов
    - Существуют специальные алгоритмы для больших данных: BIRCH, CURE, CLARANS
- Оценка качества кластеризации
  - Внутренние метрики оценки
    - Силуэтный коэффициент: измеряет, насколько объект похож на свой кластер по сравнению с другими кластерами
    - Индекс Дэвиса-Болдина: оценивает среднее сходство между кластерами
    - Индекс Калински-Харабаза: отношение разброса между кластерами к разбросу внутри кластеров
    - Коэффициент Данна: отношение минимального расстояния между кластерами к максимальному диаметру кластера
  - Внешние метрики оценки
    - Adjusted Rand Index (ARI): измеряет сходство между двумя разбиениями с поправкой на случайность
    - Normalized Mutual Information (NMI): нормализованная взаимная информация между истинными метками и предсказанными кластерами
    - V-мера: гармоническое среднее полноты и точности кластеризации
    - Homogeneity и Completeness: оценивают, насколько каждый кластер содержит только элементы одного класса и все элементы класса
  - Визуальные методы оценки
    - Дендрограммы для иерархической кластеризации
    - Метод локтя (Elbow method) для определения оптимального числа кластеров
    - Визуализация силуэтных коэффициентов
    - t-SNE или UMAP для визуализации кластеров в двумерном пространстве
  - Практические аспекты оценки
    - Выбор метрики в зависимости от задачи и данных
    - Сравнение результатов разных алгоритмов кластеризации
    - Стабильность кластеризации при изменении параметров
    - Интерпретируемость полученных кластеров
- Выбор количества кластеров
  - Метод локтя (Elbow method)
    - Построение графика суммы квадратов расстояний внутри кластеров (WCSS) в зависимости от числа кластеров
    - Выбор точки "локтя" на графике, где добавление новых кластеров даёт меньший прирост качества
    - Простой, но субъективный метод, так как точка "локтя" не всегда очевидна
  - Силуэтный анализ
    - Вычисление силуэтного коэффициента для разного числа кластеров
    - Выбор числа кластеров с максимальным средним силуэтным коэффициентом
    - Более объективный, но вычислительно затратный для больших наборов данных
  - Метод Gap-статистики
    - Сравнение наблюдаемой внутрикластерной дисперсии с ожидаемой для случайных данных
    - Выбор числа кластеров, где разница (gap) максимальна
    - Статистически обоснованный, но вычислительно сложный метод
  - Информационные критерии
    - Байесовский информационный критерий (BIC)
    - Критерий Акаике (AIC)
    - Выбор модели с минимальным значением критерия
  - Методы стабильности кластеризации
    - Оценка стабильности результатов при разных инициализациях или подвыборках данных
    - Выбор числа кластеров, дающего наиболее стабильные результаты
  - Практические рекомендации
    - Использование нескольких методов одновременно для перекрёстной проверки
    - Учёт предметной области и интерпретируемости результатов
    - Визуализация данных для предварительной оценки возможного числа кластеров

## 14. Обработка текстовых данных
- Bag of words (мешок слов)
  - Модель представления текста, игнорирующая грамматику и порядок слов
  - Основные характеристики:
    - Текст представляется как неупорядоченный набор слов
    - Каждый документ описывается вектором частот слов
    - Размерность вектора равна размеру словаря
  - Процесс создания:
    - Создание словаря из всех уникальных слов в корпусе
    - Подсчет частоты каждого слова в каждом документе
    - Формирование разреженной матрицы документ-термин
  - Преимущества:
    - Простота реализации
    - Интуитивно понятная интерпретация
    - Эффективность для базовых задач классификации текста
  - Недостатки:
    - Игнорирование порядка слов и контекста
    - Высокая размерность векторов
    - Проблема редких слов и синонимов
  - Варианты и модификации:
    - Бинарный мешок слов (наличие/отсутствие слова)
    - N-граммы для частичного учета порядка слов
    - Взвешивание слов (например, с помощью TF-IDF)
- Лемматизация
  - Процесс приведения словоформы к её нормальной (словарной) форме — лемме
  - Основные характеристики:
    - Учитывает морфологию языка и контекст
    - Более сложный процесс, чем стемминг
    - Требует словаря и правил языка
  - Применение:
    - Улучшение качества поиска информации
    - Нормализация текста для анализа
    - Уменьшение размерности в моделях обработки текста
  - Отличия от стемминга:
    - Стемминг просто отсекает окончания, лемматизация приводит к словарной форме
    - Лемматизация учитывает части речи и контекст
    - Лемматизация даёт более точные результаты, но требует больше ресурсов
  - Инструменты для лемматизации:
    - NLTK, SpaCy, pymorphy2 (для русского языка)
    - WordNet Lemmatizer
    - Стэнфордский CoreNLP
- Токенизация
  - Процесс разделения текста на отдельные токены (слова, символы, n-граммы)
  - Основные подходы:
    - Токенизация по пробелам и знакам препинания
    - Токенизация на основе регулярных выражений
    - Токенизация с учетом особенностей языка
  - Особенности и проблемы:
    - Обработка знаков препинания
    - Обработка сокращений и аббревиатур
    - Обработка составных слов
    - Работа с разными языками и кодировками
  - Применение:
    - Предварительная обработка текста для дальнейшего анализа
    - Подготовка данных для моделей машинного обучения
    - Индексация текста для поисковых систем
  - Инструменты для токенизации:
    - NLTK, SpaCy, Tokenizers (из библиотеки Hugging Face)
    - Встроенные методы в языках программирования
    - Специализированные токенизаторы для конкретных языков
- Удаление стоп-слов
  - Определение: процесс исключения из текста общеупотребительных слов, которые не несут значимой смысловой нагрузки
  - Примеры стоп-слов:
    - Предлоги, союзы, частицы (в, на, и, а, но, бы)
    - Местоимения (я, ты, он, она, они)
    - Вспомогательные глаголы (быть, иметь)
    - Некоторые наречия и прилагательные
  - Цели удаления стоп-слов:
    - Уменьшение размерности данных
    - Повышение эффективности обработки текста
    - Фокусировка на значимых словах
  - Особенности применения:
    - Зависимость от языка (разные списки стоп-слов для разных языков)
    - Зависимость от предметной области (в некоторых контекстах "обычные" слова могут быть значимыми)
    - Не всегда полезно для всех задач (например, для анализа тональности некоторые стоп-слова могут быть важны)
  - Инструменты:
    - Готовые списки стоп-слов в библиотеках NLTK, SpaCy, Gensim
    - Возможность создания пользовательских списков стоп-слов
    - Автоматическое определение стоп-слов на основе частотного анализа
- TF-IDF (Term Frequency-Inverse Document Frequency)
  - Статистическая мера для оценки важности слова в контексте документа и корпуса
  - Term Frequency (TF): частота встречаемости термина в документе
    - Показывает, насколько часто слово встречается в документе
    - Нормализуется делением на общее количество слов в документе
  - Inverse Document Frequency (IDF): обратная частота документа
    - Показывает, насколько термин является редким или общим в корпусе документов
    - Рассчитывается как логарифм от отношения общего числа документов к числу документов, содержащих термин
  - TF-IDF = TF × IDF: высокие значения получают слова с высокой частотой в конкретном документе и низкой частотой в других документах
  - Применение:
    - Ранжирование результатов поиска
    - Извлечение ключевых слов
    - Классификация текстов
    - Векторизация документов для машинного обучения
- Word embeddings (Векторные представления слов)
  - Определение: представление слов в виде векторов в многомерном пространстве
  - Основные подходы:
    - Word2Vec (CBOW и Skip-gram)
    - GloVe (Global Vectors for Word Representation)
    - FastText (учитывает морфологию слов)
    - BERT, ELMo и другие контекстные эмбеддинги
  - Свойства векторных представлений:
    - Семантически близкие слова находятся рядом в векторном пространстве
    - Поддерживают векторную арифметику (король - мужчина + женщина ≈ королева)
    - Сохраняют аналогии и отношения между словами
  - Применение:
    - Входные данные для нейронных сетей
    - Анализ семантической близости
    - Машинный перевод
    - Системы вопросов и ответов
  - Преимущества перед one-hot encoding:
    - Более компактное представление
    - Учет семантических отношений
    - Возможность работы с неизвестными словами (OOV)
  - Инструменты:
    - Gensim, SpaCy, TensorFlow, PyTorch
    - Предобученные модели (Google News, Wikipedia)
- Тематическое моделирование
  - Определение: метод анализа текстов для выявления скрытых тематических структур в коллекции документов
  - Основные алгоритмы:
    - Латентное размещение Дирихле (LDA, Latent Dirichlet Allocation)
    - Вероятностное латентное семантическое индексирование (PLSI)
    - Неотрицательное матричное разложение (NMF)
    - Латентный семантический анализ (LSA/LSI)
  - Принцип работы LDA:
    - Каждый документ представляется как смесь тем
    - Каждая тема представляется как распределение вероятностей слов
    - Генеративная вероятностная модель с распределением Дирихле
  - Применение:
    - Классификация документов
    - Рекомендательные системы
    - Анализ трендов в текстовых данных
    - Извлечение тематической структуры из корпуса текстов
  - Оценка качества моделей:
    - Перплексия (perplexity)
    - Когерентность тем (topic coherence)
    - Экспертная оценка интерпретируемости тем
  - Инструменты:
    - Gensim
    - scikit-learn
    - BERTopic (для тематического моделирования с использованием BERT)

## 15. Практические аспекты
- Работа с pandas
- Обработка данных
- Валидация моделей
- Сравнение моделей
- Работа с большими данными
- Оптимизация производительности

## Рекомендации по подготовке:
1. Уделить особое внимание теории вероятностей и статистике
2. Практиковаться в интерпретации результатов моделей
3. Разобраться с различными метриками качества и их применением
4. Изучить методы предобработки данных
5. Потренироваться в подборе гиперпараметров
6. Разобраться с визуализацией данных и их интерпретацией
7. Изучить математические основы машинного обучения
8. Практиковаться в решении различных типов задач
9. Разобраться с особенностями работы с разными типами данных
10. Изучить методы оптимизации и регуляризации
