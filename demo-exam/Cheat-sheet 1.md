# Список тем для успешного прохождения теста:

1. **Визуализация данных:**
    * Hexagonal Bin Plot: назначение и интерпретация.
2. **Статистика:**
    * Нормальное распределение и его свойства.
    * Математическое ожидание случайной величины.
3. **Анализ данных:**
    * Зависимость между величинами (линейная и нелинейная).
4. **Линейная регрессия:**
    * Интерпретация коэффициентов линейной регрессии.
5. **Снижение размерности:**
    * Метод главных компонент (PCA): этапы реализации.
6. **Недообучение модели:**
    * Методы борьбы с недообучением (усложнение модели, добавление признаков).
7. **Логистическая регрессия:**
    * Решающая поверхность логистической регрессии.
8. **Решающие деревья:**
    * Гиперпараметры, влияющие на качество (max\_depth, min\_samples\_leaf).
9. **Случайный лес:**
    * Особенности алгоритма: обучение на подвыборках, использование части признаков, out-of-bag error.
10. **Кластеризация:**
    * Основные алгоритмы кластеризации (k-means, DBSCAN, Agglomerative clustering) и их особенности.
11. **Нейронные сети:**
    * Однослойная нейронная сеть и реализуемые ею логические функции (AND).
12. **Оценка вероятностей:**
    * Голосование (voting) как метод объединения предсказаний.
13. **Факторизационные машины:**
    * Понимание структуры модели и подсчет количества параметров.
14. **Работа с данными (pandas):**
    * Чтение данных из CSV.
    * Удаление строк с нечисловыми кодами.
    * Заполнение пропусков.
15. **Метрики классификации:**
    * Recall (полнота)

## Cheat-Sheet для изучения:

### 1. Визуализация данных

* **Hexagonal Bin Plot**: Визуализация двумерной плотности данных, альтернатива scatter plot при большом количестве точек.


### 2. Статистика

* **Нормальное распределение**: Симметричное распределение, описываемое средним значением и стандартным отклонением. Правило трех сигм.
* **Математическое ожидание**: Среднее значение случайной величины, взвешенное по вероятностям.  EX = ∑(xi * pi)


### 3. Анализ данных

* **Зависимость величин**: Величины могут быть зависимы, но не обязательно линейно зависимы. Важно проверять различные типы зависимостей.


### 4. Линейная регрессия

* **Интерпретация коэффициентов**: Коэффициент при признаке показывает изменение целевой переменной при изменении признака на единицу при фиксированных остальных признаках.


### 5. Снижение размерности

* **PCA (Метод главных компонент)**:

1. Центрирование данных.
2. Вычисление матрицы ковариации (или корреляции).
3. Вычисление собственных значений и собственных векторов.
4. Выбор главных компонент (векторов) по наибольшим собственным значениям.
5. Проецирование данных на выбранные компоненты.


### 6. Недообучение модели

* **Методы борьбы**:
    * Усложнение модели (увеличение количества параметров, использование более сложных алгоритмов).
    * Добавление новых признаков (полиномиальные признаки, взаимодействие признаков).


### 7. Логистическая регрессия

* **Решающая поверхность**: Гиперплоскость, разделяющая классы в пространстве признаков. Определяется коэффициентами модели.
* **Сигмоидная функция**: 	𝜎(𝑡) =1/(1+𝑒−𝑡)


### 8. Решающие деревья

* **Гиперпараметры**:
    * `max_depth`: Максимальная глубина дерева.
    * `min_samples_leaf`: Минимальное количество объектов в листе.
    * `min_samples_split`: минимальное количество объектов, необходимых для разделения внутреннего узла


### 9. Случайный лес

* **Особенности**:
    * Ансамбль решающих деревьев.
    * Обучение каждого дерева на случайной подвыборке данных (bootstrap).
    * Использование случайного подмножества признаков при построении каждого дерева.
    * Out-of-bag error для оценки качества модели без кросс-валидации.


### 10. Кластеризация

* **Алгоритмы**:
    * `k-means`: Разделение на k кластеров на основе расстояния до центроидов.
    * `DBSCAN`: Кластеризация на основе плотности данных.
    * `Agglomerative clustering`: Иерархическая кластеризация, объединение ближайших кластеров.


### 11. Нейронные сети

* **Однослойная нейронная сеть**: Может реализовывать простые логические функции, такие как AND, OR, NOT.


### 12. Оценка вероятностей

* **Голосование (Voting)**: Объединение предсказаний нескольких моделей для получения более стабильного и точного результата.


### 13. Факторизационные машины

* Модель для задач классификации и регрессии, основанная на разложении матрицы взаимодействий между признаками. Количество параметров зависит от количества признаков и размерности скрытых векторов.  number of parameters = number of linear terms + number of interaction terms + bias = n + n * (n - 1) / 2 * k + 1, где n - количество признаков, k - размерность скрытого вектора


### 14. Работа с данными (pandas)

* Основные операции: чтение данных (`pd.read_csv`), фильтрация (`df[df['column'] > value]`), обработка пропусков (`df.fillna(value)`).


### 15. Метрики классификации

* **Recall (полнота)**:  отношение верно классифицированных положительных объектов к общему числу положительных объектов.  Recall = TP / (TP + FN)

Этот cheat-sheet содержит ключевые понятия и методы, необходимые для успешного прохождения теста. Используй его для подготовки и удачи на экзамене!



