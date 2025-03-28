## Data Culture Оценивание. Тестирования

**Машинное обучение. Демо-варианты**
Февраль - март 2025

### Демо-вариант

**Тест начат** Суббота, 8 марта 2025, 20:28

**Состояние** Завершены

**Завершен** Суббота, 8 марта 2025, 22:08

**Прошло времени** 1 ч. 40 мин.

**Кол-во правильных ответов** 6,71/10,04

**Оценка** 6,68 из 10,00 (67%)

---

**Вопрос 1**  ✅ Верно  
Баллов: 0,22 из 0,22

Hexagonal Bin Plot - альтернатива диаграмме рассеяния (scatter plot), решающая проблему визуализации, при которой в случае большого объема данных точки начинают перекрываться. Hexagonal Bin Plot визуализирует плотность распределения данных, а не сами объекты. Точки сгруппированы (binned) в шестиугольники, и распределение (число точек в шестиугольнике) отражено цветом шестиугольника. Впервые эта техника описана в 1987 (D.B.Carr et al. Scatterplot Matrix Techniques for large N, Journal of the American Statistical Association, No.389 pp 424-436).

Ниже изображен hexagonal bin plot для некоторого набора данных. Выберите два верных утверждения относительно этих данных, анализируя график.

*   ✅ Чаще всего в данных встречаются точки с координатами, лежащими в окрестности координаты (0, 0)
*   ❌ Чаще всего в данных встречаются точки с координатами, лежащими в окрестности координаты (0, 4)
*   ❌ В данных нет точек с координатой b, близкой к 3
*   ✅ Величины a и b зависимы
*   ❌ Величины a и b независимы

---

**Вопрос 2** ✅ Верно  
Баллов: 0,22 из 0,22

Известно распределение длины тела зайцев-беляков в см. Какие из перечисленных ниже утверждений верны, если распределение является нормальным со средним значением 63 и стандартным отклонением 2?

*   ❌ 50% зайцев-беляков имеют рост от 60 до 66 см.
*   ✅ Рост 68.2% зайцев-беляков находится между 61 и 65 см.
*   ❌ Рост всех зайцев беляков находится в диапазоне от 57 до 69 см.
*   ✅ 13.6% зайцев-беляков имеют рост от 59 до 61 см.

---

**Вопрос 3** ✅ Верно  
Баллов: 0,22 из 0,22

Случайная величина X задана своим законом распределения:

| 𝑥𝑖 | -2   | -1   | 5     | 7   |
|----| ---- | ---- | ----- | --- |
| 𝑝𝑖 | 2p   | 0.3  | 0.4   | p   |

Чему равны значение p и математическое ожидание данной случайной величины (EX)?

*   ✅ p = 0.1, EX = 2
*   ❌ p = 0.1, EX = 2.6
*   ❌ p = 0.3, EX = 9
*   ❌ p = 0.3, EX = 2.6

---

**Вопрос 4** 🔶 Частично правильный  
Баллов: 0,04 из 0,22

Объекты в данной задаче имеют один числовой признак x, а y - целевая переменная. Ниже изображена диаграмма рассеяния, визуализирующая данные.

Среди перечисленных ниже утверждений найдите верные:

*   ❌ Для решения задачи предсказания целевой переменной y по нашим данным не подходит модель линейной регрессии, так как в данных нет линейной зависимости y от x.
*   ❌ Величины x и y независимы.
*   ❌✅ Абсолютное значение коэффициента корреляции Пирсона между величинами x и y мало.
*   ✅ Величины x и y зависимы, но не линейно.
*   ❌ Для решения задачи предсказания целевой переменной y по нашим данным не подходит ни одна модель регрессии, так как в данных нет линейной зависимости y от x.

---

**Вопрос 5** ✅ Верно  
Баллов: 0,22 из 0,22

Для решения задачи прогнозирования зарплаты выпускника ВУЗа была применена линейная регрессия. Используемые признаки:

1.  $x_1$: возраст выпускника - от 20 до 25 лет.
2.  $x_2$: средний балл выпускника - от 4 до 10.
3.  $x_3$: число пересдач выпускника за весь период обучения - от 0 до 10.
4.  $x_4$: средняя зарплата родителей выпускника - от 20 до 500 тысяч рублей.
5.  $x_5$: закодированный бинарный фактор - 0 или 1.

Целевая переменная:  
y: Зарплата выпускника - от 20 до 500 тысяч рублей.

Линейная модель была обучена на имеющихся данных, получено уравнение:
$$
y = 50 + 20x_1 + 2x_2 - 3x_3 + 5x_4 + 30x_5
$$

Выберите все утверждения, верно интерпретирующие коэффициенты модели:

*   ✅ При увеличении среднего балла выпускника на 2 зарплата выпускника увеличивается в среднем на 4 тысячи рублей.
*   ❌ После масштабирования данных (приведение среднего значения каждого признака к 0 и стандартного отклонения к 1) некоторые значимые признаки перестанут быть значимыми (значимость признака здесь проверяется с помощью критерия Стьюдента).
*   ❌ Наименее важный признак (среди $x_1, \ldots , x_5$)в модели - это $x_3$, так как при нем стоит наименьший коэффициент
*   ✅ 30 тысяч рублей - средняя разница в зарплате между выпускниками с $x_5=1$ и выпускниками с $x_5=0$.

---

**Вопрос 6** 🔶 Частично правильный  
Баллов: 0,15 из 0,22

В нашей задаче 100 объектов. Каждый объект имеет 100 числовых признаков. Матрица объект-признак, содержащая данные - $X$.  
Мы хотим снизить размерность до 10 с помощью метода главных компонент.  
Выберите все шаги, которые необходимо проделать для решения этой задачи:

*   ✅ Выбрать 10 собственных векторов матрицы $X^TX$ с наибольшими собственными значениями и спроецировать исходные данные на эти векторы.
*   ❌ Нормировать данные (привести стандартные отклонения каждого признака к 1 или к одному масштабу)
*   ❌ Выбрать 10 собственных векторов матрицы $X$ с наибольшими собственными значениями и спроецировать исходные данные на эти векторы.
*   ✅ Вычислить собственные значения и собственные векторы матрицы $X^TX$
*   ✅ Центрировать данные (привести средние значения по каждому признаку к 0)
*   ❌ Вычислить собственные значения и собственные векторы матрицы $X$.

---

**Вопрос 7** ✅ Верно  
Баллов: 0,22 из 0,22

У вас есть фотографии рыб, обитающих в Тихом океане. Вам нужно по фотографии рыбы определить ее название. Все названия закодированы числами от 1 до 1000, выборка сбалансирована. Ваш алгоритм для каждой фотографии выдает одно число - код названия рыбы. Качество алгоритма вычисляется на тестовой выборке, состоящей из 10000 изображений.

Какие метрики качества нельзя использовать в данной задаче?

*   ✅ коэффициент детерминации
*   ❌ recall
*   ❌ precision
*   ❌ accuracy
*   ✅ DCG@1000

---

**Вопрос 8** ✅ Верно  
Баллов: 0,22 из 0,22

Рассмотрим линейную модель регрессии в задаче предсказания целевой переменной по двум признакам:
$$
a(x) = w_0 + w_1x_1 + w_2x_2
$$
Функция потерь имеет вид
$$
Q(w) = \sum_{i=1}^{l}|y_i -a(x_i)|
$$

Выяснилось, что модель недообучилась. Какие из перечисленных ниже подходов корректно описаны и их можно предпринять для снижения недообучения?

*   ✅ Вместо линейной регрессии будем использовать решающее дерево без ограничений на глубину, количество объектов в вершинах и количество объектов в листьях.
*   ❌ Заменим функцию потерь на MSE, так как используемая в задаче функция потерь не имеет производной в нуле
*   ❌ Уберём константный коэффициент 𝑤0, так как он увеличивает сложность модели и при этом не влияет на обобщающую способность модели
*   ✅ Добавим полиномиальных признаков второй степени, чтобы увеличить обобщающую способность модели

---

**Вопрос 9** 🔶 Частично правильный  
Баллов: 0,11 из 0,22

Дан следующий набор текстов:

1.  В лесу живут дикие звери.
2.  Дикий зверь напал на курицу.
3.  Курица - домашняя птица.
4.  Кошки бывают дикие и домашние.

После нескольких шагов обработки тексты приняли следующий вид:
```
1. [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  
2. [0, 0, 1, 1, 1, 1, 0, 0, 0, 0]  
3. [0, 0, 0, 0, 0, 1, 1, 1, 0, 0]  
4. [0, 0, 1, 0, 0, 0, 1, 0, 1, 1]
```

Какие преобразования произвели с текстами?

*   ✅ Bag of words кодирование
*   ✅ Лемматизация
*   ✅ Токенизация
*   ❌ Удаление всех частей речи, кроме существительных
*   ✅ Удаление частиц, предлогов, союзов, артиклей
*   ❌ Tf-idf кодирование

---

**Вопрос 10** 🔶 Частично правильный  
Баллов: 0,11 из 0,22

Вам нужно с помощью машинного обучения решить задачу выдачи рекомендаций: для каждого пользователя, посетившего страницу интернет-магазина сантехники, рекомендовать пять товаров для покупки (ваша рекомендация - это набор из пяти товаров, т.е. для каждого товара у вас есть предсказание: 1 - пользователю рекомендован товар, и 0 - не рекомендован). В обучающей выборке находятся только пользователи, купившие пять и более товаров.  
Какие метрики качества из перечисленных ниже можно использовать в данной задаче?

*   ❌ MAE
*   ✅ hitrate@5
*   ❌ DCG@5
*   ❌ nDCG@5
*   ✅ precision@5
*   ❌ MSE

---

**Вопрос 11** ❌ Неверно  
Баллов: 0,00 из 0,22

Решается задача бинарной классификации с классами {0, 1}. Алгоритм выдаёт некоторую оценку, принадлежащую отрезку $$0, 1], что объект относится к классу 1. Качество алгоритма PR-AUC = 0.15 (площадь под precision-recall кривой).  
Какой вывод можно сделать об алгоритме?

*   ❌ Алгоритм имеет хорошее качество: это можно увидеть, возведя в квадрат предсказанные значения.
*   ❌ По величине PR-AUC однозначных выводов сделать нельзя: необходимо вычислить также ROC-AUC и уже затем оценить результат.
*   ❌✅ Алгоритм имеет хорошее качество: это можно увидеть, поменяв предсказанные метки (0 и 1) местами.
*   ❌ Алгоритм имеет плохое качество, т.е. задача классификации решена плохо.

---

**Вопрос 12** ✅ Верно  
Баллов: 0,22 из 0,22

Формула для логистической регрессии, обученной на некотором датасете, имеет вид: 
$$
a(x) = (w_0 + w_1x_1 + w_2x_2), \text{ где } w_0=5,  w_1=0,  w_2=-2, \sigma(t)=\frac{1}{1+e^{-t}}  
$$
Какой из графиков верно отображает решающую поверхность и предсказания модели?

*   ✅ (Здесь должна быть картинка графика - горизонтальная линия на уровне $x_2=2.5$)

---

**Вопрос 13** 🔶 Частично правильный  
Баллов: 0,15 из 0,22

У нас есть данные для задачи бинарной классификации. Каждый объект описывается десятью признаками: $x_1, x_2, \ldots , x_{10}$ . Известна истинная зависимость целевой переменной от признаков: $y = I[x_3^2 + \frac{x_7^2}{16} \le 3]$, где $I[\ldots]$ - индикатор. Признаки $x_1, x_2, x_4, x_5, x_6, x_8, x_9, x_{10}$ - это шумы, никак не влияющие на целевую переменную. Мы хотим обучить решающее дерево (DecisionTreeClassifier из библиотеки sklearn) для решения этой задачи.  
Какие гиперпараметры решающего дерева будут сильно влиять на качество алгоритма (accuracy)? (“Сильно влиять” означает, что если значение гиперпараметра подобрано неверно, то accuracy значительно упадет).

*   ❌✅ max\_features
*   ✅ max\_depth
*   ❌ criterion
*   ✅ min\_samples\_leaf

---

**Вопрос 14** ✅ Верно  
Баллов: 0,22 из 0,22

Выберите все утверждения, относящиеся к случайному лесу:

*   ✅ Каждое дерево обучается на части объектов обучающей выборки
*   ✅ При использовании случайного леса нет необходимости в кросс-валидации или в отдельном тестовом наборе, так как получить несмещенную оценку ошибки можно с помощью out-of-bag error
*   ❌ Смещение случайного леса, состоящего из N деревьев, в N раз меньше смещения одного дерева в случае, если предсказания деревьев не коррелируют между собой
*   ❌ Деревья обучаются последовательно, и каждое следующее исправляет ошибку предыдущих деревьев
*   ✅ При построении дерева на каждом шаге используется только часть признаков

---

**Вопрос 15** ✅ Верно  
Баллов: 0,22 из 0,22

На рисунке изображены результаты работы трех алгоритмов кластеризации на различных наборах данных (первый столбец - алгоритм A, второй столбец - алгоритм B, третий столбец - алгоритм C):

Определите, какой алгоритм соответствует каждому из столбцов:

*   ✅ A - kmeans, B - dbscan, C - agglomerative clustering

---

**Вопрос 16** ✅ Верно  
Баллов: 0,22 из 0,22

Рассмотрим однослойную нейронную сеть, принимающую на вход два бинарных признака $x_1, x_2 \in \{−1, 1\}$, с функцией активации `sign` (знак полученной взвешенной суммы).  
Какую функцию булевой алгебры логики реализует эта нейронная сеть?

*   ❌ OR
*   ❌ XOR
*   ✅ AND
*   ❌ NAND

---

**Вопрос 17** ✅ Верно  
Баллов: 0,42 из 0,42

Алгоритм бинарной классификации, обученный на некотором наборе данных, был применён к тестовому набору из 500 объектов положительного и 500 объектов отрицательного класса. Получено значение метрики recall = 0.7. После улучшения алгоритма его заново применили к тестовому набору данных: теперь на 45 объектах тестовой выборки, где алгоритм ошибочно предсказывал отрицательный класс, предсказания поменялись на положительный класс. Остальные предсказания остались такими же.  
Чему стало равно значение метрики recall? (В качестве разделителя целой и дробной части используйте точку. Например, 0.25)  
Ответ: 0.79

---

**Вопрос 18** ✅ Верно  
Баллов: 0,42 из 0,42

В выборке 950 здоровых людей и 50 больных простудой. Переменная $y_i=1$ для здоровых людей и $y_i=0$ для больных простудой. Единственный признак 𝑥 равен 0, если температура тела человека > 37, и 1 иначе. В наших данных у всех наблюдаемых людей 𝑥= 0.  

| №   | x   | y   |
| ----- | ----- | ----- |
| 1   | 0   | 1   |
| 2   | 0   | 1   |
| ... | ... | ... |
| 950 | 0   | 1   |
| 951 | 0   | 0   |
| ... | ... | ... |
| 1000 | 0 | 0 |

Мы хотим, чтобы алгоритм классификации выдавал для каждого человека $𝑏_𝑖$ - оценку вероятности того, что человек болен простудой.  
Обучите алгоритм классификации с функцией потерь $L(y_i,b_i) = |y_i−b_i|$. В ответ запишите значение любого из полученных $b_i$.  
Ответ: 1

> Если все признаки константы, то минимизация MSE дает "среднее", а MAE - "медиану"

---

**Вопрос 19** ✅ Верно  
Баллов: 0,42 из 0,42

Вася хочет понять, будет завтра дождь или нет. Для этого он изучил 7 сайтов, предсказывающих погоду. Известно, что каждый сайт дает верные предсказания с вероятностью 0.8. Вася решил сделать вывод на основе голосования (voting), основанного на предсказаниях этих сайтов.  
С какой вероятностью Вася получит верный прогноз? Ответ выразите в долях единицы и округлите до сотых. (В качестве разделителя целой и дробной части используйте точку. Например, 0.25)  
Ответ: 0.97

> Распределение Бернули Bern(7, 0.8)
> P(x=k) = C_n^k * p^k * (1-p)^{n-k}
> P(x>=4) = P(4) + P(5) + P(6) + P(7)

---

**Вопрос 20** ✅ Верно  
Баллов: 0,42 из 0,42

В задаче построения рекомендаций хорошо показывают себя факторизационные машины. Рассмотрим факторизационную машину с прогнозом 
$$
a(x) = w_0 + \sum_{j=1}^{d}w_jx_j + \sum_{j=1}^{d}\sum_{k=j+1}^{d}\langle v_j, v_k \rangle x_jx_k
$$
где $x_1, \ldots , x_d$ - признаки объекта и $d=150$, $v_j$ - скрытые векторы размерности $r=10$.  
Сколько параметров имеет эта модель? В ответе укажите одно число.  
Про рекомендательные системы для решения этой задачи знать ничего не нужно. Нужна только данная вам формула.  
Ответ: 1651

---

**Вопрос 21** ✅ Верно  
Баллов: 0,42 из 0,42

Двадцать человек сидят по кругу и играют в игру: все одновременно называют произвольные натуральные числа.
Побеждают те, у кого четность названного числа отличается от четности чисел обоих соседей. Найдите
математическое ожидание числа победителей.
Ответ: 5

> Решается "методом индикаторов":
> $$
> I_i = \{1,\ если выиграл\ i-й\ человек\}
> $$
> $$
> E(\sum{I_i}) = \sum{E(I_i)} = 20 \cdot E(I_1)
> $$
> $$
> E(I_1) = P(вероятность\ основного\ события)
> $$
>   (это свойство индикаторов)  
> События: "Чет, Нечет, Чет" или "Нечет, Чет, Нечет"  
> $p=0.5 * 0.5 * 0.5 + 0.5 * 0.5 * 0.5 = 1/4$  
> $E = 20 * p = 5$

---

**Вопрос 22** ❌ Неверно  
Баллов: 0,00 из 0,42

У вас есть цветное изображение размера 128x128 пикселей, вы пропускаете его через сверточный слой нейронной сети, состоящий из X фильтров с ядром 5x5.  
Дополнительные параметры свёрточного слоя: stride = 2, zero padding.  
После пропуска изображения через заданный сверточный слой получился feature map, сумма всех размеров которого равна 130.  
Сколько фильтров (X) присутствует в сверточном слое сети?  
Ответ: 0

---

**Вопрос 23** ✅ Верно  
Баллов: 0,05 из 0,05

В файлах `X_train.csv` и `X_test.csv` находятся данные о результатах аудита некоторой компании для некоторых фирм.  
В файле `y_train.csv` находятся ответы для тренировочной выборки, 0 и 1: для надежной фирмы - 0, для фирмы-мошенника - 1.

В этом задании предлагается изучить представленные данные, а также выявить зависимость результатов аудита и ответов (фирма надежная или является мошенником).  
В файлах `X_train.csv` и `X_test.csv` находятся значения различных показателей, измеряемых аудиторами.

Далее за задания можно получить максимум 4 балла.  
Считайте данные в три pandas dataframe: `X_train`, `y_train` и `X_test`.  
В этом задании работаем только с тренировочными данными (`X_train`, `y_train`).  

Сколько различных значений находится в столбце `LOCATION_ID`?  
Ответ: 40

---

**Вопрос 24** ✅ Верно  
Баллов: 0,10 из 0,10

Удалите из столбца `LOCATION_ID` в таблице `X_train` все строки, содержащие нечисловые коды. Удалите соответствующие строки из `y_train`. В ответ запишите новое количество строк в таблице `X_train`.  
Ответ: 541

---

**Вопрос 25** ✅ Верно  
Баллов: 0,05 из 0,05

Есть ли пропуски в тренировочных данных? В ответе укажите количество строк, содержащих пропущенные значения.  
Ответ: 1

---

**Вопрос 26** ✅ Верно  
Баллов: 0,10 из 0,10

Если пропуски в тренировочных данных есть, то заполните их средним значением по соответствующим столбцам в случае, если признак числовой, и новой категорией "NEW", если столбец категориальный.  
Аналогичные действия осуществите для тестовых данных (если в некотором числовом столбце в тестовых данных есть пропуск, заполните его средним значением `X_train` по данному столбцу).  
В ответ запишите любое из средних значений (округленное до сотых; разделитель целой и дробной части - точка), которыми вы заполняете пропуски, или "NEW" (без кавычек), если вы использовали эту категорию для заполнения пропусков в каких-либо столбцах  
Ответ: 16.45

---

**Вопрос 27** ✅ Верно  
Баллов: 0,25 из 0,25

Закодируйте `LOCATION_ID` следующим образом: если некоторое значение `LOCATION_ID` встречается в `X_train`
меньше 10 раз, то замените его на 'A', если значение встречается от 10 до 19 раз - замените его на 'B', от 20 до 29 раз -
на 'C' и так далее по английскому алфавиту.
Примените полученную кодировку и к `X_train`, и к `X_test`.
Если в X_test встречается значение категории, которого нет в `X_train`, то его замените на 'A'.
Сколько различных категорий ('A','B' и т.д.) имеет преобразованный столбец `LOCATION_ID` в таблице `X_train`?
Убедитесь, что новые категории в `X_train` и `X_test` совпадают и по названиям, и по их количеству.

Ответ:5

---

> Это задание состоит из двух подзаданий: a) и b).
> Подзадание b) имеет 5 подпунктов равного веса

**Вопрос 28** ✅ Верно  
Баллов: 0,5 из 0,5

Найдите все пары числовых признаков (не используйте признак `LOCATION_ID`), имеющие между собой корреляцию
Пирсона, по модулю большую 0.9. В каждой паре признаков удалите тот, который имеет меньшую корреляцию с
таргетом. Вычисления делайте по тренировочным данным, а затем изменяйте и тренировочные, и тестовые данные.
Сколько признаков вы удалили?
Ответ: 1

---

**Вопрос 29** ✅ Верно  
Баллов: 0,14 из 0,14

Вычислите среднее значение PARA_A внутри каждого класса: y_train = 0 и y_train = 1. В ответ запишите
максимальное из двух полученных чисел, округленное до сотых; между целой и дробной частями - точка.
Ответ: 4,16

---

**Вопрос 30** ❌ Неверно 
Баллов: 0,0 из 0,14

Вычислите медиану столбца TOTAL среди объектов, для которых Money_Value < 10. Ответ округлите до сотых.
Ответ: 1,08

---

**Вопрос 31** ✅ Верно   
Баллов: 0,14 из 0,14

Верно ли, что для объектов с Risk = 1 и значением PARA_A > 0.5, значение Loss всегда равно 0? В ответ запишите "да"
или "нет" (без кавычек).
Ответ: нет

---

**Вопрос 32** ✅ Верно   
Баллов: 0,14 из 0,14

Верно ли, что для LOCATION_ID типа "A" среднее значение Risk больше, чем для среднего значения Risk, вычисленного
по всем остальным локациям одновременно?
Ответ: да

---

**Вопрос 33** ✅ Верно   
Баллов: 0,14 из 0,14

Среди объектов с Risk = 0 и numbers = 5.0 вычислите 75%-квантиль значений в столбце History.
> Комментарий:
> для вычисления квартилей дискретного распределения используйте интерполяцию большим значением (higher interpolation). Это означает, что если искомая квартиль лежит между двумя измерениями i и j, то значение квартили равно j. Ответ округлите до целого.

Ответ: 0

---

**Вопрос 34** ✅ Верно   
Баллов: 0,25 из 0,25

Закодируйте столбец LOCATION_ID в X_train и X_test с помощью one-hot encoding (количество новых числовых
столбцов должно быть равно количеству категорий в LOCATION_ID). Сколько признаков стало в задаче после
применения этой кодировки?
Ответ: 11

---

**Вопрос 35** ❌ Неверно   
Баллов: 0,00 из 0,50

В качестве модели возьмите алгоритм k ближайших соседей из sklearn (KNeighborsClassifier) с параметрами по
умолчанию. Посчитайте среднее accuracy алгоритма на кросс-валидации с тремя фолдами (используйте только
тренировочные данные). Ответ округлите до сотых.
> Комментарий:
> параметры по умолчанию метода предполагаются следующими (n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)

Ответ: 0.91

---

**Вопрос 36** ❌ Неверно   
Баллов: 0,00 из 0,25


Подберите количество соседей в алгоритме k ближайших соседей (n_neighbors), перебирая гиперпараметр от 2 до
20 с шагом 1 и используя перебор по сетке (GridSearchCV из библиотеки sklearn.model_selection) с тремя фолдами
и метрикой качества - accuracy. В ответ запишите наилучшее среди искомых значение n_neighbors.
> Комментарий:
> остальные гиперпараметры модели оставьте по умолчанию (weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)

Ответ: 5

---

**Вопрос 37** ❌ Неверно   
Баллов: 0,00 из 0,50

Добавьте в X_train и в X_test новый признак 'mult', равный произведению двух признаков с наибольшей по модулю
корреляцией с таргетом (корреляция считается по тренировочным данным).
На данных с новым признаком заново с помощью GridSearchCV подберите количество соседей в методе k
ближайших соседей, в ответ напишите наилучшее количество соседей (по метрике accuracy).
> Комментарий: 
> остальные гиперпараметры модели оставьте дефолтными (weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)

Ответ: 6

---

**Вопрос 38** ❌ Неверно   
Баллов: 0,00 из 0,75

Теперь вы можете использовать любую модель машинного обучения для решения задачи. Также можете делать
любую другую обработку признаков. Ваша задача - получить наилучшее качество по метрике accuracy.
Качество проверяется на тестовых данных.
- accuracy >= 0.75 - 0.25 балла
- accuracy >= 0.8 - 0.75 балла

Пример файла для отправки результатов: result.tx
