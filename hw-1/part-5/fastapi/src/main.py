from fastapi import FastAPI, HTTPException
from typing import List
import dill as pickle
import pandas as pd
import logging

from model.item import Item, Items


# включаем журналирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI()

# Загружаем сохранённые модели
best_ridge = pickle.load(open("./ml/best_ridge.pkl", "rb"))
pipe_preprocess = pickle.load(open("./ml/pipe_preprocess2.pkl", "rb"))
ohe_task15 = pickle.load(open("./ml/ohe_task15.pkl", "rb"))


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    '''Предсказание цены автомобиля.

    Args:
        item (Item): объект для предсказания

    Returns:
        float: предсказанная цена
    '''
    logger.info(f"Получен объект для предсказания:\n {item}")
    try:
        # Преобразуем объект item в словарь и затем в DataFrame
        X = preprocess(item)
        # Делаем предсказание с помощью модели Ridge
        y = best_ridge.predict(X)
        logger.info(f"Предсказание: {y[0]}")
        return float(y[0])
    except Exception as e:
        logger.error(f"Ошибка во время предсказания: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_items")
def predict_items(items: Items) -> List[float]:
    '''Предсказание цены автомобилей.

    Args:
        items (Items): список объектов для предсказания

    Returns:
        List[float]: список предсказанных цен
    '''
    logger.info(f"Получен список объектов для предсказания:\n {items}")
    predictions = []

    for item in items.objects:
        try:
            # Преобразуем объект item в словарь и затем в DataFrame
            x_i = preprocess(item)
            # Делаем предсказание с помощью модели Ridge
            y_i = best_ridge.predict(x_i)
            predictions.append(float(y_i[0]))  # формируем результат
            logger.info(f"Предсказание: {y_i[0]}")
        except Exception as e:
            logger.error(f"Ошибка во время предсказания: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    logger.info(f"Предсказания: {predictions}")
    return predictions


def preprocess(item: Item) -> pd.DataFrame:
    '''Пайплайн предобработки для одного объекта.

    Args:
        item (Item): объект для предсказания

    Returns:
        pd.DataFrame: DataFrame с одной строкой и колонками, нужными модели
    '''
    # Преобразуем объект в словарь
    item_dict = item.model_dump()
    # Создаем DataFrame из словаря
    df = pd.DataFrame.from_records([item_dict])

    # Применяем предварительную обработку к признакам
    X_test = pipe_preprocess.transform(df)

    # Применяем one-hot-encoding к категориальным признакам
    X_test_cat_ohe = pd.DataFrame(
        ohe_task15.transform(X_test[ohe_task15.feature_names_in_]),
        columns=ohe_task15.get_feature_names_out(),
        index=X_test.index
    )

    # TODO: добавить в `pipe_preprocess` ColumnTransformer

    # Объединяем категориальные признаки с остальными
    X_test_ready = pd.concat([X_test,  X_test_cat_ohe], axis=1)
    X_test_ready = X_test_ready.drop(ohe_task15.feature_names_in_, axis=1)

    # Удаляем ненужные признаки
    X_test_ready.drop(columns=['name', 'selling_price'], inplace=True, axis=1)

    return X_test_ready  # Возвращаем предсказанное значение


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
