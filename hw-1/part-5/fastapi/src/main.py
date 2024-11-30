from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import dill as pickle  
import numpy as np
import pandas as pd
from ml.classes import DropColumns, FirstWordExtractor, FloatConverter, IntConverter, MedianImputer




# Определяем модель для одного объекта
class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float

# Определяем модель для списка объектов
class Items(BaseModel):
    objects: List[Item]

app = FastAPI()

# Загружаем сохранённые модели
# TODO: разобраться с основным пайплайном - не применился OHE...
best_ridge = pickle.load(open("./ml/best_ridge.pkl", "rb"))
pipe_preprocess = pickle.load(open("./ml/pipe_preprocess2.pkl", "rb"))
ohe_task15 = pickle.load(open("./ml/ohe_task15.pkl", "rb"))

@app.get("/")
async def top():
    return {"message": "Hello World"}
 
@app.post("/predict_item")
def predict_item(item: Item) -> float:
 
    # Преобразуем объект item в словарь и затем в DataFrame
    print("==============")
    print(item)
    print("==============")

    item_dict = item.model_dump()  # Преобразуем объект в словарь
    print(item_dict)
    print("==============")

    df = pd.DataFrame.from_records([item_dict])  # Создаем DataFrame из словаря
    print(df.head(1))
    print("==============")

    # Применяем предварительную обработку к признакам
    X_test = pipe_preprocess.transform(df)
    print(X_test.head(1))
    print("==============")

    X_test_cat_ohe = pd.DataFrame(
        ohe_task15.transform(X_test[ohe_task15.feature_names_in_]),
        columns=ohe_task15.get_feature_names_out(),
        index=X_test.index
    )
    print(X_test_cat_ohe.head(1))
    print("==============")

    X_test_ready = pd.concat([X_test,  X_test_cat_ohe], axis=1).drop(ohe_task15.feature_names_in_, axis=1)
    print(X_test_ready.head(1))
    print("==============")

    X_test_ready.drop(columns=['name', 'selling_price'], inplace=True, axis=1)
    print(X_test_ready.head(1))
    print("==============")
    
    # Делаем предсказание с помощью модели Ridge
    y_pred = best_ridge.predict(X_test_ready)
    print(y_pred)
    print("==============")
    
    return float(y_pred[0])  # Возвращаем предсказанное значение

@app.post("/predict_items")
def predict_items(items: Items) -> List[float]:
    predictions = []
    
    for item in items.objects:
        features = np.array([
            item.year,
            item.selling_price,
            item.km_driven,
            item.fuel,
            item.seller_type,
            item.transmission,
            item.owner,
            item.mileage,
            item.engine,
            item.max_power,
            item.torque,
            item.seats
        ]).reshape(1, -1)

        X_test = pipe_preprocess.transform(features)
        y_pred = best_ridge.predict(X_test)
        
        predictions.append(float(y_pred[0]))  # Добавляем предсказание в список
    
    return predictions

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)