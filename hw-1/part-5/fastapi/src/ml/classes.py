from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


class DropDuplicates(BaseEstimator, TransformerMixin):
    '''Трансформер для удаления дубликатов в пайплайне'''
    def __init__(self, subset=None, keep='first'):
        self.subset = subset
        self.keep = keep

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.subset:
            return X.drop_duplicates(subset=self.subset, keep=self.keep)
        else:
            return X.drop_duplicates(keep=self.keep)


class DropColumns(BaseEstimator, TransformerMixin):
    '''Трансформер для удаления столбцов в пайплайне'''
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        data = X.copy()  # Создаем копию, т.к. менять исходный датафрейм нельзя
        return data.drop(labels=self.columns, axis=1)


class FirstWordExtractor(BaseEstimator, TransformerMixin):
    '''Трансформер для выделение первого слова и создания нового столбца в пайплайне.
    
    Пример: 
        make_extractor = MakeExtractor(['name'], ['make'])
        создать столбец 'make' (производитель) из столбца 'name' (название автомобиля)
    '''
    def __init__(self, source_cols: list[str], target_cols=None):
        self.source_cols = source_cols
        self.target_cols = target_cols

        if len(self.source_cols) != len(self.target_cols):
            raise ValueError("Количесвто элементов в 'source_cols' и 'target_cols' не совпадает.")

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for source_col, target_col in zip(self.source_cols, self.target_cols):
            X[target_col] = X[source_col].str.split().str.get(0)
        return X


class FloatConverter(BaseEstimator, TransformerMixin):
    '''Трансформер для преобразования столбцов в числа в пайплайне'''
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
       for col in self.columns:
           try:
               X[col] = X[col].astype(float)
           except ValueError as e:
               print(f"Есть нечисловые значения в {col}: {e}. Заменены на NaN.")
               # после замены на NaN это можно заполнить медианами через SimpleImputer
               X[col] = pd.to_numeric(X[col], errors='coerce')
       return X


class IntConverter(BaseEstimator, TransformerMixin):
    '''Трансформер для преобразования столбцов в числа в пайплайне'''
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
       for col in self.columns:
           try:
               X[col] = X[col].astype(int)
           except ValueError as e:
               print(f"Есть нечисловые значения в {col}: {e}. Заменены на NaN.")
               # после замены на NaN это можно заполнить медианами через SimpleImputer
               X[col] = pd.to_numeric(X[col], downcast='integer',  errors='coerce')
       return X


class MedianImputer(BaseEstimator, TransformerMixin):
    '''Трансформер для заполнения медианой в пайплайне'''
    def __init__(self, columns):
        self.columns = columns
        self.medians = dict()
        self.is_fitted_ = False
        
    def fit(self, X, y=None):
        try:
            self.medians = X[self.columns].median().to_dict()
            self.is_fitted_ = True
        except KeyError as e:
            print(f"Не найдены столбцы {e} в датасете.")
        except ValueError as e:
            print(f"Невозможно вычислить медиану для столбца {e}.")
        return self

    def transform(self, X):
        if self.is_fitted_:
           for col in self.columns:
                X[col] = X[col].fillna(self.medians[col])
        return X
