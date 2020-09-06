# План работ
1. Подгружаем библиотеки и модули
2. Подгружаем таблицы, исправляем типы данных и производим кодировку, попутно добавляя новые фичи
3. Соединяем таблицы, заполняем пропуски
4. Смотрим на зависимости м/у фичами и убираем те, что имеют высокую корреляцию
5. Производим анализ причин почему могут уходить клиенты
6. Избавляемся от дисбаланса классов
7. Стандартизируем признаки
8. Обучаем модели на стоковых гиперпараметрах
9. Смотрим какие фичи более всего влияют на обучение моделей, их оставляем остальные удаляем
10. Заново обучаем модели на стоковых гиперпараметрах и выбираем лучшую
11. Подбираем гиперпараметры для лучшей модели
12. Оцениваем финальный результат

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, f1_score,recall_score,precision_score
from sklearn.metrics import confusion_matrix

from sklearn.metrics import SCORERS

from time import *

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', -1)

Подгружаем датасеты

contract = pd.read_csv('/datasets/final_provider/contract.csv')
personal = pd.read_csv('/datasets/final_provider/personal.csv')
internet = pd.read_csv('/datasets/final_provider/internet.csv')
phone = pd.read_csv('/datasets/final_provider/phone.csv')

contract.info()

contract -  надо изменить типы данных у дат BeginDate, EndDate и ежемесячных и суммарных платежей. Надо закодировать категориальные признаки  

personal.info()

personal -  надо закодировать категориальные признаки


internet.info()

internet -  надо закодировать категориальные признаки


phone.info()

***Проведем анализ данных, для каждой из таблиц, параллельно будем кодировать категориальные признаки, менять тип данных и создавать фичи***

***ENCODING***

Кодируем данные для всех таблиц, для интернета и телефона добавим по столбцу с единицами. Так, мы сможем понять, кто к чему был подключен.

*INTERNET*

encoder = OrdinalEncoder()

internet_col = (internet.columns)

internet_col

internet_o= pd.DataFrame(encoder.fit_transform(internet[internet_col]), columns = internet_col)
internet[['InternetService', 'OnlineSecurity', 'OnlineBackup',
       'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']] = internet_o[['InternetService', 'OnlineSecurity', 'OnlineBackup',
       'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']]

internet.insert(0, 'Internet_customer', 1)

internet.head(3)

<font color='blue'><u><b>КОММЕНТАРИЙ РЕВЬЮЕРА</b></u></font><br />
<font color='green'>
ловко!!!

*PHONE*

encoder = OrdinalEncoder()

phone['MultipleLines_1'] = phone.MultipleLines

phone_col = phone.columns

phone_o = pd.DataFrame(encoder.fit_transform(phone[phone_col]), columns = phone_col)
phone['MultipleLines'] = phone_o['MultipleLines']
phone = phone.drop('MultipleLines_1', axis = 1)

phone.insert(0, 'Phone_customer', 1)

phone.head(5)

<font color='blue'><u><b>КОММЕНТАРИЙ РЕВЬЮЕРА</b></u></font><br />
<font color='green'>
Ок

*CONTRACT*

#contract_ohe = pd.get_dummies(contract[['Type','PaymentMethod']], drop_first =True)

#contract = pd.concat([contract[['customerID','BeginDate','EndDate','PaperlessBilling','MonthlyCharges','TotalCharges']]] + [contract_ohe],sort=True, axis = 1)

#contract.head(3)

contract_col = ['PaperlessBilling','Type','PaymentMethod' ]

contract_o = pd.DataFrame(encoder.fit_transform(contract[contract_col]), columns = contract_col)
contract[contract_col] = contract_o[contract_col]

*PERSONAL*

personal_o = pd.DataFrame(encoder.fit_transform(personal[['gender', 'Partner', 'Dependents']]),
                          columns =[['gender',  'Partner', 'Dependents']] )
personal[['gender',  'Partner', 'Dependents']] = personal_o[['gender',  'Partner', 'Dependents']]

personal.head(3)

<font color='purple'>Создадим фичу, которая отвечает за наличие семьи</font>

personal['family'] = personal['Partner'] + personal['Dependents']

def family_member(x):
    if x >= 1:
        x = 1
    else:
        x = 0
    return x

personal['family'] =personal['family'].apply(lambda x: family_member(x))

***MERGING***

Соединяем таблицы

data = pd.merge(contract, personal , how = 'outer', on = 'customerID')
data = pd.merge(data, internet, how = 'outer', on = 'customerID')
data = pd.merge(data, phone, how = 'outer', on = 'customerID')

data.TotalCharges = pd.to_numeric(data.TotalCharges, errors = 'coerce')
data.SeniorCitizen = pd.to_numeric(data.SeniorCitizen, errors = 'coerce')

data.EndDate = pd.to_datetime(data.EndDate, errors = 'coerce')

data.BeginDate = pd.to_datetime(data.BeginDate)

***Features engireering***

Создаем столбец искомой переменной, ушел клинет или нет. Если в столбце EndDate стоит NaN, значит клиент еще не ушел, следовательно ставим 0. Для остальных ставим единицу.

<font color='blue'><u><b>КОММЕНТАРИЙ РЕВЬЮЕРА</b></u></font><br />
<font color='green'>
да, согласен)

def end_date(y):
    if y != 0:
        y = 1
    return y

data['target'] = data['EndDate'].fillna(0)
data['target'] = data['target'].apply(lambda x: end_date(x))

Добавим столбец difference, как разницу между расторжением и заключением договора в днях, для тех кто не разорвал договор поставим дату 2020-02-01

<font color='blue'><u><b>КОММЕНТАРИЙ РЕВЬЮЕРА</b></u></font><br />
<font color='green'>
длительность договора - хороший признак может быть)

end = pd.to_datetime('2020-02-01')
data.EndDate = data.EndDate.fillna(end)
data['difference'] = (data.EndDate - data.BeginDate).dt.days

data.info()

data[data.target == 0].difference.plot(kind = 'hist', bins = 50)

data[data.target == 1].difference.plot(kind = 'hist', bins = 50)

data.head(3)

Посмотрим на скорость оттока клиентов по месяцам

data['EndDate'].unique()

data_leaving_rate = data[data['target']==1].pivot_table( index ='EndDate', values = 'target', aggfunc = 'count').sort_index()

data_leaving_rate.head()

 Таблица обрезана на последнии 4 месяца, нельзя убедиться в увеличеннии числа ушедших клиентов.

<font color='blue'><u><b>КОММЕНТАРИЙ РЕВЬЮЕРА</b></u></font><br />
<font color='green'>
можно посмотреть, а как давно эти ушедшие были с нами...

***Работа с пропусками***

Заполним пропуски нулями

print(data.isnull().sum())

data.dropna(subset = ['TotalCharges'], inplace = True)

<font color='blue'><u><b>КОММЕНТАРИЙ РЕВЬЮЕРА</b></u></font><br />
<font color='orange'>
Тут, если рассуждать с точки зрения подхода в целом (предположим, что пропусков 2000..) то нужна аргументация: а почему нулями заполняется колонка? какие предпосылки к этому?

<font color='purple'>Тут я имел ввиду, что 0 будут означать отсутствие приобритенных опций у клиента будь то интернет или телефон. Для себя я понял, что так как в таблицах internet и phone данных меньше, то это значит что отсутсвующие id клиентов, как раз те, что не пользуются такими услугами, поэтому при обьединении таблиц нулями заполняются Nans где услуга не была приобритена. Плюс к этому в каждой из таблиц internet и phone я создал столбцы с единицами, чтобы при обьединении было понятно, какой id пользовался или нет услугой</font>

data = data.drop(['BeginDate','EndDate','customerID'], axis = 1)

<font color='blue'><u><b>КОММЕНТАРИЙ РЕВЬЮЕРА</b></u></font><br />
<font color='green'>
ДА!!!!!! здорово, очень здорово, что не собираешься использовать даты как факторы. Они бы накрутили метрику - это да. Но вот бизнесово - убили бы нашу модель.

data = data.fillna(0)

<font color='blue'><u><b>КОММЕНТАРИЙ РЕВЬЮЕРА</b></u></font><br />
<font color='green'>
если нет услуг - то ставим 0 - тут понятно.

#### Анализ аномалий

data.describe()

Предоставленные данные выглядят адекватно

***EDA***

Проведем анализ данных

data[data.target == 0]['MonthlyCharges'].describe()

data[data.target == 1]['MonthlyCharges'].describe()

В среднем те клиенты, которые ушли платили больше на 13 денежных единиц.

<font color='blue'><u><b>КОММЕНТАРИЙ РЕВЬЮЕРА</b></u></font><br />
<font color='green'>
на 13 денежных единиц... :)

fig, ax = plt.subplots(figsize = (3,3))
sns.countplot(x = 'Internet_customer', data = data[data['target']==1]).set_title('Ушeдшие клиенты')
fig, ax = plt.subplots(figsize = (3,3))
sns.countplot(x = 'Internet_customer', data = data[data['target']==0]).set_title('Активные клиенты');

fig, ax = plt.subplots(figsize = (3,3))
sns.countplot(x = 'Phone_customer', data = data[data['target']==1]).set_title('Ушeдшие клиенты')
fig, ax = plt.subplots(figsize = (3,3))
sns.countplot(x = 'Phone_customer', data = data[data['target']==0]).set_title('Активные клиенты');

Портрет типичного клиента который ушел: практически все пользовались интернетом. Можно предположить, что имеено неудовлетворенность при использовании интернета послужила причиной отвала клиентов. Одной из причин последнему может быть высокая стоимость.  Телефонная связь скорее всего не являлась причиной ухода.

<font color='blue'><u><b>КОММЕНТАРИЙ РЕВЬЮЕРА</b></u></font><br />
<font color='green'>
понятно)

Вопросы тим лиду:
- Как проверить имеет ли смысл добавлять фичу difference?
- Как обосновать выбор метрик оценки для модели?

<font color="blue">
1. Ммм. добавить и проверить. Если скор улучшиться значит имеет. Вообще LTV в услугах одна из ключевых метрик, поэтому скажу сразу - стоит. Но надо аккуратно, чтобы не допустить утечки целевого признака
2. Метрика для нас одна ROC-AUC. Минимальное значение 0.75 еще важно в отчёте вывести accuracy.    

Дальнеийший план:
- Разделить выборки на тест треин
- Избавиться от дисбаланса
- Нормализовать тренировочные данные
- Обучить большое колличесвто моделей на стандартных гиперпараметрах, выбрать ту что справляется лучше, по выбранным метрикам
- Насторить гиперпарамтры лучшей модели гридсерчем с кросс валидацией, получить финальный результат




<font color="blue"> Хороший план, давай его осуществлять

***MODELLING***

Попробуем оценить наличие внутренней корреляции между фичами, если последняя будет присутсвовать, то можно избавиться от одного из

pearsoncorr = data.corr(method='pearson')
fig, ax = plt.subplots(figsize = (20,20))
sns.heatmap(pearsoncorr, 
            xticklabels=pearsoncorr.columns,
            yticklabels=pearsoncorr.columns,
            cmap='RdBu_r',
            annot=True,
            linewidth=0.5)

<font color='purple'>Попробуем убрать фичу MonthlyCharges, так как она хорошо коррелируется с Total charges, возможно модель обучится лучше.

PS я создал фичу difference и она хорошо коррелируется с Type и Total charges, но я все равно ее использую в при обучении модели, потому что результат auc_roc резко падает если не использовать ее при обучении. </font>

data = data.drop(['MonthlyCharges'], axis =1)

*Splitting*

train, test = train_test_split(data, shuffle=False, test_size = 0.25)

train_features = train.drop('target', axis = 1)
train_target = train['target']

test_features = test.drop('target', axis = 1)
test_target = test['target']

*Оцениваем дисбаланс классов. Класс 1 присутствует в датасете в неполном мере, следовательно нужно учесть баланс классов.*

fig, ax = plt.subplots(figsize = (3,3))
sns.countplot(x = 'target', data = data);

Балансируем данные 2 способовами. Сначала сделаем апсамплинг целевеого признака и посмотрим как участя модели, затем тоже самое сделаем черем даунсамплинг.

<font color='blue'><u><b>КОММЕНТАРИЙ РЕВЬЮЕРА</b></u></font><br />
<font color='green'>
Ок

*Upsampling Training Data*

def upsampling(train_features, train_target):
    train_features_0 = train_features[train_target == 0]
    train_features_1 = train_features[train_target ==1]
    target_0 = train_target[train_target == 0]
    target_1 = train_target[train_target ==1]
    
    train_features_up = pd.concat([train_features_0] + int(round(len(train_features_0)/len(train_features_1))) *
                                  [train_features_1])
    train_target_up = pd.concat([target_0] + int(round(len(train_features_0)/len(train_features_1))) *
                                  [target_1])
    train_features_up, train_target_up = shuffle(train_features_up, train_target_up,random_state = 12345)
    return train_features_up, train_target_up

train_features_up, train_target_up = upsampling(train_features, train_target)

<font color='blue'><u><b>КОММЕНТАРИЙ РЕВЬЮЕРА</b></u></font><br />
<font color='green'>
балансируем на обучающей выборке - ОК.

train_features_up.shape, train_target_up.shape

fig, ax = plt.subplots(figsize = (3,3))
sns.countplot(train_target_up);

*Downsampling*

len(train_features[train_target == 1])/len(train_features[train_target == 0])

def downsampling(train_features, train_target):
    train_features_0 = train_features[train_target == 0]
    train_features_1 = train_features[train_target ==1]
    target_0 = train_target[train_target == 0]
    target_1 = train_target[train_target ==1]
    
    train_features_dw = pd.concat(
        [train_features_0.sample(frac=0.35, random_state=12345)] + [train_features_1])
    train_target_dw = pd.concat(
        [target_0.sample(frac=0.35, random_state=12345)] + [target_1])
    
    train_features_dw, train_target_dw = shuffle(train_features_dw, train_target_dw,random_state = 12345)
    return train_features_dw, train_target_dw

train_features_dw, train_target_dw = downsampling(train_features, train_target)

train_features_dw.shape, train_target_dw.shape

fig, ax = plt.subplots(figsize = (3,3))
sns.countplot(train_target_dw);

Переда тем как будем обучать модели, проведем норализацию или стандартизацию данных. 

*Cheking histogramms*

#sns.distplot(train_features_up['MonthlyCharges'])

sns.distplot(train_features_up['TotalCharges'])

#sns.distplot(train_features_up['difference'])

Изначально так как распределения признаков выглядят как негауссовые, данные были нормализованы, но при этом модели обучались, хуже чем при стандартизированных данных. Поэтому делаем стандартизацию.

*Normalizing*

normalizer = Normalizer()
train_features_up[['MonthlyCharges', 'TotalCharges','difference']] = normalizer.fit_transform(
    train_features_up[['MonthlyCharges', 'TotalCharges','difference']])

test_features[['MonthlyCharges', 'TotalCharges','difference']] = normalizer.transform(
    test_features[['MonthlyCharges', 'TotalCharges','difference']])

train_features_dw[['MonthlyCharges', 'TotalCharges','difference']] = normalizer.fit_transform(
    train_features_dw[['MonthlyCharges', 'TotalCharges','difference']])

*Standartizing*

test_features_up = test_features.copy()
test_features_dw = test_features.copy()

scaler = StandardScaler()
train_features_up[[ 'TotalCharges','difference']] = scaler.fit_transform(
    train_features_up[['TotalCharges','difference']])

test_features_up[['TotalCharges','difference']] = scaler.transform(
    test_features[[ 'TotalCharges','difference']])

train_features_dw[[ 'TotalCharges','difference']] = scaler.fit_transform(
    train_features_dw[['TotalCharges','difference']])

test_features_dw[['TotalCharges','difference']] = scaler.transform(
    test_features[['TotalCharges','difference']])

<font color='blue'><u><b>КОММЕНТАРИЙ РЕВЬЮЕРА</b></u></font><br />
<font color='green'>
понял.<br>

<font color='blue'><u><b>КОММЕНТАРИЙ РЕВЬЮЕРА</b></u></font><br />
<font color='orange'>
тут ещё один момент: проверить бы - существут ли у нас проблема мультиколлинеарности. КОгда количесивенные факторы силно коррелируют друг с другом. Например 'MonthlyCharges', 'TotalCharges'? может один из них имеет смысл и убрать?

*Choosing best model*

def best_model(model, train_features, train_target, test_features, test_target):
    result = {}
    
    start = time()
    model.fit(train_features, train_target)

    
    answers = model.predict(test_features)
    answers_proba = model.predict_proba(test_features)
    answers_proba_1 = answers_proba[:,1]
    end = time()
    
    result['accuracy_score'] = accuracy_score(test_target,answers)
    result['recall_score'] = recall_score(test_target,answers)
    result['precision_score'] = precision_score(test_target,answers)
    result['f1_score'] = f1_score(test_target,answers)
    result['roc_auc_score'] = roc_auc_score(test_target, answers_proba_1)
    result['time'] = end - start

    print('name of model {}'.format(model))
    return result

test_target_mean = pd.Series(0, index=range(len(test_target)))
def mean_model(test_mean, test_target):
    result = {}
    
    result['accuracy_score'] = accuracy_score(test_target,test_mean)
    result['recall_score'] = recall_score(test_target,test_mean)
    result['precision_score'] = precision_score(test_target,test_mean)
    result['f1_score'] = f1_score(test_target,test_mean)
    result['roc_auc_score'] = roc_auc_score(test_target, test_mean)

    print('test_target_mean')
    return result

Обучаем все модели, которые использовали в течении курса. Начнем с доунсамплиного датасета

<font color='blue'><u><b>КОММЕНТАРИЙ РЕВЬЮЕРА</b></u></font><br />
<font color='green'>
все?? мне страшно )

LR = LogisticRegression(random_state = 12345, solver = 'lbfgs')
result_LR = best_model(LR, train_features_dw, train_target_dw, test_features_dw, test_target)
DTC = DecisionTreeClassifier(random_state = 12345)
result_DTC = best_model(DTC, train_features_dw, train_target_dw, test_features_dw, test_target)
RFC = RandomForestClassifier(random_state = 12345, n_estimators = 100)
result_RFC = best_model(RFC, train_features_dw, train_target_dw, test_features_dw, test_target)
CBC = CatBoostClassifier(random_state = 12345, verbose=0)
result_CBC = best_model(CBC, train_features_dw, train_target_dw, test_features_dw, test_target)
XGBC = XGBClassifier(random_state = 12345)
result_XGBC = best_model (XGBC, train_features_dw, train_target_dw, test_features_dw, test_target)
lgbC = LGBMClassifier()
result_lgbC = best_model(lgbC, train_features_dw, train_target_dw, test_features_dw, test_target)
mean_target = mean_model(test_target_mean, test_target)

scores_dw = pd.DataFrame({'name_model':["LogisticRegression","DecisionTreeClassifier","RandomForestClassifier",
                                      "CatBoostClassifier", "XGBClassifier","LGBMClassifier","Test_filled_with_mean"] ,\
                    'accuracy_score' : [result_LR["accuracy_score"], result_DTC["accuracy_score"], result_RFC["accuracy_score"],
                                         result_CBC["accuracy_score"],result_XGBC["accuracy_score"], result_lgbC["accuracy_score"],mean_target["accuracy_score"]],\
                          'recall_score' : [result_LR["recall_score"], result_DTC["recall_score"], result_RFC["recall_score"],
                                         result_CBC["recall_score"], result_XGBC["recall_score"], result_lgbC["recall_score"], mean_target["recall_score"]],\
                          'precision_score' : [result_LR["precision_score"], result_DTC["precision_score"], result_RFC["precision_score"],
                                         result_CBC["precision_score"],result_XGBC["precision_score"], result_lgbC["precision_score"],mean_target["precision_score"]],\
                    'f1_score' : [result_LR["f1_score"], result_DTC["f1_score"], result_RFC["f1_score"],
                                         result_CBC["f1_score"],result_XGBC["f1_score"], result_lgbC["f1_score"],mean_target["f1_score"]],\
                    'roc_auc_score' : [result_LR["roc_auc_score"], result_DTC["roc_auc_score"], result_RFC["roc_auc_score"],
                                         result_CBC["roc_auc_score"],result_XGBC["roc_auc_score"], result_lgbC["roc_auc_score"],mean_target['roc_auc_score']],\
                    'execution_time' : [result_LR["time"], result_DTC["time"], result_RFC["time"],
                                         result_CBC["time"],result_XGBC["time"], result_lgbC["time"],0]
                       })               

scores_dw.sort_values(by = 'roc_auc_score', ascending = False)

На даунсамплином трейне лучший результат у CatBoostClassifier = 0.892606	

LR = LogisticRegression(random_state = 12345, solver = 'lbfgs')
result_LR = best_model(LR, train_features_up, train_target_up, test_features_up, test_target)
DTC = DecisionTreeClassifier(random_state = 12345)
result_DTC = best_model(DTC, train_features_up, train_target_up, test_features_up, test_target)
RFC = RandomForestClassifier(random_state = 12345, n_estimators = 100)
result_RFC = best_model(RFC, train_features_up, train_target_up, test_features_up, test_target)
CBC = CatBoostClassifier(random_state = 12345, verbose=0)
result_CBC = best_model(CBC, train_features_up, train_target_up, test_features_up, test_target)
CBC.get_all_params()
XGBC = XGBClassifier(random_state = 12345)
result_XGBC = best_model (XGBC, train_features_up, train_target_up, test_features_up, test_target)
lgbC = LGBMClassifier()
result_lgbC = best_model(lgbC, train_features_up, train_target_up, test_features_up, test_target)

scores_up = pd.DataFrame({'name_model':["LogisticRegression","DecisionTreeClassifier","RandomForestClassifier",
                                      "CatBoostClassifier", "XGBClassifier","LGBMClassifier","Test_filled_with_mean"] ,\
                    'accuracy_score' : [result_LR["accuracy_score"], result_DTC["accuracy_score"], result_RFC["accuracy_score"],
                                         result_CBC["accuracy_score"],result_XGBC["accuracy_score"], result_lgbC["accuracy_score"],mean_target["accuracy_score"]],\
                          'recall_score' : [result_LR["recall_score"], result_DTC["recall_score"], result_RFC["recall_score"],
                                         result_CBC["recall_score"], result_XGBC["recall_score"], result_lgbC["recall_score"], mean_target["recall_score"]],\
                          'precision_score' : [result_LR["precision_score"], result_DTC["precision_score"], result_RFC["precision_score"],
                                         result_CBC["precision_score"],result_XGBC["precision_score"], result_lgbC["precision_score"],mean_target["precision_score"]],\
                    'f1_score' : [result_LR["f1_score"], result_DTC["f1_score"], result_RFC["f1_score"],
                                         result_CBC["f1_score"],result_XGBC["f1_score"], result_lgbC["f1_score"],mean_target["f1_score"]],\
                    'roc_auc_score' : [result_LR["roc_auc_score"], result_DTC["roc_auc_score"], result_RFC["roc_auc_score"],
                                         result_CBC["roc_auc_score"],result_XGBC["roc_auc_score"], result_lgbC["roc_auc_score"],mean_target['roc_auc_score']],\
                    'execution_time' : [result_LR["time"], result_DTC["time"], result_RFC["time"],
                                         result_CBC["time"],result_XGBC["time"], result_lgbC["time"],0]
                         })

scores_up.sort_values(by = 'roc_auc_score', ascending = False)

*На апсамплиноговом трейне без тюнинга гиперпараметров лучший результат auc_score = 0.908705 у LGBMClassifier, чуть уступает CatBoostClassifier*

<font color='purple'>*Проверим важность признаков на которых обучилась модель LGBMClassifier изначально для апскейлиного датасета*</font>

lgbC = LGBMClassifier()
lgbC.fit(train_features_up, train_target_up)    
imp = lgbC.feature_importances_
print(imp)

d = {'feature': ['Type', 'PaperlessBilling', 'PaymentMethod', 'TotalCharges', 'gender',
       'SeniorCitizen', 'Partner', 'Dependents','Family', 'Internet_customer',
       'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
       'TechSupport', 'StreamingTV', 'StreamingMovies', 'Phone_customer',
       'MultipleLines', 'difference'], 'importance': imp}

data_f = pd.DataFrame(data = d).sort_values(by = 'importance', ascending = False)
data_f = data_f.set_index('feature')
data_f.plot(kind = 'bar');
plt.ylabel('Feature Importance Score');
plt.xlabel('Feature');

<font color='purple'>*Видно, что основной вклад делают две фичи, difference и Total_charges. Далее я пользуюсь твоим советом и изхожу из такой логики, так есть две фичи которые играют самую важную роль при обучении, я оставляю только их для обучения модели, а остальные фичи удаляю, далее смотрю как качественно обучилась модель
*</font>

train_features_up = train_features_up[['difference', 'TotalCharges','PaymentMethod']]
test_features_up = test_features_up[['difference','TotalCharges','PaymentMethod']]

LR = LogisticRegression(random_state = 12345, solver = 'lbfgs')
result_LR = best_model(LR, train_features_up, train_target_up, test_features_up, test_target)
DTC = DecisionTreeClassifier(random_state = 12345)
result_DTC = best_model(DTC, train_features_up, train_target_up, test_features_up, test_target)
RFC = RandomForestClassifier(random_state = 12345, n_estimators = 100)
result_RFC = best_model(RFC, train_features_up, train_target_up, test_features_up, test_target)
CBC = CatBoostClassifier(random_state = 12345, verbose=0)
result_CBC = best_model(CBC, train_features_up, train_target_up, test_features_up, test_target)
CBC.get_all_params()
XGBC = XGBClassifier(random_state = 12345)
result_XGBC = best_model (XGBC, train_features_up, train_target_up, test_features_up, test_target)
lgbC = LGBMClassifier()
result_lgbC = best_model(lgbC, train_features_up, train_target_up, test_features_up, test_target)

scores_up = pd.DataFrame({'name_model':["LogisticRegression","DecisionTreeClassifier","RandomForestClassifier",
                                      "CatBoostClassifier", "XGBClassifier","LGBMClassifier","Test_filled_with_mean"] ,\
                    'accuracy_score' : [result_LR["accuracy_score"], result_DTC["accuracy_score"], result_RFC["accuracy_score"],
                                         result_CBC["accuracy_score"],result_XGBC["accuracy_score"], result_lgbC["accuracy_score"],mean_target["accuracy_score"]],\
                          'recall_score' : [result_LR["recall_score"], result_DTC["recall_score"], result_RFC["recall_score"],
                                         result_CBC["recall_score"], result_XGBC["recall_score"], result_lgbC["recall_score"], mean_target["recall_score"]],\
                          'precision_score' : [result_LR["precision_score"], result_DTC["precision_score"], result_RFC["precision_score"],
                                         result_CBC["precision_score"],result_XGBC["precision_score"], result_lgbC["precision_score"],mean_target["precision_score"]],\
                    'f1_score' : [result_LR["f1_score"], result_DTC["f1_score"], result_RFC["f1_score"],
                                         result_CBC["f1_score"],result_XGBC["f1_score"], result_lgbC["f1_score"],mean_target["f1_score"]],\
                    'roc_auc_score' : [result_LR["roc_auc_score"], result_DTC["roc_auc_score"], result_RFC["roc_auc_score"],
                                         result_CBC["roc_auc_score"],result_XGBC["roc_auc_score"], result_lgbC["roc_auc_score"],mean_target['roc_auc_score']],\
                    'execution_time' : [result_LR["time"], result_DTC["time"], result_RFC["time"],
                                         result_CBC["time"],result_XGBC["time"], result_lgbC["time"],0]      
                         })

scores_up.sort_values(by = 'roc_auc_score', ascending = False)

<font color='purple'>**Дальнийший план**

1. Оставляем только'difference', 'TotalCharges', так как имея только эти две фичи все модели обучилась лучше
2. CatBoostClassifier  показал лучший результат, но я не буду пытаться улучшить его тюнингом, тк как jupyter постоянно падает при долгом обучении модели.
3. Следовательно я попытаюсь поднять метрику у  LGBMClassifier, который по времени в 100 раз быстрее обучается и показал 2ой по результат по метрике roc_auc.
4. **(тут ход конем)** Если LGBMClassifier после тюнинга дает лучший результат чем CatBoostClassifier, то пускаю его в продакшен, если не дает , то пускаю в продакшн CatBoostClassifier на стоковых гиперпараметрах</font>

Выделяем основные гиперпараметры и будем варьировать каждый из них на гридсерче, а затем выбирать лучший. В конце соединим лучшию комбинацию параметров и обучим модель заново.

grid = {}
def hp_model(model, train_features, train_target, grid):
    start = time()
    grid_search =  GridSearchCV(estimator = model, param_grid  = grid,
                            scoring = 'roc_auc' ,cv = 5, verbose = 0)

    grid_fit = grid_search.fit(train_features,train_target)
    best = grid_search.best_params_   
    return best

LGBMClassifier(

               boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
               importance_type='split', learning_rate=0.1, max_depth=-1,
               min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,
               n_estimators=100, n_jobs=-1, num_leaves=31, objective=None,
               random_state=None, reg_alpha=0.0, reg_lambda=0.0, silent=True,
               subsample=1.0, subsample_for_bin=200000, subsample_freq=0
  
                       )

Варьируем max_depth

md = [int((x)) for x in np.linspace(start = 20, stop = 40, num = 11)]
random_grid = {'max_depth':md
              }
LGBMC = LGBMClassifier(random_state =12345)
best_md =  hp_model(LGBMC, train_features_up, train_target_up, grid = random_grid)

best_md

Варьируем n_estimators

ne = [int(x) for x in np.linspace(start = 100, stop = 1300, num = 49)]
ne = ne[1:]
random_grid = {'n_estimators':ne
              }
LGBMC = LGBMClassifier(random_state =12345,max_depth = best_md.get('max_depth'))
best_ne =  hp_model(LGBMC, train_features_up, train_target_up, grid = random_grid)

best_ne

Варьируем learning_rate

lr = [round(float(x),2) for x in np.linspace(start = 0.01, stop = 0.1, num = 10)]
#lr = lr[1:]
random_grid = {'learning_rate':lr
              }
LGBMC = LGBMClassifier(random_state =12345,max_depth = best_md.get('max_depth'),
                       n_estimators = best_ne.get('n_estimators'))
best_lr =  hp_model(LGBMC, train_features_up, train_target_up, grid = random_grid)

best_lr

Варьируем num_leaves

random_grid = {
    'num_leaves':[25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40]}
LGBMC = LGBMClassifier(random_state =12345,max_depth = best_md.get('max_depth'),
                       n_estimators = best_ne.get('n_estimators'),
                       learning_rate = best_lr.get('learning_rate'))
best_nl =  hp_model(LGBMC, train_features_up, train_target_up, grid = random_grid)

best_nl

Варьируем subsample_for_bin

bins = [int(x) for x in np.linspace(start = 200000, stop = 1000000, num = 10)]
random_grid = {
    'subsample_for_bin':bins}
LGBMC = LGBMClassifier(random_state =12345,max_depth = best_md.get('max_depth'),
                       n_estimators = best_ne.get('n_estimators'),
                       learning_rate = best_lr.get('learning_rate'),
                       num_leaves = best_nl.get('num_leaves'))
best_bins =  hp_model(LGBMC, train_features_up, train_target_up, grid = random_grid)

best_bins

Варьируем boosting_type

random_grid = {
    'boosting_type':['dart','gbdt','goss']}
LGBMC = LGBMClassifier(random_state =12345,max_depth = best_md.get('max_depth'),
                       n_estimators = best_ne.get('n_estimators'),
                       learning_rate = best_lr.get('learning_rate'),
                       num_leaves = best_nl.get('num_leaves'),
                       subsample_for_bin = best_bins.get('subsample_for_bin'))
best_bt =  hp_model(LGBMC, train_features_up, train_target_up, grid = random_grid)

best_bt

def final_model(model, train_features, train_target, test_features, test_target):
    result = {}
    
    start = time()
    model.fit(train_features, train_target)

    
    answers = model.predict(test_features)
    answers_proba = model.predict_proba(test_features)
    answers_proba_1 = answers_proba[:,1]
    
    end = time()
    
    result['accuracy_score'] = accuracy_score(test_target,answers)
    result['recall_score'] = recall_score(test_target,answers)
    result['precision_score'] = precision_score(test_target,answers)
    result['f1_score'] = f1_score(test_target,answers)
    result['roc_auc_score'] = roc_auc_score(test_target, answers_proba_1)
    result['time'] = end - start

    print('name of model {}'.format(model))
    return answers,result

Обучаем на лучшей комбинации гиперпараметров

lgbC = LGBMClassifier(random_state =12345,max_depth = best_md.get('max_depth'),
                       n_estimators = best_ne.get('n_estimators'),
                       learning_rate = best_lr.get('learning_rate'),
                       num_leaves = best_nl.get('num_leaves'),
                      subsample_for_bin = best_bins.get('subsample_for_bin'),
                      boosting_type = best_bt.get('boosting_type'))
answers, result_lgbC = final_model(lgbC, train_features_up, train_target_up, test_features_up, test_target)

tn, fp, fn, tp = confusion_matrix(test_target, answers).ravel()

(tn, fp, fn, tp)

scores_final = pd.DataFrame({'name_model':[ "LGBMClassifier"] ,\
                        'accuracy_score' : [result_lgbC['accuracy_score']] ,\
                        'recall_score' : [result_lgbC['recall_score']] ,\
                        'precision_score' : [result_lgbC['precision_score']] ,\
                        'f1_score' : [result_lgbC['f1_score']],\
                        'roc_auc_score' : [result_lgbC["roc_auc_score"]],\
                        'execution_time': [result_lgbC['time']]
                            })  

scores_final

scores_up.sort_values(by = 'roc_auc_score', ascending = False).head(1)

# Вывод

**Получили, что LGBMClassifier после тюнинга стал хуже, чем был до него и естественно проиграл СatBoostClassifierу. Поэтому пускаем в продакшн СatBoostClassifier на стоковых параметрах**

{'nan_mode': 'Min',
 'eval_metric': 'Logloss',
 'iterations': 1000,
 'sampling_frequency': 'PerTree',
 'fold_permutation_block': 0,
 'leaf_estimation_method': 'Newton',
 'boosting_type': 'Plain',
 'feature_border_type': 'GreedyLogSum',
 'bayesian_matrix_reg': 0.1000000015,
 'l2_leaf_reg': 3,
 'random_strength': 1,
 'rsm': 1,
 'boost_from_average': False,
 'model_size_reg': 0.5,
 'approx_on_full_history': False,
 'subsample': 0.8000000119,
 'use_best_model': False,
 'class_names': ['0', '1'],
 'random_seed': 12345,
 'depth': 6,
 'has_time': False,
 'fold_len_multiplier': 2,
 'border_count': 254,
 'classes_count': 0,
 'sparse_features_conflict_fraction': 0,
 'leaf_estimation_backtracking': 'AnyImprovement',
 'best_model_min_trees': 1,
 'model_shrink_rate': 0,
 'loss_function': 'Logloss',
 'learning_rate': 0.03022100031,
 'score_function': 'Cosine',
 'task_type': 'CPU',
 'leaf_estimation_iterations': 10,
 'bootstrap_type': 'MVS',
 'permutation_count': 4}

По итогу roc_auc = 0.916936 для ***СatBoostClassifier и именно его я предлагаю пустить в продакшн.***
К сожалению не получилось оттюнить ***LGBMClassifier***. Скорее всего это связано с тем, что стоковые параметры моделей и так настроены на максимальный результат изначально. В целом, результат модели хороший > 0.9. Также важно что **recall_score для СatBoostClassifier = 0.786008**, а это значит что на 100 собирающихся уйти клиентов, моя модель сможет поймать 78 человек, ну а далее менеджеры могут начать придумывать ходы как их оставить.
