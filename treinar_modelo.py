import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
import numpy as np
import joblib

dados_tuberculose = pd.read_csv("dados_tuberculose.csv", sep=";", low_memory=False)
dados_tuberculose = dados_tuberculose.head(30000)
classes_usadas = ['Cura', 'Óbito por TB', 'Abandono', 'Óbito por outras causas', 'Transferência']
dados_tuberculose = dados_tuberculose[dados_tuberculose['SITUA_ENCE'].isin(classes_usadas)]
dados_tuberculose = dados_tuberculose.drop(columns=[
    "ID_UNIDADE", "CS_GESTANT", "CS_ESCOL_N", "ID_MN_RESI", "ID_RG_RESI",
    "AGRAVOUTRA", "DT_NOTIFIC", "CS_RACA", "TEST_SENSI", "TRATSUP_AT",
    "TEST_MOLEC", "ID_AGRAVO", "HIV", "TRATAMENTO", "POP_LIBER", "CULTURA_ES"
])

dados_tuberculose = dados_tuberculose.rename(columns={
    "NU_ANO":"ANO",
    "ID_MUNICIP":"MUNICIPIO",
    "ID_REGIONA":"REGIAO",
    "DT_DIAG":"DATA DIAGNOSTICO",
    "NU_IDADE_N":"IDADE",
    "CS_SEXO":"SEXO",
    "CS_ZONA":"ZONA",
    "AGRAVAIDS":"AGRAV HIV",
    "AGRAVALCOO":"AGRAV ALCOOLISMO",
    "AGRAVDIABE":"AGRAV DIABETES",
    "AGRAVDOENC":"AGRAV DOENCA",
    "AGRAVDROGA":"AGRAV DROGAS",
    "AGRAVTABAC":"AGRAV TABACO",
    "DT_INIC_TR":"INICIO DO TRATAMENTO",
    "SITUA_ENCE":"STATUS ENCERRAMENTO",
    "DT_ENCERRA":"DATA ENCERRAMENTO",
    "BACILOSC_1":"1º BACILOSCOPIA",
    "BACILOSC_2":"2º BACILOSCOPIA",
    "BACILOSC_3":"3º BACILOSCOPIA",
    "BACILOSC_4":"4º BACILOSCOPIA",
    "BACILOSC_5":"5º BACILOSCOPIA",
    "BACILOSC_6":"6º BACILOSCOPIA",
    "RAIOX_TORA":"RAIO-X",
    "FORMA":"TIPO"
})

dados_tuberculose.index.name = "ID"
dados_tuberculose = dados_tuberculose.dropna()
dados_tuberculose = dados_tuberculose[~dados_tuberculose.apply(lambda row: row.astype(str).str.contains('ignorado', case=False).any(), axis=1)]

cols_baciloscopia = ['1º BACILOSCOPIA', '2º BACILOSCOPIA', '3º BACILOSCOPIA', '4º BACILOSCOPIA', '5º BACILOSCOPIA', '6º BACILOSCOPIA']
dados_tuberculose['BACILOSCOPIA_NEGATIVA'] = dados_tuberculose[cols_baciloscopia].astype(str).apply(
    lambda row: sum(row.str.contains('Negativa', case=False, na=False)), axis=1)

df_cura = dados_tuberculose[dados_tuberculose['STATUS ENCERRAMENTO'] == 'Cura']
df_outros = dados_tuberculose[dados_tuberculose['STATUS ENCERRAMENTO'] != 'Cura']
df_cura_reduzido = resample(df_cura, replace=False, n_samples=500, random_state=42)
dados_balanceados = pd.concat([df_cura_reduzido, df_outros])
dados_balanceados = dados_balanceados.sample(frac=1, random_state=42).reset_index(drop=True)

colunas_entrada = [
    'IDADE', 'SEXO', 'ZONA',
    'AGRAV HIV', 'AGRAV DIABETES', 'AGRAV DROGAS',
    'AGRAV ALCOOLISMO', 'AGRAV TABACO',
    'TIPO', 'RAIO-X', '1º BACILOSCOPIA', 'BACILOSCOPIA_NEGATIVA'
]
X = dados_balanceados[colunas_entrada]
y = dados_balanceados['STATUS ENCERRAMENTO']

le_dict = {}
for col in X.columns:
    if X[col].dtype == 'object':
        le = LabelEncoder()
        X.loc[:, col] = le.fit_transform(X[col].astype(str))
        le_dict[col] = le

y_encoder = LabelEncoder()
y = y_encoder.fit_transform(y.astype(str))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)

print("Colunas utilizadas:", list(X.columns))
print("Acurácia:", accuracy_score(y_test, y_pred))
print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred, target_names=y_encoder.classes_))

joblib.dump({
    "modelo": modelo,
    "encoders": le_dict,
    "target_encoder": y_encoder,
    "X_test": X_test,
    "y_test": y_test,
    "y_pred": y_pred
}, "modelo/modelo_balanceado.pkl")
