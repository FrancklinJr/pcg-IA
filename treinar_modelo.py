import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, matthews_corrcoef
import xgboost as xgb


def carregar_e_preparar_dados():
    df = pd.read_csv("dados_tuberculose.csv", sep=";", low_memory=False).head(30000)

    classes_usadas = ['Cura', 'Óbito por TB', 'Abandono', 'Óbito por outras causas', 'Transferência']
    df = df[df['SITUA_ENCE'].isin(classes_usadas)]

    df = df.rename(columns={
        "SITUA_ENCE": "STATUS_ENCERRAMENTO", "NU_IDADE_N": "IDADE", "CS_SEXO": "SEXO",
        "CS_ZONA": "ZONA", "AGRAVAIDS": "AGRAV_HIV", "AGRAVALCOO": "AGRAV_ALCOOLISMO",
        "AGRAVDIABE": "AGRAV_DIABETES", "AGRAVDROGA": "AGRAV_DROGAS", "AGRAVTABAC": "AGRAV_TABACO",
        "DT_INIC_TR": "INICIO_DO_TRATAMENTO", "DT_ENCERRA": "DATA_ENCERRAMENTO",
        "BACILOSC_1": "1_BACILOSCOPIA", "RAIOX_TORA": "RAIO-X", "FORMA": "TIPO"
    })

    df['INICIO_DO_TRATAMENTO'] = pd.to_datetime(df['INICIO_DO_TRATAMENTO'], format='%d/%m/%Y', errors='coerce')
    df['DATA_ENCERRAMENTO'] = pd.to_datetime(df['DATA_ENCERRAMENTO'], format='%d/%m/%Y', errors='coerce')
    df['DURACAO_TRATAMENTO'] = (df['DATA_ENCERRAMENTO'] - df['INICIO_DO_TRATAMENTO']).dt.days

    colunas_essenciais = [
        'IDADE', 'SEXO', 'ZONA', 'AGRAV_HIV', 'AGRAV_ALCOOLISMO', 'AGRAV_DIABETES',
        'AGRAV_DROGAS', 'AGRAV_TABACO', 'TIPO', 'RAIO-X', '1_BACILOSCOPIA',
        'DURACAO_TRATAMENTO', 'STATUS_ENCERRAMENTO'
    ]
    df = df.dropna(subset=colunas_essenciais)
    df = df[df['DURACAO_TRATAMENTO'] >= 0]

    for col in df.select_dtypes(include=['object']).columns:
        if col != 'STATUS_ENCERRAMENTO':
            df = df[~df[col].astype(str).str.contains('ignorado', case=False, na=False)]

    df['IDADE_FAIXA'] = pd.cut(df['IDADE'], bins=[0, 19, 39, 59, 79, 120],
                               labels=['0-19', '20-39', '40-59', '60-79', '80+'])
    df['TOTAL_AGRAVOS'] = df[['AGRAV_HIV', 'AGRAV_ALCOOLISMO', 'AGRAV_DIABETES',
                              'AGRAV_DROGAS', 'AGRAV_TABACO']].apply(
        lambda x: (x.astype(str).str.lower() == 'sim').sum(), axis=1)
    df['RAIO-X_BINARIO'] = np.where(
        df['RAIO-X'].astype(str).str.contains('normal', case=False, na=False), 'Normal', 'Anormal'
    )

    print(f"Linhas preservadas após limpeza: {len(df)}")
    return df


def treinar_modelo_binario(dados):
    print("\n" + "=" * 50)
    print("Treinando Modelo 1: Cura vs. Não Cura")
    print("=" * 50)

    dados['TARGET_BINARIO'] = np.where(dados['STATUS_ENCERRAMENTO'] == 'Cura', 1, 0)
    colunas_entrada = [
        'IDADE', 'IDADE_FAIXA', 'SEXO', 'ZONA', 'AGRAV_HIV', 'AGRAV_ALCOOLISMO',
        'AGRAV_DIABETES', 'AGRAV_DROGAS', 'AGRAV_TABACO', 'TIPO', 'RAIO-X_BINARIO',
        '1_BACILOSCOPIA', 'DURACAO_TRATAMENTO', 'TOTAL_AGRAVOS'
    ]
    X = pd.get_dummies(dados[colunas_entrada], drop_first=True)
    y = dados['TARGET_BINARIO']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    ratio = (y == 0).sum() / (y == 1).sum()

    modelo_final = xgb.XGBClassifier(
        random_state=42, eval_metric='logloss', tree_method='hist',
        colsample_bytree=1.0, gamma=0.2, learning_rate=0.01, max_depth=4,
        min_child_weight=1, n_estimators=200, subsample=1.0, scale_pos_weight=ratio
    )
    modelo_final.fit(X_train, y_train)
    y_pred = modelo_final.predict(X_test)

    print("\n--- RESULTADOS FINAIS (Cura vs Não Cura) ---")
    print(f"Acurácia: {accuracy_score(y_test, y_pred):.4f}")
    print(f"MCC: {matthews_corrcoef(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, target_names=['Não Cura', 'Cura']))
    joblib.dump({
        "modelo": modelo_final,
        "colunas": X.columns.tolist(),
        "X_test": X_test},
        "modelo/modelo_balanceado.pkl")

if __name__ == "__main__":
    dados = carregar_e_preparar_dados()
    treinar_modelo_binario(dados)
