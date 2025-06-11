import joblib
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, pointbiserialr
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import classification_report

st.set_page_config(layout="wide")

colunas = st.sidebar.selectbox("Escolha sua visualização", ["TABELA BRUTA", "TABELA LIMPA", "INDÍCE DE RISCO", "Previsão com Machine Learning (ML)"])

if colunas == "TABELA BRUTA":
    dados_brutos = pd.read_csv("dados_tuberculose.csv", sep=";")
    st.header("Tabela DataSet com dados brutos")
    st.dataframe(dados_brutos)

    st.header("🧾 LEGENDA")
    st.markdown("""
    | Variável         | Descrição                                              | Variável         | Descrição                                              |
    |------------------|--------------------------------------------------------|------------------|--------------------------------------------------------|
    | **ID_AGRAVO**     | ID do Agravo                                           | **DT_NOTIFIC**    | Data de Notificação                                    |
    | **NU_ANO**        | Ano de Notificação                                     | **ID_MUNICIP**    | Município de Notificação                               |
    | **ID_REGIONA**    | Regional de Notificação                                | **ID_UNIDADE**    | Unidade de Notificação                                 |
    | **DT_DIAG**       | Data de Diagnóstico                                    | **NU_IDADE_N**    | Idade                                                  |
    | **CS_SEXO**       | Sexo                                                   | **CS_GESTANT**    | Gestante                                               |
    | **CS_RACA**       | Raça/Cor                                               | **CS_ESCOL_N**    | Escolaridade                                           |
    | **ID_MN_RESI**    | Município de Residência                                | **ID_RG_RESI**    | Regional de Residência                                 |
    | **CS_ZONA**       | Zona de residência do paciente                         | **AGRAVAIDS**     | Doenças/Agravos associados: Aids                       |
    | **AGRAVALCOO**    | Doenças/Agravos associados: Alcoolismo                | **AGRAVDIABE**    | Doenças/Agravos associados: Diabetes                   |
    | **AGRAVDOENC**    | Doenças/Agravos associados: Doença Mental              | **AGRAVOUTRA**    | Doenças/Agravos associados: Outras                     |
    | **AGRAVDROGA**    | Doenças/Agravos associados: Drogas ilícitas            | **AGRAVTABAC**    | Doenças/Agravos associados: Tabagismo                  |
    | **TRATAMENTO**    | Tipo de Entrada                                        | **CULTURA_ES**    | Resultado da cultura de escarro                        |
    | **HIV**           | Resultado da sorologia para HIV                        | **HISTOPATOL**    | Resultado do exame histopatológico                     |
    | **DT_INIC_TR**    | Data de início do tratamento                           | **BACILOSC_1**    | Baciloscopia – 1º mês                                  |
    | **BACILOSC_2**    | Baciloscopia – 2º mês                                  | **BACILOSC_3**    | Baciloscopia – 3º mês                                  |
    | **BACILOSC_4**    | Baciloscopia – 4º mês                                  | **BACILOSC_5**    | Baciloscopia – 5º mês                                  |
    | **BACILOSC_6**    | Baciloscopia – 6º mês                                  | **TRATSUP_AT**    | Tratamento Diretamente Observado (TDO)                |
    | **SITUA_ENCE**    | Situação de encerramento                               | **DT_ENCERRA**    | Data de encerramento                                   |
    | **POP_LIBER**     | População privada de liberdade                         | **TEST_MOLEC**    | Teste Molecular Rápido para TB                         |
    | **TEST_SENSI**    | Teste de Sensibilidade                                 | **RAIOX_TORA**    | Radiografia do Tórax                                   |
    | **FORMA**         | Forma clínica da tuberculose                           |                  |                                                        |
    """)

elif colunas == "TABELA LIMPA":
    dados_tuberculose = pd.read_csv("dados_tuberculose.csv", sep=";")
    dados_tuberculose = dados_tuberculose.drop(columns=["ID_UNIDADE","CS_GESTANT","CS_ESCOL_N","ID_MN_RESI","ID_RG_RESI","HIV","AGRAVOUTRA","CULTURA_ES","DT_NOTIFIC",
                                                        "CS_RACA","TRATAMENTO","POP_LIBER","TEST_SENSI","TRATSUP_AT","TEST_MOLEC","ID_AGRAVO"])

    dados_tuberculose = dados_tuberculose.rename(columns={
        "ID_AGRAVO":"ID",
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
    dados_tuberculose['BACILOSCOPIA_NEGATIVA'] = dados_tuberculose[['1º BACILOSCOPIA', '2º BACILOSCOPIA', '3º BACILOSCOPIA', '4º BACILOSCOPIA', '5º BACILOSCOPIA', '6º BACILOSCOPIA']].apply(lambda row: sum(row.str.contains('Negativa', case=False)), axis=1)

    st.title("Tabela DataSet limpo")
    st.write("Essa tabela foi feita a partir de Machine Learning, com o objetivo de prever casos de tuberculose e seus agravantes.")
    st.dataframe(dados_tuberculose, use_container_width=True)

    st.header("📊 Análise de Correlação entre Variáveis Numéricas")
    
    numeric_cols = dados_tuberculose.select_dtypes(include=['int64', 'float64']).columns
    
    if len(numeric_cols) > 1:
        corr_matrix = dados_tuberculose[numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(corr_matrix, 
                   annot=True, 
                   fmt=".2f", 
                   cmap='coolwarm',
                   vmin=-1, 
                   vmax=1,
                   linewidths=0.5,
                   ax=ax)
        plt.title("Matriz de Correlação (Pearson)")
        st.pyplot(fig)
        
        st.subheader("Exemplo: Correlação entre Idade e Baciloscopias Negativas")
        r, p_value = pearsonr(dados_tuberculose['IDADE'], dados_tuberculose['BACILOSCOPIA_NEGATIVA'])
        st.write(f"""
        - **Coeficiente (r):** {r:.3f}  
        - **p-valor:** {p_value:.4f}
        """)
        
        if abs(r) > 0.5:
            st.success("✅ Correlação forte significativa (p < 0.05)")
        elif abs(r) > 0.3:
            st.warning("⚠️ Correlação moderada")
        else:
            st.info("ℹ️ Baixa ou nenhuma correlação linear")
        
        st.markdown("""
        ### Interpretação:
        - **Valores próximos de 1**: Correlação positiva forte
        - **Valores próximos de -1**: Correlação negativa forte
        - **Valores próximos de 0**: Sem correlação linear
        """)
    else:
        st.warning("Não há variáveis numéricas suficientes para análise de correlação.")

    st.header("🧾 LEGENDA")
    st.markdown("""
    | Variável                  | Descrição                                               |
    |---------------------------|---------------------------------------------------------|
    | **ID**                    | ID do Agravo                                            |
    | **ANO**                   | Ano de Notificação                                      |
    | **MUNICIPIO**             | Município de Notificação                                |
    | **REGIAO**                | Regional de Notificação                                 |
    | **DATA DIAGNOSTICO**      | Data de Diagnóstico                                     |
    | **IDADE**                 | Idade                                                   |
    | **SEXO**                  | Sexo                                                    |
    | **ZONA**                  | Zona de residência do paciente                          |
    | **AGRAV HIV**             | Doenças e agravos associados à Aids                    |
    | **AGRAV ALCOOLISMO**      | Doenças e agravos associados ao Alcoolismo             |
    | **AGRAV DIABETES**        | Doenças e agravos associados ao Diabetes               |
    | **AGRAV DOENCA**          | Doenças e agravos associados à Doença Mental           |
    | **AGRAV DROGAS**          | Doenças e agravos associados ao uso de drogas ilícitas |
    | **AGRAV TABACO**          | Doenças e agravos associados ao Tabagismo              |
    | **INICIO DO TRATAMENTO**  | Data em que o paciente iniciou o tratamento atual       |
    | **STATUS ENCERRAMENTO**   | Situação de encerramento                                |
    | **DATA ENCERRAMENTO**     | Data de encerramento                                    |
    | **1º BACILOSCOPIA**       | Baciloscopia no 1º mês                                  |
    | **2º BACILOSCOPIA**       | Baciloscopia no 2º mês                                  |
    | **3º BACILOSCOPIA**       | Baciloscopia no 3º mês                                  |
    | **4º BACILOSCOPIA**       | Baciloscopia no 4º mês                                  |
    | **5º BACILOSCOPIA**       | Baciloscopia no 5º mês                                  |
    | **6º BACILOSCOPIA**       | Baciloscopia no 6º mês                                  |
    | **RAIO-X**                | Radiografia do tórax                                    |
    | **TIPO**                  | Forma clínica da tuberculose                            |
    | **BACILOSCOPIA_NEGATIVA** | Número de baciloscopias negativas                       |
    """)

elif colunas == "INDÍCE DE RISCO":
    st.title("📊 Índice de Risco para Tuberculose")
    st.write("Esta tabela estima o risco com base nos fatores agravantes de cada paciente.")

    drop_cols = ["ID_UNIDADE", "CS_GESTANT", "CS_ESCOL_N", "ID_MN_RESI", "ID_RG_RESI", "HIV", "AGRAVOUTRA",
                 "CULTURA_ES",
                 "DT_NOTIFIC", "CS_RACA", "TRATAMENTO", "POP_LIBER", "TEST_SENSI", "TRATSUP_AT", "TEST_MOLEC",
                 "ID_AGRAVO"]

    rename_map = {
        "ID_AGRAVO": "ID", "NU_ANO": "ANO", "ID_MUNICIP": "MUNICIPIO", "ID_REGIONA": "REGIAO",
        "DT_DIAG": "DATA DIAGNOSTICO", "NU_IDADE_N": "IDADE", "CS_SEXO": "SEXO", "CS_ZONA": "ZONA",
        "AGRAVAIDS": "AGRAV HIV", "AGRAVALCOO": "AGRAV ALCOOLISMO", "AGRAVDIABE": "AGRAV DIABETES",
        "AGRAVDOENC": "AGRAV DOENCA", "AGRAVDROGA": "AGRAV DROGAS", "AGRAVTABAC": "AGRAV TABACO",
        "DT_INIC_TR": "INICIO DO TRATAMENTO", "SITUA_ENCE": "STATUS ENCERRAMENTO", "DT_ENCERRA": "DATA ENCERRAMENTO",
        "BACILOSC_1": "1º BACILOSCOPIA", "BACILOSC_2": "2º BACILOSCOPIA", "BACILOSC_3": "3º BACILOSCOPIA",
        "BACILOSC_4": "4º BACILOSCOPIA", "BACILOSC_5": "5º BACILOSCOPIA", "BACILOSC_6": "6º BACILOSCOPIA",
        "RAIOX_TORA": "RAIO-X", "FORMA": "TIPO"
    }

    dados = pd.read_csv("dados_tuberculose.csv", sep=";")
    dados = dados.drop(columns=drop_cols).rename(columns=rename_map).dropna()
    dados = dados[~dados.apply(lambda row: row.astype(str).str.contains('ignorado', case=False).any(), axis=1)]

    bac_cols = [f'{i}º BACILOSCOPIA' for i in range(1, 7)]
    dados['BACILOSCOPIA_NEGATIVA'] = dados[bac_cols].apply(lambda row: sum(row.str.contains('Negativa', case=False)),
                                                           axis=1)

    pesos = {'AGRAV HIV': 5, 'AGRAV DIABETES': 2.5, 'AGRAV DROGAS': 2, 'AGRAV ALCOOLISMO': 1.5, 'AGRAV TABACO': 1}
    dados['PONTUACAO RISCO'] = dados[list(pesos)].apply(
        lambda row: sum(pesos[col] for col in pesos if row[col] == 'Sim'), axis=1)


    def classificar_risco(p):
        return 'Baixo Risco' if p <= 3 else 'Médio Risco' if p <= 6 else 'Alto Risco'


    dados['NÍVEL DE RISCO'] = dados['PONTUACAO RISCO'].apply(classificar_risco)

    st.dataframe(dados[["MUNICIPIO", "IDADE", "SEXO", *pesos.keys(), "PONTUACAO RISCO", "NÍVEL DE RISCO",
                        "STATUS ENCERRAMENTO"]],
                 use_container_width=True)

    st.subheader("🔍 Correlação entre Pontuação de Risco e Outras Variáveis")
    dados['AGRAV_HIV_NUM'] = dados['AGRAV HIV'].map({'Sim': 1, 'Não': 0})
    dados['STATUS_NUM'] = dados['STATUS ENCERRAMENTO'].map({'Cura': 1, 'Óbito': 0, 'Abandono': 0})

    corr_resultados = []
    for var in ['IDADE', 'BACILOSCOPIA_NEGATIVA', 'AGRAV_HIV_NUM']:
        r, p = pearsonr(dados['PONTUACAO RISCO'], dados[var])
        interpretacao = "Forte" if abs(r) > 0.5 else "Moderada" if abs(r) > 0.3 else "Fraca"
        corr_resultados.append(
            {"Variável": var, "Correlação (r)": f"{r:.3f}", "p-valor": f"{p:.4f}", "Interpretação": interpretacao})

    st.table(pd.DataFrame(corr_resultados))

    st.markdown("### 🧮 **Pontos por Agravante**")
    st.table(pd.DataFrame(pesos.items(), columns=["Agravante", "Pontuação"]))

    st.markdown("""### 🗂️ **Legenda – Nível de Risco para Tuberculose**
    - 🟢 **Baixo Risco (0 a 3)**: Poucos ou nenhum fator agravante relevante.
    - 🟡 **Médio Risco (3.5 a 6)**: Fatores moderados. Exige atenção.
    - 🔴 **Alto Risco (>6)**: HIV ou múltiplos fatores. Alto acompanhamento médico recomendado.""")

elif colunas == "Previsão com Machine Learning (ML)":
    modelo_data = joblib.load("modelo/modelo_balanceado.pkl")
    modelo = modelo_data["modelo"]
    le_dict = modelo_data["encoders"]
    y_encoder = modelo_data["target_encoder"]
    X_test = modelo_data["X_test"]
    y_test = modelo_data["y_test"]
    y_pred = modelo_data["y_pred"]

    st.title("🤖 Previsão do Status de Encerramento com Machine Learning")

    st.subheader("📈 Métricas Estatísticas do Modelo")
    col1, col2 = st.columns(2)
    with col1:
        mcc = matthews_corrcoef(y_test, y_pred)
        st.metric(label="Correlação de Matthews (MCC)", value=f"{mcc:.4f}")
    with col2:
        acuracia = accuracy_score(y_test, y_pred)
        st.metric(label="Acurácia", value=f"{acuracia * 100:.2f}%")

    conf_matrix = confusion_matrix(y_test, y_pred)
    st.subheader("🔢 Matriz de Confusão")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", ax=ax,
                cbar=False,
                xticklabels=y_encoder.classes_,
                yticklabels=y_encoder.classes_,
                annot_kws={"size": 10})
    ax.set_xlabel("Classe prevista", fontsize=9)
    ax.set_ylabel("Classe verdadeira", fontsize=9)
    st.pyplot(fig)

    report = classification_report(y_test, y_pred, output_dict=True, target_names=y_encoder.classes_)
    report_df = pd.DataFrame(report).transpose()
    report_df = report_df.drop(columns=["support"], errors="ignore")

    st.write("**Relatório de Classificação**")
    st.dataframe(report_df.style.format({
        'precision': '{:.2f}',
        'recall': '{:.2f}',
        'f1-score': '{:.2f}',
    }),
        height=300)

    st.title("🎯 Previsão de Encerramento")

    idade = st.slider("Idade", 0, 120, 30)
    sexo = st.selectbox("Sexo", le_dict['SEXO'].classes_)
    zona = st.selectbox("Zona", le_dict['ZONA'].classes_)
    hiv = st.selectbox("HIV", le_dict['AGRAV HIV'].classes_)
    diabetes = st.selectbox("Diabetes", le_dict['AGRAV DIABETES'].classes_)
    drogas = st.selectbox("Drogas", le_dict['AGRAV DROGAS'].classes_)
    alcool = st.selectbox("Alcoolismo", le_dict['AGRAV ALCOOLISMO'].classes_)
    tabaco = st.selectbox("Tabagismo", le_dict['AGRAV TABACO'].classes_)
    baciloscopia_1 = st.selectbox("1º Baciloscopia", le_dict['1º BACILOSCOPIA'].classes_)
    baciloscopia_negativa = st.slider("Nº de Baciloscopias Negativas", 0, 6, 0)
    raiox = st.selectbox("Resultado do Raio-X", le_dict['RAIO-X'].classes_)
    tipo = st.selectbox("Tipo", ["Pulmonar", "Extrapulmonar", "Pulmonar + Extrapulmonar"])

    input_dict = {
        'IDADE': idade,
        'SEXO': le_dict['SEXO'].transform([sexo])[0],
        'ZONA': le_dict['ZONA'].transform([zona])[0],
        'AGRAV HIV': le_dict['AGRAV HIV'].transform([hiv])[0],
        'AGRAV DIABETES': le_dict['AGRAV DIABETES'].transform([diabetes])[0],
        'AGRAV DROGAS': le_dict['AGRAV DROGAS'].transform([drogas])[0],
        'AGRAV ALCOOLISMO': le_dict['AGRAV ALCOOLISMO'].transform([alcool])[0],
        'AGRAV TABACO': le_dict['AGRAV TABACO'].transform([tabaco])[0],
        'TIPO': le_dict['TIPO'].transform([tipo])[0],
        'RAIO-X': le_dict['RAIO-X'].transform([raiox])[0],
        '1º BACILOSCOPIA': le_dict['1º BACILOSCOPIA'].transform([baciloscopia_1])[0],
        'BACILOSCOPIA_NEGATIVA': baciloscopia_negativa,
    }

    entrada = pd.DataFrame([input_dict])



    if st.button("Prever Status de Encerramento"):
        predicao = modelo.predict(entrada)[0]
        classe_predita = y_encoder.inverse_transform([predicao])[0]
        st.success(f"🧾 Previsão: {classe_predita}")
