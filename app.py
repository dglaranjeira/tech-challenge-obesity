import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from datetime import datetime
from pathlib import Path
from supabase import create_client, Client

# ── Cliente Supabase — credenciais lidas de st.secrets ───────────────────────
# Configurar em Streamlit Cloud > Settings > Secrets:
#   SUPABASE_URL = "https://xxxx.supabase.co"
#   SUPABASE_KEY = "eyJ..."
@st.cache_resource
def get_supabase_client() -> Client:
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

# ── Cálculo de classe de obesidade por IMC (referência clínica) ──────────────
def classificar_por_imc(peso_kg, altura_m):
    """Retorna a classe de obesidade calculada pelo IMC, no mesmo padrão do dataset."""
    if altura_m <= 0:
        return None, None
    imc = peso_kg / (altura_m ** 2)
    if imc < 18.5:
        classe = "Insufficient_Weight"
    elif imc < 25.0:
        classe = "Normal_Weight"
    elif imc < 27.5:
        classe = "Overweight_Level_I"
    elif imc < 30.0:
        classe = "Overweight_Level_II"
    elif imc < 35.0:
        classe = "Obesity_Type_I"
    elif imc < 40.0:
        classe = "Obesity_Type_II"
    else:
        classe = "Obesity_Type_III"
    return round(imc, 2), classe

# ======================================================
# Configuração da página
# ======================================================

st.set_page_config(
    page_title="Tech Challenge - Obesidade",
    page_icon="⚕️",
    layout="wide"
)

# ======================================================
# CSS customizado
# ======================================================

st.markdown("""
<style>
.main-title {font-size:42px;font-weight:800;color:#1f2b3d;margin-bottom:8px;}
.subtitle {font-size:18px;color:#4b5563;margin-bottom:28px;}
.section-title {font-size:28px;font-weight:700;color:#1f2b3d;margin-top:25px;margin-bottom:10px;}
.info-box {background-color:#fff8db;color:#7a5c00;padding:18px 22px;border-radius:12px;
           border-left:6px solid #f2c94c;margin-bottom:24px;font-size:16px;}
.success-box {background-color:#e9f9ef;color:#14532d;padding:18px 22px;border-radius:12px;
              border-left:6px solid #22c55e;margin-bottom:24px;font-size:16px;}
.kpi-card {background:linear-gradient(135deg,#1f2b3d 0%,#111827 100%);padding:24px 18px;
           border-radius:14px;text-align:center;box-shadow:0px 4px 14px rgba(0,0,0,0.18);
           min-height:120px;border:1px solid rgba(255,255,255,0.08);}
.kpi-title {color:#93a4c7;font-size:13px;font-weight:700;letter-spacing:1px;
            text-transform:uppercase;margin-bottom:12px;}
.kpi-value {color:#ffffff;font-size:34px;font-weight:850;}
.model-card {background-color:#f8fafc;border:1px solid #e5e7eb;padding:22px;
             border-radius:14px;margin-bottom:20px;}
.stButton>button {background-color:#1f2b3d;color:white;border-radius:10px;
                  padding:0.6rem 1.2rem;border:none;font-weight:700;}
.stButton>button:hover {background-color:#111827;color:white;border:none;}
</style>
""", unsafe_allow_html=True)

# ======================================================
# Traduções
# ======================================================

traducao_obesidade = {
    "Insufficient_Weight": "Abaixo do Peso",
    "Normal_Weight":        "Peso Normal",
    "Overweight_Level_I":   "Sobrepeso Nível I",
    "Overweight_Level_II":  "Sobrepeso Nível II",
    "Obesity_Type_I":       "Obesidade Tipo I",
    "Obesity_Type_II":      "Obesidade Tipo II",
    "Obesity_Type_III":     "Obesidade Tipo III",
}
traducao_genero      = {"Male": "Masculino", "Female": "Feminino"}
traducao_sim_nao     = {"yes": "Sim", "no": "Não"}
traducao_alcool      = {"no": "Não consome", "Sometimes": "Às vezes",
                        "Frequently": "Frequentemente", "Always": "Sempre"}
traducao_caec        = {"no": "Não", "Sometimes": "Às vezes",
                        "Frequently": "Frequentemente", "Always": "Sempre"}
traducao_transporte  = {"Public_Transportation": "Transporte Público", "Walking": "Caminhada",
                        "Automobile": "Automóvel", "Motorbike": "Moto", "Bike": "Bicicleta"}
traducoes_features   = {
    "Age": "Idade", "FCVC": "Consumo de vegetais", "NCP": "Refeições principais",
    "CH2O": "Consumo de água", "FAF": "Atividade física", "TUE": "Uso de tecnologia",
    "Gender": "Gênero", "family_history": "Histórico familiar", "FAVC": "Alimentos calóricos",
    "CAEC": "Come entre refeições", "SMOKE": "Fumo", "SCC": "Monitora calorias",
    "CALC": "Consumo de álcool", "MTRANS": "Transporte",
}

# ======================================================
# Carregamento do modelo — uma vez, disponível em todo o app
# ======================================================
# O pkl contém todos os artefatos gerados pelo notebook:
# pipeline, label_encoder, target_classes, model_name,
# accuracy_test, feature_names e n_samples.

MODEL_PATH = Path(__file__).parent / "model_pipeline.pkl"

@st.cache_resource
def carregar_artefatos():
    return joblib.load(MODEL_PATH)

try:
    artefatos = carregar_artefatos()
except FileNotFoundError:
    st.error("⚠️ model_pipeline.pkl não encontrado. Execute o notebook de treinamento primeiro.")
    st.stop()

pipeline       = artefatos["pipeline"]
label_encoder  = artefatos["label_encoder"]
model_name     = artefatos.get("model_name", "Random Forest")
acuracia       = artefatos.get("accuracy_test", 0.0)
feature_names  = artefatos.get("feature_names", [])
n_samples      = artefatos.get("n_samples", 0)
# df do dashboard construído a partir do CSV original — apenas para gráficos, sem retreino

colunas_modelo = feature_names

# ======================================================
# Dados para o dashboard — lidos do CSV original
# Usado apenas para os gráficos analíticos.
# O modelo não é retreinado — vem inteiramente do pkl.
# ======================================================

DATA_PATH = Path(__file__).parent / "obesity.csv"

@st.cache_data
def carregar_dashboard():
    df = pd.read_csv(DATA_PATH)
    df["Obesity_PT"]         = df["Obesity"].map(traducao_obesidade)
    df["Gender_PT"]          = df["Gender"].map(traducao_genero)
    df["family_history_PT"]  = df["family_history"].map(traducao_sim_nao)
    df["FAVC_PT"]            = df["FAVC"].map(traducao_sim_nao)
    df["SMOKE_PT"]           = df["SMOKE"].map(traducao_sim_nao)
    df["SCC_PT"]             = df["SCC"].map(traducao_sim_nao)
    df["CAEC_PT"]            = df["CAEC"].map(traducao_caec)
    df["CALC_PT"]            = df["CALC"].map(traducao_alcool)
    df["MTRANS_PT"]          = df["MTRANS"].map(traducao_transporte)
    return df

df_dash = carregar_dashboard()

# ======================================================
# Helpers
# ======================================================

def card_kpi(titulo, valor):
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">{titulo}</div>
        <div class="kpi-value">{valor}</div>
    </div>""", unsafe_allow_html=True)

def obter_importancia_features():
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier   = pipeline.named_steps["classifier"]
    num_cols     = list(preprocessor.transformers_[0][2])
    cat_cols     = list(preprocessor.transformers_[1][2])
    cat_names    = preprocessor.named_transformers_["cat"].get_feature_names_out(cat_cols).tolist()
    all_names    = num_cols + cat_names
    df_imp = pd.DataFrame({
        "Variável":    all_names,
        "Importância": classifier.feature_importances_
    }).sort_values("Importância", ascending=False)
    return df_imp

def traduzir_feature(nome):
    for chave, valor in traducoes_features.items():
        nome = nome.replace(chave, valor)
    for orig, trad in [
        ("_yes"," = Sim"),("_no"," = Não"),("_Female"," = Feminino"),("_Male"," = Masculino"),
        ("_Sometimes"," = Às vezes"),("_Frequently"," = Frequentemente"),("_Always"," = Sempre"),
        ("_Public_Transportation"," = Transporte Público"),("_Walking"," = Caminhada"),
        ("_Automobile"," = Automóvel"),("_Motorbike"," = Moto"),("_Bike"," = Bicicleta"),
    ]:
        nome = nome.replace(orig, trad)
    return nome

# ======================================================
# Sidebar
# ======================================================

st.sidebar.title("⚕️ Navegação")
pagina = st.sidebar.radio("Escolha uma página:",
                          ["Predição", "Dashboard Analítico", "Sobre o Projeto"])
st.sidebar.divider()
st.sidebar.info("Tech Challenge - Fase 4\n\nSistema preditivo para apoio à análise do nível de obesidade.")

# ======================================================
# Página 1 — Predição
# ======================================================

if pagina == "Predição":

    st.markdown('<div class="main-title">⚕️ Sistema Preditivo de Nível de Obesidade</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Ferramenta de apoio à decisão para estimar o nível de obesidade '
                'com base em hábitos alimentares, histórico familiar, atividade física e estilo de vida.</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="info-box"><b>Aviso importante:</b> este sistema é uma ferramenta de apoio à decisão '
                'e não substitui avaliação médica profissional.<br><br>',
                unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="section-title">Dados do Paciente</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        gender_pt = st.selectbox("Gênero", ["Feminino", "Masculino"])
        age = st.number_input("Idade", min_value=1, max_value=100, value=25)
        family_history_pt = st.selectbox(
            "Histórico familiar de sobrepeso?",
            ["Sim", "Não"],
            help="Algum familiar (pais, irmãos) tem ou teve sobrepeso ou obesidade?"
        )
        favc_pt = st.selectbox(
            "Consome alimentos altamente calóricos com frequência? (FAVC)",
            ["Sim", "Não"],
            help="Alimentos como fast food, frituras e doces industrializados."
        )

    with col2:
        fcvc_pt = st.selectbox(
            "Frequência de consumo de vegetais (FCVC)",
            ["Raramente", "Às vezes", "Sempre"],
            index=1,
            help="Com que frequência inclui vegetais nas refeições? Raramente = quase nunca · Às vezes = algumas refeições · Sempre = na maioria das refeições."
        )
        ncp_pt = st.selectbox(
            "Número de refeições principais por dia (NCP)",
            ["1 refeição", "2 refeições", "3 refeições", "4 ou mais refeições"],
            index=2,
            help="Conte apenas refeições principais (café da manhã, almoço, jantar). Lanches contam na próxima pergunta."
        )
        caec_pt = st.selectbox(
            "Come entre as refeições? (CAEC)",
            ["Não", "Às vezes", "Frequentemente", "Sempre"],
            help="Com que frequência faz lanches ou petiscos fora das refeições principais?"
        )
        smoke_pt = st.selectbox("Fuma? (SMOKE)", ["Não", "Sim"])

    with col3:
        ch2o_pt = st.selectbox(
            "Consumo diário de água (CH2O)",
            ["Menos de 1 litro", "Entre 1 e 2 litros", "Mais de 2 litros"],
            index=1,
            help="Considere toda a água ingerida ao longo do dia, incluindo chás e sucos naturais."
        )
        scc_pt = st.selectbox(
            "Monitora calorias ingeridas? (SCC)",
            ["Não", "Sim"],
            help="Usa aplicativo, diário ou outro método para acompanhar a ingestão calórica?"
        )
        faf_pt = st.selectbox(
            "Frequência de atividade física por semana (FAF)",
            ["Nenhuma", "1 a 2 dias", "3 a 4 dias", "4 a 5 dias"],
            index=1,
            help="Atividade física moderada ou intensa: caminhada rápida, academia, esporte, etc."
        )
        tue_pt = st.selectbox(
            "Tempo diário em dispositivos tecnológicos (TUE)",
            ["0 a 2 horas", "3 a 5 horas", "Mais de 5 horas"],
            index=1,
            help="Tempo de tela fora do trabalho: celular, TV, computador, videogame, etc."
        )

    col4, col5 = st.columns(2)
    with col4:
        calc_pt = st.selectbox(
            "Frequência de consumo de álcool (CALC)",
            ["Não consome", "Às vezes", "Frequentemente", "Sempre"],
            help="Com que frequência consome bebidas alcoólicas?"
        )
    with col5:
        mtrans_pt = st.selectbox(
            "Meio de transporte principal (MTRANS)",
            ["Transporte Público", "Caminhada", "Automóvel", "Moto", "Bicicleta"],
            help="Qual o meio de transporte mais usado no dia a dia?"
        )

    gender         = {"Feminino": "Female", "Masculino": "Male"}[gender_pt]
    family_history = {"Sim": "yes", "Não": "no"}[family_history_pt]
    favc           = {"Sim": "yes", "Não": "no"}[favc_pt]
    smoke          = {"Não": "no", "Sim": "yes"}[smoke_pt]
    scc            = {"Não": "no", "Sim": "yes"}[scc_pt]
    caec           = {v: k for k, v in traducao_caec.items()}[caec_pt]
    calc           = {v: k for k, v in traducao_alcool.items()}[calc_pt]
    mtrans         = {v: k for k, v in traducao_transporte.items()}[mtrans_pt]

    # Variáveis ordinais mapeadas para inteiros — consistente com o tratamento do notebook,
    # que arredonda os valores sintéticos contínuos (ex: FAF=1.673) para o inteiro mais próximo
    # antes do treino. O app envia os mesmos valores inteiros que o modelo recebeu no treino.
    fcvc = {"Raramente": 1, "Às vezes": 2, "Sempre": 3}[fcvc_pt]
    ncp  = {"1 refeição": 1, "2 refeições": 2, "3 refeições": 3, "4 ou mais refeições": 4}[ncp_pt]
    ch2o = {"Menos de 1 litro": 1, "Entre 1 e 2 litros": 2, "Mais de 2 litros": 3}[ch2o_pt]
    faf  = {"Nenhuma": 0, "1 a 2 dias": 1, "3 a 4 dias": 2, "4 a 5 dias": 3}[faf_pt]
    tue  = {"0 a 2 horas": 0, "3 a 5 horas": 1, "Mais de 5 horas": 2}[tue_pt]

    dados_paciente = pd.DataFrame([{
        "Gender": gender, "Age": age, "family_history": family_history,
        "FAVC": favc, "FCVC": fcvc, "NCP": ncp, "CAEC": caec, "SMOKE": smoke,
        "CH2O": ch2o, "SCC": scc, "FAF": faf, "TUE": tue, "CALC": calc, "MTRANS": mtrans,
    }])[colunas_modelo]

    st.divider()

    # Estado da predição persiste fora do botão para o formulário de comparação ficar visível
    if st.button("Prever nível de obesidade"):
        pred_enc   = pipeline.predict(dados_paciente)[0]
        pred_prob  = pipeline.predict_proba(dados_paciente)[0]
        pred_class = label_encoder.inverse_transform([pred_enc])[0]

        st.session_state["pred_class"] = pred_class
        st.session_state["pred_prob"]  = pred_prob
        st.session_state["dados_paciente"] = dados_paciente

    if "pred_class" in st.session_state:
        pred_class = st.session_state["pred_class"]
        pred_prob  = st.session_state["pred_prob"]
        pred_pt    = traducao_obesidade.get(pred_class, pred_class)

        st.markdown('<div class="section-title">Resultado da Previsão</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="success-box"><b>Nível previsto:</b> {pred_pt}</div>',
                    unsafe_allow_html=True)

        # Gráfico + formulário de comparação lado a lado
        col_chart, col_form = st.columns([1.3, 1])

        with col_chart:
            # 1. Ordem clínica correta utilizando o novo termo
            ordem_clinica = [
                "Abaixo do Peso",
                "Peso Normal",
                "Sobrepeso Nível I",
                "Sobrepeso Nível II",
                "Obesidade Tipo I",
                "Obesidade Tipo II",
                "Obesidade Tipo III"
            ]

            # 2. Mapeamento de cores correspondente
            mapa_cores = {
                "Abaixo do Peso":      "#3b82f6",  # Azul
                "Peso Normal":        "#22c55e",  # Verde
                "Sobrepeso Nível I":   "#eab308",  # Amarelo
                "Sobrepeso Nível II":  "#f97316",  # Laranja
                "Obesidade Tipo I":    "#ef4444",  # Vermelho claro
                "Obesidade Tipo II":   "#dc2626",  # Vermelho
                "Obesidade Tipo III":  "#991b1b"   # Vermelho escuro / Roxo
            }

            classes = label_encoder.classes_
            df_prob = pd.DataFrame({
                "Classe":        [traducao_obesidade.get(c, c) for c in classes],
                "Probabilidade": pred_prob,
            })
            
            # 3. Força a coluna a ser categórica respeitando a ordem sequencial biológica
            df_prob["Classe"] = pd.Categorical(df_prob["Classe"], categories=ordem_clinica, ordered=True)
            df_prob = df_prob.sort_values("Classe")

            # 4. Gera o gráfico único, ordenado e com cores semânticas fixas
            fig_prob = px.bar(
                df_prob,
                x="Classe",
                y="Probabilidade",
                color="Classe",
                color_discrete_map=mapa_cores,
                title="Probabilidade estimada por classe",
                text=df_prob["Probabilidade"].apply(lambda x: f"{x*100:.1f}%")
            )
            
            fig_prob.update_traces(width=0.7, textposition="outside")
            fig_prob.update_layout(
                showlegend=False,
                bargap=0.2,
                height=450,
                title_font_size=18,
                yaxis_tickformat=".0%",
                xaxis_title=""
            )
            
            st.plotly_chart(fig_prob, use_container_width=True)

        with col_form:
            st.markdown('<div class="model-card"><b>Comparação com IMC real</b><br>'
                        '<span style="font-size:0.9em;color:#666;">Informe peso e altura para '
                        'comparar a previsão do modelo com a classificação clínica por IMC.</span>'
                        '</div>', unsafe_allow_html=True)

            peso_real = st.number_input("Peso atual (kg)", min_value=20.0, max_value=300.0,
                                        value=70.0, step=0.5)
            altura_real = st.number_input("Altura (m)", min_value=0.50, max_value=2.50,
                                          value=1.70, step=0.01)

            imc, classe_imc = classificar_por_imc(peso_real, altura_real)

            if imc is not None:
                classe_imc_pt  = traducao_obesidade.get(classe_imc, classe_imc)
                bate           = (classe_imc == pred_class)
                cor_box        = "#e9f9ef" if bate else "#fff4e6"
                cor_borda      = "#22c55e" if bate else "#f59e0b"
                cor_texto      = "#14532d" if bate else "#7c4a03"
                simbolo        = "✅" if bate else "⚠️"
                rotulo_match   = "Previsão coincide com IMC" if bate else "Previsão diverge do IMC"

                st.markdown(
                    f"""<div style="background-color:{cor_box};color:{cor_texto};
                    padding:14px 18px;border-radius:10px;border-left:5px solid {cor_borda};
                    margin-top:10px;font-size:15px;">
                    <b>IMC calculado:</b> {imc}<br>
                    <b>Classe por IMC:</b> {classe_imc_pt}<br>
                    <b>Previsão do modelo:</b> {pred_pt}<br>
                    {simbolo} {rotulo_match}
                    </div>""",
                    unsafe_allow_html=True
                )

            st.divider()

            # ── Salvar registro para melhoria do modelo ──────────────────────
            st.markdown("<b>Salvar registro para melhoria do modelo</b>", unsafe_allow_html=True)
            st.caption("O médico confirma qual é a classe real do paciente antes de salvar.")

            classe_real_pt = st.selectbox(
                "Classe real (confirmada pelo médico)",
                list(traducao_obesidade.values()),
                index=list(traducao_obesidade.values()).index(traducao_obesidade.get(classe_imc, "Peso Normal"))
                      if classe_imc else 1
            )
            classe_real = {v: k for k, v in traducao_obesidade.items()}[classe_real_pt]

            if st.button("💾 Salvar registro"):
                # Monta linha no mesmo formato da tabela registros_medicos no Supabase.
                # Os nomes das colunas devem bater com o SQL de criação da tabela.
                dados_orig = st.session_state["dados_paciente"].iloc[0].to_dict()
                registro = {
                    "gender":          dados_orig["Gender"],
                    "age":             float(dados_orig["Age"]),
                    "height":          float(altura_real),
                    "weight":          float(peso_real),
                    "family_history":  dados_orig["family_history"],
                    "favc":            dados_orig["FAVC"],
                    "fcvc":            float(dados_orig["FCVC"]),
                    "ncp":             float(dados_orig["NCP"]),
                    "caec":            dados_orig["CAEC"],
                    "smoke":           dados_orig["SMOKE"],
                    "ch2o":            float(dados_orig["CH2O"]),
                    "scc":             dados_orig["SCC"],
                    "faf":             float(dados_orig["FAF"]),
                    "tue":             float(dados_orig["TUE"]),
                    "calc":            dados_orig["CALC"],
                    "mtrans":          dados_orig["MTRANS"],
                    "obesity_real":    classe_real,
                    "predicao_modelo": pred_class,
                    "imc":             float(imc) if imc is not None else None
                }

                try:
                    supabase = get_supabase_client()
                    supabase.table("registros_medicos").insert(registro).execute()
                    st.success(f"Registro salvo no banco. Classe real: {classe_real_pt}")
                except KeyError:
                    st.error("Credenciais do Supabase não configuradas. "
                             "Defina SUPABASE_URL e SUPABASE_KEY em Settings > Secrets.")
                except Exception as e:
                    st.error(f"Erro ao salvar no banco: {e}")

            # ── Visualizar registros recentes salvos no banco ────────────────
            with st.expander("📋 Ver registros salvos recentemente"):
                try:
                    supabase = get_supabase_client()
                    resp = supabase.table("registros_medicos") \
                                   .select("*") \
                                   .order("created_at", desc=True) \
                                   .limit(20) \
                                   .execute()
                    if resp.data:
                        df_hist = pd.DataFrame(resp.data)
                        st.dataframe(df_hist, use_container_width=True, height=300)

                        # Download dos últimos 20 registros
                        csv_bytes = df_hist.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "⬇️ Baixar registros (CSV)",
                            data=csv_bytes,
                            file_name="registros_medicos.csv",
                            mime="text/csv"
                        )
                    else:
                        st.info("Nenhum registro salvo ainda.")
                except Exception as e:
                    st.caption(f"Não foi possível carregar os registros: {e}")

# ======================================================
# Página 2 — Dashboard Analítico
# ======================================================

elif pagina == "Dashboard Analítico":

    st.markdown('<div class="main-title">📊 Dashboard Analítico — Padrões de Obesidade</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Insights sobre a distribuição dos níveis de obesidade e fatores '
                'associados ao comportamento, histórico familiar e estilo de vida dos pacientes.</div>',
                unsafe_allow_html=True)

    # KPIs
    obesos     = df_dash["Obesity"].isin(["Obesity_Type_I","Obesity_Type_II","Obesity_Type_III"]).mean()*100
    sobrepeso  = df_dash["Obesity"].isin(["Overweight_Level_I","Overweight_Level_II"]).mean()*100
    peso_normal= (df_dash["Obesity"] == "Normal_Weight").mean()*100

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1: card_kpi("Pacientes",  f"{n_samples:,}".replace(",","."))
    with k2: card_kpi("Obesidade",  f"{obesos:.1f}%")
    with k3: card_kpi("Sobrepeso",  f"{sobrepeso:.1f}%")
    with k4: card_kpi("Peso Normal",f"{peso_normal:.1f}%")
    with k5: card_kpi("Acurácia RF",f"{acuracia:.1%}")

    st.divider()

    contagem = df_dash["Obesity_PT"].value_counts().reset_index()
    contagem.columns = ["Nível de obesidade", "Quantidade"]

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            contagem,
            x="Nível de obesidade",
            y="Quantidade",
            color="Nível de obesidade",
            title="Distribuição dos Níveis de Obesidade",
            text="Quantidade"
        )
        fig.update_traces(width=0.7, textposition="outside")
        fig.update_layout(
            showlegend=False,
            xaxis_title="",
            yaxis_title="Quantidade de pacientes",
            title_font_size=20,
            bargap=0.2,
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.pie(contagem, names="Nível de obesidade", values="Quantidade",
                     title="Proporção das Classes", hole=0.55)
        fig.update_layout(title_font_size=20)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    hist = pd.crosstab(df_dash["Obesity_PT"], df_dash["family_history_PT"],
                       normalize="index") * 100
    hist = hist.reset_index().melt(id_vars="Obesity_PT", var_name="Histórico familiar",
                                   value_name="Percentual")
    fig = px.bar(hist, x="Obesity_PT", y="Percentual", color="Histórico familiar",
                 title="Histórico Familiar × Nível de Obesidade", barmode="stack")
    fig.update_layout(xaxis_title="", yaxis_title="% dentro de cada nível", title_font_size=20)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('<div class="model-card"><b>Insight:</b> o histórico familiar aparece como um dos fatores '
                'relevantes para identificar predisposição a níveis mais elevados de sobrepeso e obesidade.</div>',
                unsafe_allow_html=True)

    st.divider()

    col3, col4 = st.columns(2)
    with col3:
        fig = px.box(df_dash, x="Obesity_PT", y="FAF", color="Obesity_PT",
                     title="Atividade Física × Nível de Obesidade")
        fig.update_layout(showlegend=False, xaxis_title="",
                          yaxis_title="Frequência de atividade física", title_font_size=20)
        st.plotly_chart(fig, use_container_width=True)
    with col4:
        fig = px.box(df_dash, x="Obesity_PT", y="CH2O", color="Obesity_PT",
                     title="Consumo de Água × Nível de Obesidade")
        fig.update_layout(showlegend=False, xaxis_title="",
                          yaxis_title="Consumo diário de água", title_font_size=20)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    col5, col6 = st.columns(2)

    favc_cross = pd.crosstab(df_dash["Obesity_PT"], df_dash["FAVC_PT"],
                              normalize="index") * 100
    favc_cross = favc_cross.reset_index().melt(id_vars="Obesity_PT",
                                               var_name="Consome alimentos calóricos",
                                               value_name="Percentual")
    with col5:
        fig = px.bar(favc_cross, x="Obesity_PT", y="Percentual",
                     color="Consome alimentos calóricos",
                     title="Alimentos Calóricos × Nível de Obesidade", barmode="stack")
        fig.update_layout(xaxis_title="", yaxis_title="% dentro de cada nível", title_font_size=20)
        st.plotly_chart(fig, use_container_width=True)

    mtrans_cross = pd.crosstab(df_dash["Obesity_PT"], df_dash["MTRANS_PT"],
                               normalize="index") * 100
    mtrans_cross = mtrans_cross.reset_index().melt(id_vars="Obesity_PT",
                                                   var_name="Meio de transporte",
                                                   value_name="Percentual")
    with col6:
        fig = px.bar(mtrans_cross, x="Obesity_PT", y="Percentual", color="Meio de transporte",
                     title="Meio de Transporte × Nível de Obesidade", barmode="stack")
        fig.update_layout(xaxis_title="", yaxis_title="% dentro de cada nível", title_font_size=20)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown('<div class="section-title">Importância das Variáveis no Modelo</div>',
                unsafe_allow_html=True)

    df_imp = obter_importancia_features().head(15)
    df_imp["Variável"] = df_imp["Variável"].apply(traduzir_feature)
    fig = px.bar(df_imp.sort_values("Importância"), x="Importância", y="Variável",
                 orientation="h", color="Importância",
                 title="Top 15 Variáveis Mais Importantes")
    fig.update_layout(title_font_size=20)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown('<div class="section-title">Conclusão Executiva</div>', unsafe_allow_html=True)
    st.markdown('<div class="success-box">A análise indica que fatores como histórico familiar, hábitos '
                'alimentares, frequência de atividade física, consumo de água, uso de tecnologia e meio de '
                'transporte contribuem para a identificação dos diferentes níveis de obesidade.<br><br>'
                'Como peso e altura foram removidos do modelo, a solução se torna mais útil como ferramenta '
                'de triagem baseada em comportamento e estilo de vida, apoiando a equipe médica na '
                'identificação de padrões de risco.</div>', unsafe_allow_html=True)

# ======================================================
# Página 3 — Sobre o Projeto
# ======================================================

else:

    st.markdown('<div class="main-title">📘 Sobre o Projeto</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Resumo metodológico da solução de Machine Learning '
                'desenvolvida para o Tech Challenge.</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Objetivo</div>', unsafe_allow_html=True)
    st.write("O objetivo deste projeto é desenvolver uma solução de Machine Learning capaz de auxiliar "
             "profissionais de saúde na previsão do nível de obesidade de pacientes.")

    st.markdown('<div class="section-title">Pipeline de Machine Learning</div>', unsafe_allow_html=True)
    st.markdown('<div class="model-card">A pipeline desenvolvida inclui:<br><br>'
                '<b>1.</b> Leitura e entendimento da base de dados;<br>'
                '<b>2.</b> Análise exploratória dos dados;<br>'
                '<b>3.</b> Análise de duplicatas sem remoção (justificada no notebook);<br>'
                '<b>4.</b> Remoção de variáveis com risco de data leakage;<br>'
                '<b>5.</b> Pré-processamento de variáveis numéricas e categóricas;<br>'
                '<b>6.</b> Treinamento e comparação de modelos com StratifiedKFold K=10;<br>'
                '<b>7.</b> Fine tuning via RandomizedSearchCV com controle de overfitting;<br>'
                '<b>8.</b> Avaliação com métricas macro, curva ROC e curva de aprendizado;<br>'
                '<b>9.</b> Implantação do modelo em aplicação Streamlit.</div>',
                unsafe_allow_html=True)

    st.markdown('<div class="section-title">Prevenção de Data Leakage</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box"><b>Aviso importante:</b> este sistema é uma ferramenta de apoio à decisão '
                'e não substitui avaliação médica profissional.<br><br>'
                'As variáveis <b>peso</b> e <b>altura</b> foram removidas do modelo para reduzir risco de '
                '<i>data leakage</i>, já que o nível de obesidade pode estar diretamente relacionado ao IMC.</div>',
                unsafe_allow_html=True)

    st.markdown('<div class="section-title">Modelo Utilizado</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1: card_kpi("Modelo", model_name)
    with c2: card_kpi("Acurácia CV K=10", f"{acuracia:.1%}")
    with c3: card_kpi("Objetivo", "Triagem")

    st.write("O modelo final utilizado foi um **Random Forest Classifier** com hiperparâmetros otimizados "
             "via RandomizedSearchCV. A complexidade das árvores foi limitada intencionalmente para reduzir "
             "overfitting, priorizando confiança na estimativa em detrimento de acurácia máxima aparente.")

    st.markdown('<div class="section-title">Limitações</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Esta aplicação não deve ser utilizada como diagnóstico médico '
                'definitivo. O resultado gerado pelo modelo deve ser interpretado como apoio à decisão, '
                'sendo necessária a avaliação clínica de profissionais de saúde.</div>',
                unsafe_allow_html=True)
