# ⚕️ Tech Challenge - Sistema Preditivo de Nível de Obesidade

Este projeto foi desenvolvido como entrega do **Tech Challenge - Fase 4 da Pós-Tech FIAP**.

O objetivo é criar uma solução de **Machine Learning** capaz de auxiliar uma equipe médica na estimativa do nível de obesidade de pacientes, utilizando informações sobre hábitos alimentares, histórico familiar, atividade física e estilo de vida.

A aplicação final foi desenvolvida em **Streamlit** e inclui:

- sistema preditivo de nível de obesidade;
- dashboard analítico com indicadores e gráficos;
- explicação sobre prevenção de data leakage;
- análise de fatores associados à obesidade.

---

## 🚀 Aplicação Online

Acesse a aplicação no Streamlit:

🔗 https://tech-challenge-obesity-kiwwtssmdzugnaeastnk29.streamlit.app/

---

## 🎯 Objetivo do Projeto

O desafio propõe o desenvolvimento de uma pipeline de Machine Learning para prever se uma pessoa apresenta ou pode vir a apresentar obesidade, com assertividade mínima de 75%.

Além da predição, a solução apresenta uma visão analítica dos dados e um **módulo de persistência em banco de dados** para apoiar a equipe médica na interpretação e coleta de novos padrões clínicos.

---

## 📊 Dataset

A base utilizada contém informações relacionadas a características pessoais, hábitos alimentares e estilo de vida.

Principais variáveis utilizadas:

- idade;
- gênero;
- histórico familiar de sobrepeso;
- consumo de alimentos calóricos;
- consumo de vegetais;
- número de refeições principais;
- consumo de água;
- atividade física;
- uso de tecnologia;
- consumo de álcool;
- meio de transporte.

As variáveis `Weight` e `Height` foram removidas da modelagem para reduzir risco de **data leakage**, pois o nível de obesidade pode estar diretamente relacionado ao cálculo do IMC.

---

## 🧠 Pipeline de Machine Learning

A pipeline desenvolvida seguiu as etapas:

1. Carregamento da base de dados;
2. Análise exploratória dos dados;
3. Remoção de duplicatas;
4. Prevenção de data leakage;
5. Separação entre variáveis preditoras e variável alvo;
6. Pré-processamento de variáveis numéricas e categóricas;
7. Comparação entre diferentes modelos;
8. Avaliação de overfitting;
9. Validação cruzada;
10. Escolha do modelo final;
11. Deploy da aplicação em Streamlit.

---

## ⚙️ Pré-processamento

Foram utilizados:

- `StandardScaler` para variáveis numéricas;
- `OneHotEncoder` para variáveis categóricas;
- `ColumnTransformer` para combinar os tratamentos;
- `Pipeline` do Scikit-learn para organizar o fluxo de modelagem.

---

## 🤖 Modelos Avaliados

Foram comparados diferentes algoritmos de classificação:

- Logistic Regression;
- Random Forest sem limite;
- Random Forest Ajustado;
- Extra Trees;
- Gradient Boosting;
- SVC;
- KNN;
- XGBoost

A escolha do modelo final levou em consideração não apenas a acurácia, mas também:

- média da validação cruzada;
- diferença entre treino e teste;
- risco de overfitting;
- interpretabilidade;
- capacidade de generalização.

---

## ✅ Modelo Final

O modelo final escolhido foi o **Random Forest Ajustado**.

Foram utilizados hiperparâmetros para controlar a complexidade do modelo e reduzir risco de overfitting:

```python
RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=4,
    max_features="sqrt",
    random_state=42
)
```

## ✅ Resultados Obtidos

| Métrica | Resultado |
|---|---:|
| Acurácia no teste | ~79% |
| Acurácia no treino | ~88% |
| Gap treino-teste | ~9% |
| Média da validação cruzada | ~80% |
| Desvio da validação cruzada | ~1% |

O modelo atende ao requisito mínimo de assertividade acima de 75% e apresenta melhor equilíbrio entre desempenho e generalização quando comparado a modelos mais complexos que apresentaram maior risco de overfitting.

---

## 📈 Recursos da Aplicação Streamlit

### Dashboard

- total de pacientes;
- percentual de pacientes com obesidade;
- percentual com sobrepeso;
- percentual com peso normal;
- distribuição dos níveis de obesidade;
- histórico familiar por nível de obesidade;
- atividade física por nível de obesidade;
- consumo de água;
- consumo de alimentos calóricos;
- meio de transporte;
- importância das variáveis do modelo.

### Painel de Predição Avançado

 - Gráfico de Probabilidade Clínico: Exibição das chances do paciente pertencer a cada classe através de um gráfico de barras ordenado de forma estática e progressiva por nível de gravidade (de Abaixo do Peso até Obesidade Tipo III).
 - Cores Semânticas de Alerta: Mapeamento visual customizado no Plotly (escala Azul $\rightarrow$ Verde $\rightarrow$ Amarelo $\rightarrow$ Vermelho) para rápida absorção diagnóstica.
 - Validação por IMC Real: Formulário interativo paralelo que calcula o IMC clínico tradicional e compara em tempo real se a predição baseada em comportamento coincide ou diverge do cenário físico atual do paciente.

### Integração com Banco de Dados Nuvem (Supabase)
- Módulo de Feedback Loop: Interface para o médico auditar a predição, selecionar a "Classe Real" confirmada em consulta e salvar o registro diretamente em um banco de dados relacional (PostgreSQL via Supabase).
 - Histórico em Tempo Real: Componente de expansão que consulta o banco de dados via API, exibe os últimos 20 registros avaliados na instituição e permite o download imediato da base atualizada em formato CSV para futuros retreinos do modelo.
 - Segurança da Informação: Credenciais de conexão e chaves de API totalmente blindadas no servidor através do gerenciador de credenciais seguras do Streamlit (st.secrets).

---

## 🩺 Prevenção de Data Leakage

As variáveis `Weight` e `Height` foram removidas do modelo final.

Essa decisão foi tomada porque o nível de obesidade pode estar diretamente relacionado ao IMC, que utiliza peso e altura em seu cálculo. Caso essas variáveis fossem utilizadas, o modelo poderia aprender uma relação muito direta com a variável alvo, gerando uma performance artificialmente elevada.

Com essa abordagem, o modelo passa a considerar principalmente fatores comportamentais, histórico familiar e estilo de vida.

---

## 📁 Estrutura do Projeto

```text
tech-challenge-obesity/
│
├── app.py
├── requirements.txt
├── runtime.txt
├── obesity.csv
├── model_pipeline.pkl
├── notebook_modelagem.ipynb
└── README.md ```
