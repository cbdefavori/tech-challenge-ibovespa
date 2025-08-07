# Previsão do IBOVESPA com Machine Learning

Este projeto foi desenvolvido como parte da **Fase 2 do Tech Challenge** da **Pós-Tech FIAP em Data Analytics**. O desafio propôs a aplicação prática de análise de dados e algoritmos de aprendizado de máquina para prever a direção do fechamento do índice **IBOVESPA** (alta ou baixa), com base em dados históricos.

---

## Cenário

Assumi o papel de **Cientista de Dados** em um grande fundo de investimentos brasileiro, com a missão de desenvolver um modelo preditivo para apoiar a área de análise quantitativa. A previsão alimentaria dashboards internos, orientando decisões estratégicas com base na tendência diária do IBOVESPA.

---

## Objetivo

Construir e avaliar modelos capazes de prever a **tendência de fechamento do IBOVESPA** para diferentes horizontes de tempo:

- **T+1** (dia seguinte)
- **T+2** (dois dias à frente)
- **T+5** (cinco dias à frente)

Foram exploradas diferentes janelas de treino (de 2 a 25 anos), múltiplas abordagens de modelagem, validações cruzadas temporais, otimizações de hiperparâmetros, estratégias de ensemble e ajustes finos de thresholds para tomada de decisão mais sensível.

---

## Estrutura do Projeto

- Aquisição de dados via `yfinance`
- Pré-processamento e criação de features (lags, retornos, médias móveis, indicadores técnicos)
- Criação de targets binários e multiclasse (±0.5%)
- Separação temporal entre treino e teste (últimos 30 dias)
- Testes com diversos algoritmos de Machine Learning
- Ajuste de thresholds (cut-off) para probabilidade de decisão
- Comparação entre targets T+1, T+2 e T+5

---

## Bibliotecas e Ferramentas Utilizadas

```python
# Manipulação de dados
pandas, numpy, calendar, datetime

# Coleta de dados
yfinance

# Modelagem
scikit-learn, xgboost, lightgbm, prophet, tensorflow.keras

# Pré-processamento e validação
MinMaxScaler, StandardScaler, TimeSeriesSplit, GridSearchCV, RandomizedSearchCV

# Modelos testados
XGBClassifier, LGBMClassifier, RandomForestClassifier, LogisticRegression, SVC, LSTM, Prophet

# Ensemble
VotingClassifier, StackingClassifier

# Avaliação
accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

# Visualização
matplotlib, seaborn

# Persistência de modelos
joblib
```

---

## Modelos Avaliados

- XGBoost (com e sem otimização)
- LightGBM (com e sem otimização)
- Random Forest (sem otimização)
- Regressão Logística
- SVM
- Prophet (executado com binarização)
- LSTM (executado com binarização)
- Ensemble (votação)
- Stacking (XGB + LGBM)

---

## Melhores Resultados

| Target | Modelo                        | Acurácia | F1-Score | Observações                                |
|--------|-------------------------------|----------|----------|--------------------------------------------|
| T+1    | 🥇 **Random Forest (simples)** | **73.33%** | **63.64%** | Melhor desempenho geral (sem otimização)   |
| T+1    | LightGBM (sem otimização)     | 70.00%   | 60.87%   | Forte desempenho exploratório              |
| T+1    | LightGBM (com otimização + cutoff 0.4) | 66.67% | 58.33% | Modelo mais balanceado                     |
| T+1    | XGBoost (com otimização)      | 66.67%   | 50.00%   | Resultados consistentes, mas menos robusto |
| T+1    | Ensemble (XGB + LGBM + RF)    | 63.33%   | 47.62%   | Robusto, mas abaixo dos modelos isolados   |
| T+2    | LightGBM (sem otimização)     | 63.33%   | 52.17%   | Boa estabilidade em janelas futuras        |
| T+5    | LightGBM (sem otimização)     | **76.67%** | 56.41% | Superou a meta de acurácia de 75%          |

---

## Avaliação de Modelos Prophet e LSTM

| Modelo   | Acurácia | F1 (macro) | Observações                                      |
|----------|----------|------------|--------------------------------------------------|
| Prophet  | 56.67%   | 55.56%     | Conversão por variação percentual (classe 0/1)   |
| LSTM     | 53.33%   | 52.17%     | Modelo binarizado com base na variação diária    |

---

## Estratégias Adotadas

- **Validação temporal** com `TimeSeriesSplit`
- **RandomizedSearchCV** para otimização de hiperparâmetros
- **Ajuste de thresholds (cutoff)** para maximizar o F1-score
- **Target multiclasse** com variação mínima de ±0.5%
- **Testes com janelas móveis** (2, 5, 10, 15, 20 e 25 anos de histórico)
- **Ensembles e Stacking** com comparações detalhadas

---

## Conclusões

- O modelo **Random Forest simples**, com features básicas e janela completa (1993–2025), superou abordagens mais complexas.
- Modelos baseados em árvore (RF, LGBM, XGB) tiveram melhor desempenho em geral.
- **Otimizações e ajustes de cutoff** aumentaram a sensibilidade em targets mais desafiadores.
- A complexidade não garantiu ganho de performance — a simplicidade venceu neste caso.
- Estratégias como target multiclasse e uso de indicadores técnicos trouxeram bons insights, mas os modelos binários com T+1 e T+2 foram os mais eficientes.

---

## Autora

**Carol Defavori**  
Analista de Dados | Cientista de Dados • Pós-Tech FIAP – Data Analytics  
[LinkedIn](https://www.linkedin.com/in/caroldefavori) • [GitHub](https://github.com/caroldefavori)

---

## Requisitos Técnicos

- Python 3.10+
- Jupyter Notebook ou JupyterLab
- Ambiente virtual com `venv`, `conda` ou `poetry`

Instale as dependências com:

```bash
pip install -r requirements.txt
```

> Algumas bibliotecas, como `xgboost`, `lightgbm`, `prophet` e `tensorflow`, podem requerer compiladores ou ambientes específicos.

---

## Reprodutibilidade

- Dados obtidos via `yfinance` com corte em **31/07/2025**
- Separação treino/teste com base em data (últimos 30 dias são teste)
- Sementes fixas e validação temporal garantem estabilidade
- Código modular, comentado e executável de ponta a ponta
