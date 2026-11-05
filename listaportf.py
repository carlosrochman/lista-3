# =====================================================================
# LISTA 3 - PARTE 1: PORTFÓLIOS
# Base de dados: só a planilha da Economatica (economatica_5.xlsx)
# Pedi ajuda para o claude também não só para a estrutura do código, mas para dizer quais bibliotecas usar.

# =====================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import statsmodels.api as sm

ARQUIVO  = "economatica_5.xlsx"
ACOES    = ["PETR4", "VALE3", "ITUB4", "WEGE3", "ABEV3"]
RF_ANUAL = 0.15                              # taxa livre de risco (Selic ~15% a.a. em dez/2025; confira no BCB)
DIAS     = 252                               # pregões por ano
RF_DIA   = (1 + RF_ANUAL) ** (1 / DIAS) - 1  # a mesma taxa, em base diária

# ---------------------------------------------------------------------
# 0) LER A PLANILHA E CALCULAR OS RETORNOS
# ---------------------------------------------------------------------
# A linha 4 tem os cabeçalhos e a coluna A tem as datas (por isso header=3 e index_col=0).
# O nome de cada coluna vem em várias linhas ("Fechamento / ajust p/ prov / ... / PETR4"):
# ficamos só com a última linha, que é o ticker.
dados = pd.read_excel(ARQUIVO, header=3, index_col=0)
dados.columns = [str(c).split("\n")[-1] for c in dados.columns]
dados = dados.iloc[:, 1:]                                  # a 1ª coluna de preço é a PETR4 repetida
dados = dados.replace("-", np.nan).astype(float)           # feriados vêm como "-"
dados = dados.dropna(how="all")                            # tira os dias sem pregão

precos = dados[ACOES]                                      # só as 5 ações
ret = precos.pct_change().dropna()                         # retorno simples diário
print(f"{len(ret)} retornos diários, de {ret.index[0].date()} a {ret.index[-1].date()}")

# ---------------------------------------------------------------------
# EXERCÍCIO 1 - FRONTEIRA EFICIENTE DE MARKOWITZ E CARTEIRA DE MÁXIMO SHARPE
# ---------------------------------------------------------------------
# Os dois ingredientes de Markowitz: retorno esperado de cada ação (média histórica)
# e matriz de covariância entre elas. Tudo anualizado (x252).
mu  = ret.mean().values * DIAS
cov = ret.cov().values * DIAS

def retorno(w): return w @ mu                       # E[Rp] = soma de w_i * E[R_i]
def risco(w):   return np.sqrt(w @ cov @ w)         # desvio padrão da carteira
def sharpe(w):  return (retorno(w) - RF_ANUAL) / risco(w)

n = len(ACOES)
soma_100 = {"type": "eq", "fun": lambda w: w.sum() - 1}   # os pesos têm que somar 100%
limites  = [(0, 1)] * n                                   # sem venda a descoberto (peso entre 0 e 100%)
w0       = np.ones(n) / n                                 # ponto de partida do otimizador: 20% em cada

# a) Carteira de mínima variância global (o "bico" da curva)
w_min = minimize(risco, w0, bounds=limites, constraints=[soma_100]).x

# b) Fronteira: para cada retorno-alvo, acha a carteira de MENOR risco
alvos = np.linspace(retorno(w_min), mu.max(), 40)
risco_front = []
for alvo in alvos:
    restricoes = [soma_100, {"type": "eq", "fun": lambda w, a=alvo: retorno(w) - a}]
    w = minimize(risco, w0, bounds=limites, constraints=restricoes).x
    risco_front.append(risco(w))

# c) Carteira de máximo Sharpe (carteira de tangência): maximiza o Sharpe = minimiza -Sharpe
w_otimo = minimize(lambda w: -sharpe(w), w0, bounds=limites, constraints=[soma_100]).x
print("\nPesos da carteira de máximo Sharpe:")
print(pd.Series(w_otimo, index=ACOES).round(4).to_string())
print(f"Retorno {retorno(w_otimo):.2%} | Volatilidade {risco(w_otimo):.2%} | Sharpe {sharpe(w_otimo):.2f}")

# d) Gráfico. Nuvem cinza = 5.000 carteiras com pesos aleatórios (todas as combinações possíveis).
#    Linha azul = fronteira eficiente: para cada nível de risco, o MAIOR retorno possível.
#    Qualquer ponto abaixo/à direita dela é ineficiente (dá para ter mais retorno com o mesmo risco).
#    Estrela vermelha = máximo Sharpe. Linha verde (CML) = sai da taxa livre de risco e tangencia a fronteira.
sim = np.random.default_rng(0).dirichlet(np.ones(n), 5000)
sim_ret = sim @ mu
sim_risco = np.sqrt(((sim @ cov) * sim).sum(axis=1))

plt.figure(figsize=(10, 6))
plt.scatter(sim_risco, sim_ret, c="lightgray", s=6, label="Carteiras aleatórias")
plt.plot(risco_front, alvos, "b-", lw=2.5, label="Fronteira eficiente")
plt.scatter(np.sqrt(np.diag(cov)), mu, c="black", s=40)               # cada ação sozinha
for i, a in enumerate(ACOES):
    plt.annotate(a, (np.sqrt(cov[i, i]), mu[i]), xytext=(5, 5), textcoords="offset points")
plt.scatter(risco(w_min), retorno(w_min), marker="D", c="orange", s=90, label="Mínima variância")
plt.scatter(risco(w_otimo), retorno(w_otimo), marker="*", c="red", s=300, label="Máximo Sharpe")
x = np.array([0, sim_risco.max()])
plt.plot(x, RF_ANUAL + sharpe(w_otimo) * x, "g--", label="Linha de mercado de capitais")
plt.xlabel("Risco (desvio padrão anual)")
plt.ylabel("Retorno esperado anual")
plt.title("Fronteira eficiente de Markowitz")
plt.legend()
plt.show()

# ---------------------------------------------------------------------
# EXERCÍCIO 2 - CARTEIRA ÓTIMA x CARTEIRA SEM PONDERAÇÃO (pesos iguais)
# ---------------------------------------------------------------------
# "Sem ponderação específica" = 20% em cada ação. Retorno diário da carteira = soma de peso x retorno.
carteiras = pd.DataFrame({
    "Máximo Sharpe": ret @ w_otimo,
    "Pesos iguais":  ret @ w0,
})

tabela = pd.DataFrame({
    "Retorno anual":     carteiras.mean() * DIAS,
    "Volatilidade anual": carteiras.std() * np.sqrt(DIAS),
})
tabela["Sharpe"] = (tabela["Retorno anual"] - RF_ANUAL) / tabela["Volatilidade anual"]
print("\n", tabela.round(4))

# Retorno acumulado: quanto R$ 1 investido no início vale ao longo do tempo
acumulado = (1 + carteiras).cumprod() - 1
acumulado.plot(figsize=(10, 5), title="Retorno acumulado: máximo Sharpe x pesos iguais")
plt.ylabel("Retorno acumulado")
plt.show()

# ---------------------------------------------------------------------
# EXERCÍCIO 3 - RETORNO ESPERADO PELO CAPM E PELO FAMA-FRENCH (3 FATORES)
# ---------------------------------------------------------------------
# Ideia: regredir o retorno em excesso da carteira (retorno - Rf) nos fatores de risco.
#   CAPM:         R_p - Rf = alfa + beta * (R_m - Rf)
#   Fama-French:  R_p - Rf = alfa + b1*(R_m - Rf) + b2*SMB + b3*HML
# O retorno esperado é Rf + soma de (beta de cada fator x prêmio médio anualizado do fator).
y = carteiras["Máximo Sharpe"] - RF_DIA

def estima(y, fatores):
    res = sm.OLS(y, sm.add_constant(fatores)).fit()
    premios = fatores.mean() * DIAS                          # prêmio anual de cada fator
    esperado = RF_ANUAL + (res.params[1:] * premios).sum()
    return res, esperado

# Proxy de mercado: a planilha NÃO tem o Ibovespa. Uso o retorno médio (pesos iguais) de todas as
# ações da planilha que têm preço em todos os dias.
mercado = dados.dropna(axis=1).pct_change().mean(axis=1).loc[ret.index]
fat_capm = (mercado - RF_DIA).to_frame("Mercado")

res_capm, er_capm = estima(y, fat_capm)
print("\nCAPM (mercado = média das ações da planilha)")
print(f"Beta {res_capm.params['Mercado']:.3f} | Alfa diário {res_capm.params['const']:.5f} | R2 {res_capm.rsquared:.2f}")
print(f"Retorno esperado (CAPM): {er_capm:.2%} ao ano")

# Fama-French: SMB (small minus big) e HML (high minus low) NÃO estão na planilha. Baixe o arquivo
# diário no site do NEFIN, salve como "nefin.csv" na mesma pasta e ajuste os nomes das colunas abaixo
# se o seu arquivo for diferente. Datas na 1ª coluna, fatores em decimal (0.01 = 1%).
try:
    nefin = pd.read_csv("nefin.csv", index_col=0, parse_dates=True)
    nefin = nefin.reindex(ret.index).dropna()
    y_ff = y.loc[nefin.index]
    fat_ff = nefin[["Rm_minus_Rf", "SMB", "HML"]]

    res_ff, er_ff = estima(y_ff, fat_ff)
    res_c2, er_c2 = estima(y_ff, fat_ff[["Rm_minus_Rf"]])      # CAPM com o mesmo mercado do NEFIN (comparação justa)
    print("\nFama-French 3 fatores")
    print(res_ff.params.round(4).to_string())
    print(f"Retorno esperado: CAPM {er_c2:.2%} | Fama-French {er_ff:.2%}")
except FileNotFoundError:
    print("\nFama-French: arquivo nefin.csv não encontrado (SMB e HML não estão na planilha).")

# Intuição da diferença: o CAPM resume o risco em um único fator (o mercado). O Fama-French soma
# dois prêmios: tamanho (SMB) e valor (HML). Como a carteira só tem empresas grandes, o beta de SMB
# tende a ser negativo (puxa o retorno esperado para baixo). O beta de HML depende do mix
# value x growth: PETR4, VALE3 e ITUB4 são mais "value" (HML positivo) e WEGE3 é mais "growth".
# A diferença entre os dois modelos é justamente o efeito desses dois fatores extras.