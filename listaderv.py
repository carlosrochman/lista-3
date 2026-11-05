# =====================================================================
# LISTA 3 - PARTE 2: DERIVATIVOS (Black-Scholes com sympy)
# Ativo de exemplo: PETR4, com preço e volatilidade tirados da planilha da Economatica
# =====================================================================
import numpy as np
import pandas as pd
import sympy as sp
import matplotlib.pyplot as plt

ARQUIVO = "economatica_5.xlsx"

# ---------------------------------------------------------------------
# 0) PEGAR PREÇO E VOLATILIDADE DA PETR4 NA PLANILHA
# ---------------------------------------------------------------------
# Linha 4 = cabeçalhos, coluna A = datas. Cada coluna tem o ticker na última linha do nome.
dados = pd.read_excel(ARQUIVO, header=3, index_col=0)
dados.columns = [str(c).split("\n")[-1] for c in dados.columns]
petr = dados["PETR4"].replace("-", np.nan).astype(float).dropna()   # feriados ("-") saem

S0    = petr.iloc[-1]                                   # último preço = preço de hoje
sigma = petr.pct_change().std() * np.sqrt(252)          # volatilidade anualizada (histórica)
r     = 0.15                                            # taxa livre de risco anual (Selic ~15%; confira no BCB)
T     = 0.25                                            # prazo: 3 meses (0,25 ano)
K     = round(S0)                                       # strike perto do dinheiro
print(f"PETR4: preço R$ {S0:.2f} | volatilidade {sigma:.1%} | strike R$ {K} | r {r:.0%} | T {T} ano")

# ---------------------------------------------------------------------
# EXERCÍCIO 1 - FUNÇÃO BLACK-SCHOLES POR EXTENSO
# ---------------------------------------------------------------------
# Objetivo: dar o preço justo de uma opção europeia (só exerce no vencimento), sem dividendos.
# Variáveis:
#   S = preço atual da ação      K = strike (preço de exercício)
#   T = tempo até o vencimento (em anos)   r = taxa livre de risco (anual)
#   sigma = volatilidade anual da ação
# N(x) = probabilidade de uma normal padrão ficar abaixo de x. d1 e d2 medem o quão "dentro do dinheiro"
# a opção está, ajustado pela volatilidade e pelo prazo.
S_, K_, T_, r_, sig_ = sp.symbols("S K T r sigma", positive=True)

N  = lambda x: (1 + sp.erf(x / sp.sqrt(2))) / 2               # CDF da normal padrão
d1 = (sp.log(S_ / K_) + (r_ + sig_**2 / 2) * T_) / (sig_ * sp.sqrt(T_))
d2 = d1 - sig_ * sp.sqrt(T_)

call = S_ * N(d1) - K_ * sp.exp(-r_ * T_) * N(d2)             # call = S*N(d1) - K*e^(-rT)*N(d2)
put  = K_ * sp.exp(-r_ * T_) * N(-d2) - S_ * N(-d1)           # put  = K*e^(-rT)*N(-d2) - S*N(-d1)

print("\nd1   = [ln(S/K) + (r + sigma^2/2)*T] / (sigma*sqrt(T))")
print("d2   = d1 - sigma*sqrt(T)")
print("Call = S*N(d1) - K*e^(-rT)*N(d2)")
print("Put  = K*e^(-rT)*N(-d2) - S*N(-d1)")

valores = {S_: S0, K_: K, T_: T, r_: r, sig_: sigma}          # números do nosso exemplo
preco_call = float(call.subs(valores))
preco_put  = float(put.subs(valores))
print(f"Preço da call: R$ {preco_call:.4f} | Preço da put: R$ {preco_put:.4f}")

# ---------------------------------------------------------------------
# EXERCÍCIO 2 - PUT-CALL PARITY
# ---------------------------------------------------------------------
# Relação: Call - Put = S - K*e^(-rT)
# Intuição: comprar uma call e vender uma put com o mesmo strike e vencimento é igual a comprar a
# ação a termo por K. No vencimento, o resultado é (S_T - K) em qualquer cenário. Então hoje
# essa combinação vale o preço da ação menos o valor presente do strike. Se não fosse assim, haveria
# arbitragem (lucro sem risco).
print("\nPut-call parity para vários strikes:")
print("Strike | Call - Put | S - K*e^(-rT) | diferença")
for k in [K - 4, K - 2, K, K + 2, K + 4]:
    v = {**valores, K_: k}
    lado_esq = float((call - put).subs(v))
    lado_dir = float((S_ - K_ * sp.exp(-r_ * T_)).subs(v))
    print(f"{k:6} | {lado_esq:10.4f} | {lado_dir:13.4f} | {lado_esq - lado_dir:.1e}")

# Gráfico: payoff no vencimento de call comprada + put vendida = linha reta (S_T - K), igual a um termo
ST = np.linspace(S0 * 0.6, S0 * 1.4, 100)
payoff_call = np.maximum(ST - K, 0)
payoff_put_vendida = -np.maximum(K - ST, 0)
plt.figure(figsize=(8, 4))
plt.plot(ST, payoff_call, "--", label="Call comprada")
plt.plot(ST, payoff_put_vendida, "--", label="Put vendida")
plt.plot(ST, payoff_call + payoff_put_vendida, "k", lw=2, label="Soma = S_T - K")
plt.axhline(0, color="gray", lw=0.8)
plt.xlabel("Preço da ação no vencimento")
plt.ylabel("Payoff")
plt.title("Put-call parity: call comprada + put vendida = posição a termo")
plt.legend()
plt.show()

# ---------------------------------------------------------------------
# EXERCÍCIO 3 - LETRAS GREGAS (derivadas do preço da opção)
# ---------------------------------------------------------------------
# Delta: quanto o preço da opção muda se a AÇÃO sobe R$ 1. (Call: entre 0 e 1. Put: entre -1 e 0.)
# Gamma: quanto o DELTA muda se a ação sobe R$ 1 (a "curvatura"; é o risco de segunda ordem).
# Theta: quanto a opção perde de valor a cada ano que passa (o "custo do tempo"; em geral negativo).
# Vega:  quanto o preço muda se a VOLATILIDADE sobe 1 (100 pontos percentuais; divida por 100 para 1 p.p.).
# Como T = tempo até o vencimento, o theta é MENOS a derivada em relação a T.
delta_call = sp.diff(call, S_)
delta_put  = sp.diff(put, S_)
gamma      = sp.diff(call, S_, 2)
theta_call = -sp.diff(call, T_)
vega       = sp.diff(call, sig_)

# Fórmulas "de livro" (mais bonitas) com n(x) = densidade da normal padrão:
n = lambda x: sp.exp(-x**2 / 2) / sp.sqrt(2 * sp.pi)
delta_call_f = N(d1)
gamma_f      = n(d1) / (S_ * sig_ * sp.sqrt(T_))
theta_call_f = -S_ * n(d1) * sig_ / (2 * sp.sqrt(T_)) - r_ * K_ * sp.exp(-r_ * T_) * N(d2)
vega_f       = S_ * n(d1) * sp.sqrt(T_)

print("\nDelta (call) = N(d1)")
print("Delta (put)  = N(d1) - 1")
print("Gamma        = n(d1) / (S*sigma*sqrt(T))")
print("Theta (call) = -S*n(d1)*sigma / (2*sqrt(T)) - r*K*e^(-rT)*N(d2)")
print("Vega         = S*n(d1)*sqrt(T)")

# Conferência: a derivada do sympy tem que dar o mesmo número que a fórmula de livro
print("\nConferência numérica (sympy x fórmula de livro):")
for nome, a, b in [("Delta", delta_call, delta_call_f), ("Gamma", gamma, gamma_f),
                   ("Theta", theta_call, theta_call_f), ("Vega", vega, vega_f)]:
    print(f"{nome}: {float(a.subs(valores)):.6f} | {float(b.subs(valores)):.6f}")

# ---------------------------------------------------------------------
# EXERCÍCIO 4 - GRÁFICOS DAS LETRAS GREGAS x PREÇO DA AÇÃO
# ---------------------------------------------------------------------
# lambdify transforma a fórmula do sympy em função numérica rápida (aceita arrays)
args = (S_, K_, T_, r_, sig_)
f_delta_c = sp.lambdify(args, delta_call_f, "scipy")
f_delta_p = sp.lambdify(args, delta_call_f - 1, "scipy")
f_gamma   = sp.lambdify(args, gamma_f, "scipy")
f_theta   = sp.lambdify(args, theta_call_f, "scipy")
f_vega    = sp.lambdify(args, vega_f, "scipy")

S_grade = np.linspace(S0 * 0.6, S0 * 1.4, 200)                 # eixo X: preço da ação
fig, ax = plt.subplots(2, 2, figsize=(12, 8))
ax[0, 0].plot(S_grade, f_delta_c(S_grade, K, T, r, sigma), label="Call")
ax[0, 0].plot(S_grade, f_delta_p(S_grade, K, T, r, sigma), label="Put")
ax[0, 0].set_title("Delta"); ax[0, 0].legend()
ax[0, 1].plot(S_grade, f_gamma(S_grade, K, T, r, sigma)); ax[0, 1].set_title("Gamma")
ax[1, 0].plot(S_grade, f_theta(S_grade, K, T, r, sigma)); ax[1, 0].set_title("Theta (call, por ano)")
ax[1, 1].plot(S_grade, f_vega(S_grade, K, T, r, sigma)); ax[1, 1].set_title("Vega")
for a in ax.flat:
    a.axvline(K, color="gray", ls=":")                          # linha do strike
    a.set_xlabel("Preço da ação")
plt.suptitle(f"Letras gregas x preço da ação (K = {K}, T = {T}, sigma = {sigma:.0%})")
plt.tight_layout()
plt.show()
# Leitura: o delta da call vai de 0 (muito fora do dinheiro) a 1 (muito dentro) e vale ~0,5 no strike.
# O gamma e o vega são máximos perto do strike (é onde a opção é mais "incerta").
# O theta é mais negativo perto do strike: opção no dinheiro é a que mais perde valor com o tempo.

# ---------------------------------------------------------------------
# EXERCÍCIO 5 (DESAFIO) - SIMULAÇÃO DE DELTA-HEDGE
# ---------------------------------------------------------------------
# Cenário: VENDEMOS 1 call e compramos "delta" ações para ficar neutros a pequenas variações do preço.
# A cada passo, reajustamos a quantidade de ações para o novo delta. O que sobra/falta fica em
# caixa, rendendo (ou custando) a taxa r. No vencimento, comparamos o valor da carteira de hedge com o
# que devemos ao dono da call. Se o hedge funciona, o erro fica perto de zero.
f_call = sp.lambdify(args, call, "scipy")

def delta_hedge(passos, caminhos=2000, mu=r, seed=1):
    dt = T / passos
    rng = np.random.default_rng(seed)
    # preços simulados (movimento browniano geométrico): cada coluna é um cenário possível
    z = rng.standard_normal((passos, caminhos))
    S = S0 * np.exp(np.cumsum((mu - sigma**2 / 2) * dt + sigma * np.sqrt(dt) * z, axis=0))
    S = np.vstack([np.full(caminhos, S0), S])                      # linha 0 = preço inicial

    delta = f_delta_c(S[0], K, T, r, sigma)
    caixa = preco_call - delta * S[0]                             # recebo o prêmio e compro delta ações
    for t in range(1, passos):
        caixa *= np.exp(r * dt)                                   # o caixa rende juros
        novo = f_delta_c(S[t], K, T - t * dt, r, sigma)           # delta novo (falta menos tempo)
        caixa -= (novo - delta) * S[t]                            # compra/vende a diferença de ações
        delta = novo
    caixa *= np.exp(r * dt)
    valor_hedge = delta * S[-1] + caixa                           # ações + caixa no vencimento
    divida = np.maximum(S[-1] - K, 0)                             # o que devo ao dono da call
    return S, valor_hedge - divida                                # erro do hedge (em R$)

# a) Um exemplo de trajetória
S_ex, _ = delta_hedge(63, caminhos=1)
plt.figure(figsize=(8, 4))
plt.plot(S_ex, label="Preço simulado")
plt.axhline(K, color="gray", ls=":", label="Strike")
plt.title("Uma trajetória simulada do ativo (63 pregões)")
plt.legend()
plt.show()

# b) Erro do hedge em 2.000 cenários: reajuste diário (63 vezes) x semanal (13 vezes)
plt.figure(figsize=(9, 4))
for passos, nome in [(13, "Semanal (13 reajustes)"), (63, "Diário (63 reajustes)")]:
    _, erro = delta_hedge(passos)
    print(f"{nome}: erro médio R$ {erro.mean():.4f} | desvio padrão R$ {erro.std():.4f}")
    plt.hist(erro, bins=50, alpha=0.6, label=nome)
plt.xlabel("Erro do hedge no vencimento (R$ por opção)")
plt.title("Delta-hedge: quanto mais reajustes, menor o erro")
plt.legend()
plt.show()
# Leitura: o erro médio fica perto de zero (o hedge funciona em média), mas não é exatamente zero
# porque só reajustamos em momentos discretos. Com mais reajustes o erro encolhe. Na prática,
# custos de transação e volatilidade que muda ao longo do tempo também fazem o hedge falhar.