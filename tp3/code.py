
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import datetime as dt
import time
import math

from scipy.optimize import newton

import itertools  # pour construire un iterator pour une def de matrice question 4 et 5

df = pd.read_csv("C:\\Users\\Wenceslas\\Desktop\\ijup_cours\\pyt_avance\\finance\\fi2\\tp3\\data_tp3.csv", sep=';')
df.head()

germany_data = df[df["country"] == "Germany"]
french_data = df[df["country"] != "Germany"]

mpl.style.use('seaborn')
fig, ax = plt.subplots(figsize=(9, 6))
plt.grid(b=True)

ax.plot(germany_data["Maturity"]
        , germany_data["rates"]
        , marker='.'
        , markerfacecolor='black'
        , markersize=9
        , c='blue'
        , label="Courbe des Taux Allemande")

ax.plot(french_data["Maturity"]
        , french_data["rates"]
        , marker='.'
        , markerfacecolor='black'
        , markersize=9
        , c='orangered'
        , label="Courbe des Taux Française")

ax.set_ylabel("Taux", rotation=0, fontsize=12)
ax.set_xlabel("Maturite", fontsize=12)
ax.set_title("Courbe des taux pour les obligations ", fontsize=18)
ax.legend()

plt.show()



securities = pd.DataFrame({
    "Titre": ["Compagnie des Alpes", "Korian", "Carrefour", "Total"],
    "Coupon": [0.03504, 0.037, 0.01, 0.01023],
    "Prix": [113, 110.23, 102.22, 106.57],
    "Maturite": pd.to_datetime(["24/10/2027", "07/10/2025", "27/10/2027", "27/10/2027"],
                               format='%d/%m/%Y')
})

date_aujourdhui = dt.datetime.strptime("25/10/2019", '%d/%m/%Y')
securities["Maturite"] = securities["Maturite"].apply(lambda x: (x.year - date_aujourdhui.year))

print(securities)

# Construction du portefeuille CASH FLOW
portefeuille_cashflow = pd.DataFrame({
    "Maturite": [i for i in range(1, np.max(securities["Maturite"].values) + 1)],
    "cash_flow": np.zeros(np.max(securities["Maturite"].values))
})
for i, val in securities.iterrows():
    for mat in range(val["Maturite"]):
        if mat != val["Maturite"] - 1:
            portefeuille_cashflow["cash_flow"][mat] += val["Coupon"] * 100
        else:
            portefeuille_cashflow["cash_flow"][mat] += 100 + val["Coupon"] * 100
print(portefeuille_cashflow)

# Ce que nous a coûté la construction de ce portefeuille, c'est la somme des prix de nos actifs
prix_portefeuille = securities["Prix"].sum()
print(prix_portefeuille)
print(portefeuille_cashflow)


def f_cashflow(x, df, p, n=100):
    somme = 0

    for i in portefeuille_cashflow.index:
        C = portefeuille_cashflow.loc[i, "cash_flow"]
        somme += C * pow(x, i)
    return p - somme


def fprime_cashflow(x, df, p, n=100):
    somme = 0

    for i in portefeuille_cashflow.index:
        C = portefeuille_cashflow.loc[i, "cash_flow"]
        somme += i * C * pow(x, i - 1)
    return - somme


def yield_to_mat_cash_flow(f, x0, fprime, args):
    new = newton(func=f, x0=x0, fprime=fprime, args=args)
    return (1 / new) - 1


portefeuille_y = yield_to_mat_cash_flow(f_cashflow, 1, fprime_cashflow,
                                        (portefeuille_cashflow, prix_portefeuille))

print(portefeuille_y)


def yield_to_maturity(P, c, T, N):
    """
    Renvoie le yield to maturity d'une obligation. Paramètres :
        - P : le prix de l'obligation
        - c : le taux de coupon de l'obligation
        - T : la maturité de l'obligation
        - N : le nominal de l'obligation
    """

    def f(x, P, c, T, N):
        """
        Approxime la fonction de cash flows. Paramètres :
            - x : la variable permettant l'optimisation par Newton
            - P : le prix de l'obligation
            - c : le taux de coupon de l'obligation
            - T : la maturité de l'obligation
            - N : le nominal de l'obligation
        """
        result = P
        for i in range(1, int(T)):
            result -= c * N * (x ** i)
        result -= N * (1 + c) * (x ** T)
        return result

    def fprime(x, P, c, T, N):
        """
        Dérivée de l'approximation de la fonction de cash flows. Paramètres :
            - x : la variable permettant l'optimisation par Newton
            - P : le prix de l'obligation
            - c : le taux de coupon de l'obligation
            - T : la maturité de l'obligation
            - N : le nominal de l'obligation
        """
        result = 0
        for i in range(1, int(T)):
            result -= i * c * N * (x ** (i - 1))
        result -= T * N * (1 + c) * (x ** (T - 1))
        return result

    x = newton(func=f,
               x0=1.1,
               # puisque y est de l'ordre 0.01 et que x = 1/(1+y) 1.1 semble etre un ordre de grandeur raisonable
               fprime=fprime,
               args=(P, c, T, N)
               )
    y = 1 / x - 1
    return y


securities["Yield"] = securities.apply(lambda row: yield_to_maturity(row[2], row[1], row[3], 100), axis=1)

print(securities)

print(portefeuille_cashflow)


def duration(df, y, n=100):
    somme = 0
    for i in portefeuille_cashflow.index:
        C = portefeuille_cashflow.loc[i, "cash_flow"]
        m = portefeuille_cashflow.loc[i, "Maturite"]
        somme -= m * C / pow(1 + y, m + 1)
    return somme  # somme -(m*(n+C))/pow(1+y, (m+1))


print(duration(portefeuille_cashflow, portefeuille_y))
print("\nMC:")
print(duration(portefeuille_cashflow, portefeuille_y) / prix_portefeuille)

def duration_i(y, C, m, n=100):
    return -(m * (1)) / pow(1 + y, (m + 1))


french_data["Duration_i"] = french_data.apply(lambda row: duration_i(y=row[1], C=1, m=row[0], n=1)
                                              , axis=1)
germany_data["Duration_i"] = germany_data.apply(lambda row: duration_i(y=row[1], C=1, m=row[0], n=1)
                                                , axis=1)

french_data2 = french_data[np.logical_and(french_data["Maturity"] <= 8, french_data["Maturity"] >= 1)].reset_index(
    drop=True, inplace=False)
germany_data2 = germany_data[np.logical_and(germany_data["Maturity"] <= 8, germany_data["Maturity"] >= 1)].reset_index(
    drop=True, inplace=False)

french_data2["Duration_i"] = french_data2["Duration_i"] * portefeuille_cashflow["cash_flow"]
germany_data2["Duration_i"] = germany_data2["Duration_i"] * portefeuille_cashflow["cash_flow"]

print(french_data2)


print(germany_data2["Duration_i"].sum())
print(french_data2["Duration_i"].sum())
print(germany_data2)

plt.plot(germany_data2["Maturity"], germany_data2["Duration_i"], label="Germany", c="orange")
plt.plot(french_data2["Maturity"], french_data2["Duration_i"], label="France")
plt.legend()
plt.show()


def matrice_d_construction(df):
    """
    Me permet d'afficher une matrice triangulaire sup avec les cash_flow dans l'ordres de maturité
    Pour la 2 ème colonne, le cash flow s'arrète à partir de la
    troisième ligne, avec une suite de 0.

    C'est un traitement nécessaire pour construire la matrice de dérivé par rapport à tous les taux
    obligataires

    Voici un exemple d'ouput:
                1.0	2.0	3.0	4.0	5.0	6.0	7.0	8.0
        0	100.0	0.0	0.0	0.0	0.0	1.0	0.25	2.75
        1	0.0	100.0	0.0	0.0	0.0	1.0	0.25	2.75
        2	0.0	0.0	100.0	0.0	0.0	1.0	0.25	2.75
        3	0.0	0.0	0.0	100.0	0.0	1.0	0.25	2.75
        4	0.0	0.0	0.0	0.0	100.0	1.0	0.25	2.75
        5	0.0	0.0	0.0	0.0	0.0	101.0	0.25	2.75
        6	0.0	0.0	0.0	0.0	0.0	0.0	100.25	2.75
        7	0.0	0.0	0.0	0.0	0.0	0.0	0.00	102.75
    """
    dicto = {i: np.array(np.repeat(df[df["Maturity"] == i]["coupon"], df.shape[0])) for i in df["Maturity"].values}

    test_df = pd.DataFrame.from_dict(dicto)
    test_matrix = pd.DataFrame.from_dict(dicto).to_numpy()
    # je rempli ma diagonal par la valeur du nominal (ici n=100)
    np.fill_diagonal(test_matrix, test_matrix.diagonal() + 100)
    df_out = pd.DataFrame(test_matrix, columns=[i for i in df["Maturity"].values])
    colname = df_out.columns
    # je met toutes les valeurs en dessous de la diagonale à 0
    return pd.DataFrame(np.triu(df_out, 0), columns=colname)


print(matrice_d_construction(french_data2))


def duration_i(y, C, m):
    return -(m * (C)) / pow(1 + y, (m + 1))


def matrice_d(df):
    """
    Sortie est la matrice n*n contenant la dérivé des n titres par rapports à leurs
    n taux
    -df : DataFrame qui doit contenir les colonnes suivantes: Maturity et coupon
    (attention aux majuscules)
    """
    try_df = matrice_d_construction(df)

    iterator = itertools.product(df["Maturity"].index,
                                 df["Maturity"].values)

    df_copy = df.copy()
    df_copy = df_copy.reset_index(drop=True, inplace=False)

    listed_stock = []
    for ite in iterator:
        m = ite[1]
        idx = ite[0]
        r = df_copy["rates"][idx]
        listed_stock.append(duration_i(r, try_df[m][idx], m))

    matrice_d_out = np.array(listed_stock).reshape(df.shape[0], df.shape[0])

    return matrice_d_out


def poids_couverture(matrice_d, vector_d):
    """
    Ressort le poids pour nous protéger contre les variations de la courbe des taux
    """
    return -np.dot(vector_d, np.linalg.inv(matrice_d))


french_data_18 = french_data2[np.logical_or(french_data2["Maturity"] == 1, french_data2["Maturity"] == 8)]
french_data_18 = french_data_18.reset_index(drop=True)


matrice_d_fr_18 = matrice_d(french_data_18)
vecteur_d_fr_18 = np.array(french_data_18["Duration_i"].values)
poids_couverture_18ans = poids_couverture(matrice_d_fr_18, vecteur_d_fr_18)
print(poids_couverture_18ans)


matrice_d_fr = matrice_d(french_data2)
print(matrice_d_fr)
vecteur_d_fr = np.array(french_data2["Duration_i"].values)
poids = np.round(poids_couverture(matrice_d_fr, vecteur_d_fr), 5)
print(poids)


def taux_vasicek_disc(r0, a, b, sig, h, k):
    r = [r0]
    i = 0
    for i in range(0, k):
        alea = sig * np.sqrt(h) * np.random.normal(0, 1, 1)
        retour = a * (b - r[i]) * h
        r.append(r[i] + retour + alea)
    return r


np.random.seed(55)
data1 = taux_vasicek_disc(0.01, 0.02, 0.005, 0.02, 1 / 12, 36)
print(data1)

mpl.style.use('seaborn')
fig, ax = plt.subplots()

ax.plot([i for i in range(len(data1))]
        , np.ravel(data1)
        , marker='.'
        , markerfacecolor='black'
        , markersize=9
        , c='orange'
        )
b = 0.005
ax.plot([i for i in range(len(data1))]
        , [b for i in range(len(data1))]
        , label="Moyenne de Long Terme"
        )

ax.set_ylabel("Taux", rotation=0, fontsize=12)
ax.set_xlabel("Maturite", fontsize=12)
ax.set_title("Courbe de taux", fontsize=18)
ax.legend()

plt.show()


np.random.seed(5)
data2 = taux_vasicek_disc(0.01, 0.1, 0.005, 0.02, 1 / 12, 36)
data2

mpl.style.use('seaborn')
fig, ax = plt.subplots()

ax.plot([i for i in range(len(data2))]
        , np.ravel(data2)
        , marker='.'
        , markerfacecolor='black'
        , markersize=9
        , c='orange'
        )
b = 0.005
ax.plot([i for i in range(len(data2))]
        , [b for i in range(len(data2))]
        , label="Moyenne de Long Terme"
        )

ax.set_ylabel("Taux", rotation=0, fontsize=12)
ax.set_xlabel("Maturite", fontsize=12)
ax.set_title("Courbe de taux", fontsize=18)
ax.legend()

plt.show()


def vasicek_new(T, b, r_0, a, sigma):
    """
    Retourne la valeur en T de la courbe de taux simulée par la nouvelle équation fournie après Vasicek.
    """
    z_infty = b - (sigma ** 2) / (2 * a ** 2)
    s = r_0 - z_infty
    phi_T = (1 - math.exp(-a * T)) / a
    z_T = z_infty + s * phi_T / T + sigma ** 2 / (4 * a ** 3) * phi_T ** 2 / T
    return z_T


r0 = -0.005
a = 0.01
b = 0
sig = 0.001
dico_courbe_taux = {
    i: vasicek_new(i, b, r0, a, sig) for i in range(1, 11)
}
print(dico_courbe_taux)

data = list(dico_courbe_taux.items())

mpl.style.use('seaborn')
fig, ax = plt.subplots()

ax.plot([i[0] for i in data]
        , [i[1] for i in data]
        , marker='.'
        , markerfacecolor='black'
        , markersize=9
        , c='orange'
        )

ax.set_ylabel("Taux", rotation=0, fontsize=12)
ax.set_xlabel("Maturite", fontsize=12)
ax.set_title("Courbe de taux", fontsize=18)

plt.show()


def derivative_z_infty(CF_T, T, b, r_0, a, sigma):
    """
    Returns the derivative of V_T relative to z_infty.
    """
    phi_T = (1 - math.exp(-a * T)) / a
    return CF_T * T * (phi_T / T - 1) * math.exp(-T * vasicek_new(T, b, r_0, a, sigma))


def derivative_s(CF_T, T, b, r_0, a, sigma):
    """
    Returns the derivative of V_T relative to s.
    """
    phi_T = (1 - math.exp(-a * T)) / a
    return CF_T * -T * phi_T / T * math.exp(-T * vasicek_new(T, b, r_0, a, sigma))



def derivative_sigma_2(CF_T, T, b, r_0, a, sigma):
    """
    Returns the derivative of V_T relative to s.
    """
    return CF_T * -T * (-1 / (2 * a ** 2) + 1 / (4 * a ** 3)) * math.exp(-T * vasicek_new(T, b, r_0, a, sigma))


derivative_z_infty(100, 10, b, r0, a, sig)

derivative_sigma_2(100, 1, b, r0, a, sig)

derivative_sigma_2(9.227, 1, b, r0, a, sig)

v2 = derivative_z_infty(100, 1, b, r0, a, sig)
v1 = derivative_z_infty(9.227, 1, b, r0, a, sig)
print(-v1 / v2)


sigma_c = derivative_sigma_2(100.25, 7, b, r0, a, sig)
sigma_p = derivative_sigma_2(5.527, 7, b, r0, a, sig)
poids_c = -sigma_p / sigma_c
print("Il faut {} pour se convrir contre le risque de concavité".format(poids_c))

pente_lambda = derivative_s(100, 5, b, r0, a, sig)
pente_p = derivative_s(9.227, 5, b, r0, a, sig)
pente_c = derivative_s(0.25, 5, b, r0, a, sig)
poids_lambda = -(pente_p + pente_c * poids_c) / pente_lambda
print("\nIl faut {} pour se convrir contre le risque de pente".format(poids_lambda))

zoo_rho = derivative_z_infty(100, 1, b, r0, a, sig)
zoo_p = derivative_z_infty(9.227, 1, b, r0, a, sig)
zoo_c = derivative_z_infty(0.25, 1, b, r0, a, sig)
zoo_lambda = derivative_z_infty(0, 1, b, r0, a, sig)
poids_rho = -(zoo_p + poids_lambda * zoo_lambda + poids_c * zoo_c) / zoo_rho
print("\nIl faut {} pour se convrir contre le risque de shift".format(poids_rho))

C:\Users\Wenceslas\PycharmProjects\test\last.py