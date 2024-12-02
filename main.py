import string
import numpy as np

def zbuduj_mtx(teksty, frazy):
    unikalne = sorted(set(slowo for tekst in teksty for slowo in tekst.split()))    
    mac = []
    
    for tekst in teksty:
        zbior = set(tekst.split())
        wiersz = [1 if slowo in zbior else 0 for slowo in unikalne]
        mac.append(wiersz)
    
    frazy_vec = [1 if slowo in unikalne and slowo in frazy else 0 for slowo in unikalne]

    return mac, frazy_vec

def operuj_mtx(mtx, wym, zap):
    mtx = mtx.T
    mac_U, wart_s, mac_Vt = np.linalg.svd(mtx, full_matrices=False)

    wym_r = wym
    wart_redu = np.copy(wart_s)
    wart_redu[wym_r:] = 0
    
    mac_Sr = np.diag(wart_redu)
    mtx_redu = mac_Sr.dot(mac_Vt)

    wart_top = np.take(wart_s, range(wym_r), axis=0)
    mac_top = np.diag(wart_top)

    mac_Vk = np.take(mac_Vt, range(wym_r), axis=0)
    mtx_redu = mac_top.dot(mac_Vk)

    zap = zap.T
    mac_top_inv = np.linalg.inv(mac_top)
    mac_UkT = np.take(mac_U.T, range(wym_r), axis=0)

    zap_redu = mac_top_inv.dot(mac_UkT).dot(zap)

    return mtx_redu, zap_redu

def cos_sim(redu_mtx, redu_q):
    q_norm = np.sqrt(np.sum(redu_q**2))
    m_norm = [np.sqrt(np.sum(kol**2)) for kol in redu_mtx.T]
    mian = [m_norm[i] * q_norm for i in range(len(m_norm))]

    licz = []
    for kol in redu_mtx.T:
        wynik = np.sum(kol * redu_q)
        licz.append(wynik)

    wyniki = [float(round(licz[i]/mian[i], 2)) for i in range(len(mian))]

    return wyniki

def main():
    ile_dok = int(input())

    doki = []

    for _ in range(ile_dok):
        doki.append(input().translate(str.maketrans('', '', string.punctuation)).strip().lower())

    zapyt = input().split(" ")

    wymiar = int(input())

    macierz, zap_vec = zbuduj_mtx(doki, zapyt)

    redu_mtx, redu_q = operuj_mtx(np.array(macierz), wymiar, np.array(zap_vec))

    print(cos_sim(redu_mtx, redu_q))

main()
