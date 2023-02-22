import numpy as np
import matplotlib.pyplot as plt
from numpy.core.numeric import NaN

# Partie pratique 1

def creer_trajectoire(F,Q,x_init,T):
    vecteur_x = [np.random.multivariate_normal(x_init,P_kalm)]
    for k in range(1,T):
        x_init = vecteur_x[k-1]
        U = np.random.multivariate_normal(4*[0],Q)
        X_k = np.dot(F,x_init) + U
        vecteur_x.append(X_k)
    return(np.array(vecteur_x))

def creer_observations(H,R,vecteur_x,T):
    vecteur_y = []
    for k in range(T):
        V = np.random.multivariate_normal(2*[0],R)
        vecteur_y.append(np.dot(H,vecteur_x[k]) + V)
    return(np.array(vecteur_y))

def filtre_de_kalman(F,Q,H,R,y_k,x_kalm_prec,P_kalm_prec):
    x_kalm = np.dot(F,x_kalm_prec)
    P_kalm = np.dot(np.dot(F,P_kalm_prec),np.transpose(F)) + Q
    S = np.dot(np.dot(H,P_kalm),np.transpose(H)) + R
    K = np.dot(np.dot(P_kalm,np.transpose(H)),np.linalg.inv(S))
    P_kalm_k = np.dot(np.eye(4)-np.dot(K,H),P_kalm)
    x_kalm_k = x_kalm + np.dot(K,y_k-np.dot(H,x_kalm))
    return x_kalm_k,P_kalm_k

def err_quadratique(vecteur_x,x_est):
    T = len(vecteur_x)
    err = []
    for k in range(T):
        err.append(np.dot(np.transpose(vecteur_x[k]-x_est[k]),(vecteur_x[k]-x_est[k])))
    return (T**(-1)*np.sum(np.sqrt(err)))
    
# Application

def filtre_de_kalman2(F,Q,H,R,y_k,x_kalm_prec,P_kalm_prec):
    x_kalm = np.dot(F,x_kalm_prec)
    P_kalm = np.dot(np.dot(F,P_kalm_prec),np.transpose(F)) + Q
    S = np.dot(np.dot(H,P_kalm),np.transpose(H)) + R
    K = np.dot(np.dot(P_kalm,np.transpose(H)),np.linalg.inv(S))
    P_kalm_k = np.dot(np.eye(4)-np.dot(K,H),P_kalm)
    if np.isnan(y_k[0]):
        x_kalm_k = x_kalm 
    else:
        x_kalm_k = x_kalm + np.dot(K,y_k-np.dot(H,x_kalm))
    return x_kalm_k,P_kalm_k

# Partie pratique 2

# r = np.sqrt(px**2+py**2)
# theta = np.arctan(py/px)

# Question 2

def h(x):
    px = x[0]
    py = x[2]
    return np.array([np.arctan(py/px), np.sqrt(px**2+py**2)]).T

def Htilde_radar(x):
    px = x[0]
    py = x[2]
    return np.array(
        [[-py/(px**2+py**2),0,px/(px**2+py**2),0],
        [px/np.sqrt(px**2+py**2),0,py/np.sqrt(px**2+py**2),0]]
    )

def filtre_de_kalman_etendu(F,Q,R,Htilde,y_k,x_kalm_prec,P_kalm_prec):
    x_kalm = np.dot(F,x_kalm_prec)
    Htilde_matrix = Htilde(x_kalm)
    P_kalm = np.dot(np.dot(F,P_kalm_prec),np.transpose(F)) + Q
    Stilde = np.dot(np.dot(Htilde_matrix,P_kalm),np.transpose(Htilde_matrix)) + R
    Ktilde = np.dot(np.dot(P_kalm,np.transpose(Htilde_matrix)),np.linalg.inv(Stilde))
    P_kalm_k = np.dot(np.eye(4)-np.dot(Ktilde,Htilde_matrix),P_kalm)
    x_kalm_k = x_kalm + np.dot(Ktilde,y_k-h(x_kalm))
    return x_kalm_k,P_kalm_k

def creer_observations_radar(R_2,vecteur_x,T):
    vecteur_y = []
    for k in range(T):
        V = np.random.multivariate_normal(2*[0],R_2)
        vecteur_y.append(h(vecteur_x[k]) + V)
    return(np.array(vecteur_y))


if __name__ == '__main__':

    T_e = 1 # période du capteur
    T = 100 # longueur du scénario
    sigma_Q = 1
    sigma_px, sigma_py = 30, 30
    

    x_init = np.array([3,40,-4,20])
    x_kalm = x_init # x0|0
    P_kalm = np.eye((4)) # P0|0

    def Q_function(sigma_Q):
        return sigma_Q**2*np.array([[T_e**3/3,T_e**2/2,0,0],
                [T_e**2/2,T_e,0,0],
                [0,0,T_e**3/3,T_e**2/2],
                [0,0,T_e**2/2,T_e]])

    Q = sigma_Q**2*np.array([[T_e**3/3,T_e**2/2,0,0],
                [T_e**2/2,T_e,0,0],
                [0,0,T_e**3/3,T_e**2/2],
                [0,0,T_e**2/2,T_e]])
    
    F = np.array(
        [[1,T_e,0,0],
        [0,1,0,0],
        [0,0,1,T_e],
        [0,0,0,1]]
    )

    H = np.array(
        [[1,0,0,0],
        [0,0,1,0]]
    )

    def R_function(sigma_px,sigma_py):
        return (np.array(
        [[sigma_px**2,0],
        [0,sigma_py**2]]
    ))

    R = np.array(
        [[sigma_px**2,0],
        [0,sigma_py**2]]
    )

    

    vecteur_x = creer_trajectoire(F,Q,x_init,T)
    vecteur_y = creer_observations(H,R,vecteur_x,T)

    x_kalm_prec = x_kalm
    P_kalm_prec = P_kalm
    x_est = []
    for k in range(T):
        x_kalm_k, P_kalm_k = filtre_de_kalman(F,Q,H,R,vecteur_y[k],x_kalm_prec,P_kalm_prec)
        x_est.append(x_kalm_k)
        x_kalm_prec = x_kalm_k
        P_kalm_prec = P_kalm_k
    x_est = np.array(x_est)

    # plt.plot(vecteur_x[:,0],vecteur_x[:,2],label="vraie trajectoire en $x$",alpha=.8)
    # plt.plot(vecteur_y[:,0],vecteur_y[:,1],'o',label="trajectoire observée en $x$",alpha=.3)
    # plt.xlabel("$x$")
    # plt.ylabel("$y$")
    # plt.show()

    def estime_x(F,Q,R,vecteur_y):
        x_kalm_prec = x_kalm
        P_kalm_prec = P_kalm
        x_est = []
        for k in range(T):
            x_kalm_k, P_kalm_k = filtre_de_kalman(F,Q,H,R,vecteur_y[k],x_kalm_prec,P_kalm_prec)
            x_est.append(x_kalm_k)
            x_kalm_prec = x_kalm_k
            P_kalm_prec = P_kalm_k
        x_est = np.array(x_est)
        return x_est
    
    Q1 = Q_function(100)
    R1 = R_function(30,30)

    # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,4))
    # plt.subplots_adjust(wspace=.4)
    # ax1.plot(x1[:,0],label="vraie trajectoire en $x$",alpha=.8)
    # ax1.plot(y1[:,0],'o',label="trajectoire observée en $x$",alpha=.3)
    # ax1.plot(xest1[:,0],'--',color='red',label="trajectoire estimée en $x$",alpha=1)
    # ax1.set_xlabel("$T$")
    # ax1.set_ylabel("$x$")
    # ax1.legend(loc="upper left",fontsize = 'x-small')
    # ax2.plot(x1[:,2],label="vraie trajectoire en $y$",alpha=.8)
    # ax2.plot(y1[:,1],'o',label="trajectoire observée en $y$",alpha=.3)
    # ax2.plot(xest1[:,2],'--',color='red',label="trajectoire estimée en $y$",alpha=1)
    # ax2.set_xlabel("$T$")
    # ax2.set_ylabel("$y$")
    # ax2.legend(loc="upper left",fontsize = 'x-small')
    # ax3.plot(x1[:,0],x1[:,2],label="vraie trajectoire",alpha=.8)
    # ax3.plot(y1[:,0],y1[:,1],'o',label="trajectoire observée",alpha=.3)
    # ax3.plot(xest1[:,0],xest1[:,2],'--',color='red',label="trajectoire estimée",alpha=1)
    # ax3.set_xlabel("$x$")
    # ax3.set_ylabel("$y$")
    # ax3.legend(loc="upper left",fontsize = 'x-small')
    # plt.show()


    err_moy = []
    n = 100
    for i in range(n):
        vecteur_x = creer_trajectoire(F,Q1,x_init,T)
        vecteur_y = creer_observations(H,R1,vecteur_x,T)

        x_kalm_prec = x_kalm
        P_kalm_prec = P_kalm
        x_est = []
        for k in range(T):
            x_kalm_k, P_kalm_k = filtre_de_kalman(F,Q1,H,R1,vecteur_y[k],x_kalm_prec,P_kalm_prec)
            x_est.append(x_kalm_k)
            x_kalm_prec = x_kalm_k
            P_kalm_prec = P_kalm_k
        x_est = np.array(x_est)
        err_moy.append(err_quadratique(vecteur_x,x_est))

    err_moyenne = n**(-1)*np.sum(err_moy)

    print(err_moyenne)

    with open("data/vecteur_x_avion_ligne.csv") as file_name:
        array_x_ligne = np.genfromtxt(file_name,delimiter=',')

    filling_values = (NaN)
    with open("data/vecteur_y_avion_ligne.csv") as file_name:
        array_y_ligne = np.genfromtxt(file_name,delimiter=',',filling_values=filling_values)

    with open("data/vecteur_x_avion_voltige.csv") as file_name:
        array_x_voltige = np.genfromtxt(file_name,delimiter=',')

    with open("data/vecteur_y_avion_voltige.csv") as file_name:
        array_y_voltige = np.genfromtxt(file_name,delimiter=',',filling_values=filling_values)

    Q1 = Q_function(2)
    Q2 = Q_function(4)
    R1 = R_function(120,20)
    x_kalm_prec = x_kalm
    P_kalm_prec = P_kalm
    x_est_ligne = []
    for k in range(T):
        x_kalm_k, P_kalm_k = filtre_de_kalman2(F,Q1,H,R1,array_y_ligne[k],x_kalm_prec,P_kalm_prec)
        x_est_ligne.append(x_kalm_k)
        x_kalm_prec = x_kalm_k
        P_kalm_prec = P_kalm_k
    x_est_ligne = np.array(x_est_ligne)

    x_kalm_prec = x_kalm
    P_kalm_prec = P_kalm
    x_est_voltige = []
    for k in range(T):
        x_kalm_k, P_kalm_k = filtre_de_kalman2(F,Q2,H,R1,array_y_voltige[k],x_kalm_prec,P_kalm_prec)
        x_est_voltige.append(x_kalm_k)
        x_kalm_prec = x_kalm_k
        P_kalm_prec = P_kalm_k
    x_est_voltige = np.array(x_est_voltige)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8,5))
    plt.subplots_adjust(hspace=.3)
    ax1.plot(array_x_ligne[:,0],array_x_ligne[:,1],alpha=0.8,label="vraie trajectoire")
    ax1.plot(array_y_ligne[:,0],array_y_ligne[:,1],'o',color='orange',alpha=0.2,label="trajectoire observée")
    ax1.plot(x_est_ligne[:,0],x_est_ligne[:,2],'--',color='red',label="trajectoire estimée")
    ax1.set_xlabel("$x$")
    ax1.set_ylabel("Avion de ligne \n $y$")
    ax1.legend()
    ax2.plot(array_x_voltige[:,0],array_x_voltige[:,1],alpha=0.8,label="vraie trajectoire")
    ax2.plot(array_y_voltige[:,0],array_y_voltige[:,1],'o',color='orange',alpha=0.2,label="trajectoire observée")
    ax2.plot(x_est_voltige[:,0],x_est_voltige[:,2],'--',color='red',label="trajectoire estimée")
    ax2.set_xlabel("$x$")
    ax2.set_ylabel("Avion de voltige \n $y$")
    ax2.legend()
    plt.show()

    x_est_ligne2 = []
    for k in range(len(x_est_ligne)):
        x_est_k = [x_est_ligne[k,0],x_est_ligne[k,2]]
        x_est_ligne2.append(x_est_k)
    x_est_ligne = np.array(x_est_ligne2)

    x_est_voltige2 = []
    for k in range(len(x_est_voltige)):
        x_est_k = [x_est_voltige[k,0],x_est_voltige[k,2]]
        x_est_voltige2.append(x_est_k)
    x_est_voltige = np.array(x_est_voltige2)

    print("Erreur quadratique pour l'avion de ligne : ",err_quadratique(array_x_ligne,x_est_ligne))
    print("Erreur quadratique pour l'avion de voltige : ",err_quadratique(array_x_voltige,x_est_voltige))

    # y1 = creer_observations(H,R1,x1,100)
    # xest1 = estime_x(F,Q1,R1,y1)

    def R2_function(sigma_angle,sigma_dist):
        return (np.array(
        [[sigma_angle**2,0],
        [0,sigma_dist**2]]
    ))

    sigma_angle = np.pi/180
    sigma_dist = 100
    R_2 = R2_function(sigma_angle,sigma_dist)
    Q_2 = Q_function(1)

    x1 = creer_trajectoire(F,Q_2,x_init,T)
    
    err_moy = []
    n = 200
    for i in range(n):
        vecteur_y2 = creer_observations_radar(R_2,x1,100)
        x_kalm_prec = x_kalm
        P_kalm_prec = P_kalm
        x_est_radar = []
        for k in range(T):
            x_kalm_k, P_kalm_k = filtre_de_kalman_etendu(F,Q_2,R_2,Htilde_radar,vecteur_y2[k],x_kalm_prec,P_kalm_prec)
            x_est_radar.append(x_kalm_k)
            x_kalm_prec = x_kalm_k
            P_kalm_prec = P_kalm_k
        x_est_radar = np.array(x_est_radar)
        err_moy.append(err_quadratique(x1,x_est_radar))

    err_moyenne = n**(-1)*np.sum(err_moy)


    x_kalm_prec = x_kalm
    P_kalm_prec = P_kalm
    x_est_radar = []
    for k in range(T):
        x_kalm_k, P_kalm_k = filtre_de_kalman_etendu(F,Q_2,R_2,Htilde_radar,vecteur_y2[k],x_kalm_prec,P_kalm_prec)
        x_est_radar.append(x_kalm_k)
        x_kalm_prec = x_kalm_k
        P_kalm_prec = P_kalm_k
    x_est_radar = np.array(x_est_radar)
    print(x_est_radar)

    plt.plot(x1[:,0],x1[:,2],alpha=.8,label="vraie trajectoire")
    plt.plot(vecteur_y2[:,1]*np.cos(vecteur_y2[:,0]),vecteur_y2[:,1]*np.sin(vecteur_y2[:,0]),'o',alpha=.2,label="trajectoire observée")
    plt.plot(x_est_radar[:,0],x_est_radar[:,2],'--',color='red',label="trajectoire estimée")
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.legend()
    plt.show()

