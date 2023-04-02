import cv2
import numpy as np
import matplotlib.pyplot as plt

#Ouverture du flux video
cap = cv2.VideoCapture("Videos/Travelling_OX.m4v")

ret, frame1 = cap.read() # Passe à l'image suivante
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY) # Passage en niveaux de gris
hsv = np.zeros_like(frame1) # Image nulle de même taille que frame1 (affichage OF)
hsv[:,:,1] = 255 # Toutes les couleurs sont saturées au maximum

index = 1
ret, frame2 = cap.read()
next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY) 
fig, ax = plt.subplots()
paused = False

x_min = x_max = 0
y_min = y_max = 0

def analyze_motion(u_mean, u_std, v_mean, v_std):
    std_threshold = 0.5
    mean_threshold = 1.0

    if u_std < std_threshold and v_std < std_threshold:
        return "Fixed plane"
    elif u_std < std_threshold:
        return "Vertical panning (Tilt)"
    elif v_std < std_threshold:
        return "Horizontal panning (Pan)"
    elif abs(u_mean) > mean_threshold or abs(v_mean) > mean_threshold:
        if u_mean > mean_threshold and v_mean > mean_threshold:
            return "Forward travelling"
        elif u_mean < -mean_threshold and v_mean < -mean_threshold:
            return "Backward travelling"
        elif u_mean > mean_threshold:
            return "Horizontal travelling (Right)"
        elif u_mean < -mean_threshold:
            return "Horizontal travelling (Left)"
        elif v_mean > mean_threshold:
            return "Vertical travelling (Up)"
        elif v_mean < -mean_threshold:
            return "Vertical travelling (Down)"
    else:
        return "Unknown"

while(ret):
    index += 1
    flow = cv2.calcOpticalFlowFarneback(prvs,next,None, 
                                        pyr_scale = 0.5,# Taux de réduction pyramidal
                                        levels = 3, # Nombre de niveaux de la pyramide
                                        winsize = 15, # Taille de fenêtre de lissage (moyenne) des coefficients polynomiaux
                                        iterations = 3, # Nb d'itérations par niveau
                                        poly_n = 7, # Taille voisinage pour approximation polynomiale
                                        poly_sigma = 1.5, # E-T Gaussienne pour calcul dérivées 
                                        flags = 0)	
    
    u_mean = flow[:, :, 0].mean()
    u_std = flow[:, :, 0].std()
    v_mean = flow[:, :, 1].mean()
    v_std = flow[:, :, 1].std()

    motion_type = analyze_motion(u_mean, u_std, v_mean, v_std)
    print("Motion type:", motion_type)

    u_bins = 9
    v_bins = 9
    u_ranges = [flow[:,:,0].mean()-3*flow[:,:,0].std(), flow[:,:,0].mean()+3*flow[:,:,0].std()]
    v_ranges = [flow[:,:,1].mean()-3*flow[:,:,1].std(), flow[:,:,1].mean()+3*flow[:,:,1].std()]
    histSize = [u_bins, v_bins]
    ranges = v_ranges + u_ranges
    channels = [1, 0]
    # calculer l'histogramme 2D
    hist2 = cv2.calcHist([flow], channels, None, histSize, ranges, accumulate=False)
    #cv2.normalize(hist2, hist2 )
    ax.clear()
    # ax.imshow(np.log(hist2+1))
    ax.imshow(hist2+1)
    ax.set_title("Histogram")
    plt.draw()
    plt.pause(0.00001)

    mag, ang = cv2.cartToPolar(flow[:,:,0], flow[:,:,1]) # Conversion cartésien vers polaire
    hsv[:,:,0] = (ang*180)/(2*np.pi) # Teinte (codée sur [0..179] dans OpenCV) <--> Argument
    hsv[:,:,2] = (mag*255)/np.amax(mag) # Valeur <--> Norme 

    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    result = np.vstack((frame2,bgr))
    cv2.imshow('Image et Champ de vitesses (Farnebäck)',result)
    k = cv2.waitKey(15) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('Frame_%04d.png'%index,frame2)
        cv2.imwrite('OF_hsv_%04d.png'%index,bgr)
    prvs = next
    ret, frame2 = cap.read()
    if (ret):
        next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY) 

    if k == 32: # ASCII value of spacebar is 32
            paused = not paused

        # if video is paused, wait for spacebar to be pressed again to resume
    while paused:
        key = cv2.waitKey(25)
        if key == 32:
            paused = not paused
plt.imshow(np.log(hist2+1))
plt.show()
cap.release()
cv2.destroyAllWindows()
