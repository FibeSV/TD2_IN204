import cv2
import numpy as np
import matplotlib.pyplot as plt


cap = cv2.VideoCapture('Videos/Extrait1-Cosmos_Laundromat1(340p).m4v')
hist2 = None
dists = []
view_changed = []
paused = False
fig, ax = plt.subplots()
fps = cap.get(cv2.CAP_PROP_FPS)

# set new frame rate (double the original)
new_fps = fps * 32
cap.set(cv2.CAP_PROP_FPS, new_fps)
frame=None
frame1=None
while(cap.isOpened()):
    frame1 = frame 
    ret, frame = cap.read()
    if ret == True:
        # convertir l'image en espace de couleur YUV
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        # extraire les composantes u et v
        u_bins = 100 
        v_bins = 120
        u_ranges = [0, 256]
        v_ranges = [0, 256]
        histSize = [u_bins, v_bins]
        ranges = u_ranges + v_ranges
        channels = [1, 2]
        # calculer l'histogramme 2D
         
        hist1 = hist2
        hist2 = cv2.calcHist([yuv], channels, None, histSize, ranges, accumulate=False)
        cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.imshow('Histogramme 2D', hist2)
        print(hist2)
        cv2.imshow('Image', frame)

        # compare histograms
        if hist1 is None:
           continue
        else: 
            dist = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
            dists.append(dist)
            print(dist)
            if dist > 0.45:
                view_changed.append({"dist":dist,"frame1":frame1,"frame":frame})

        ax.clear()
        ax.plot(dists)
        plt.draw()
        plt.pause(0.00001)
        # plt.plot(dists)
        # plt.show()    
        # attendre une touche de clavier
        key = cv2.waitKey(25)

        # if spacebar is pressed, pause video
        if key == 32: # ASCII value of spacebar is 32
            paused = not paused

        # if video is paused, wait for spacebar to be pressed again to resume
        while paused:
            key = cv2.waitKey(25)
            if key == 32:
                paused = not paused

        # if 'q' is pressed, exit loop
        if key == ord('q'):
            break
    else:
        break
plt.plot(np.arange(0,len(dists)/fps, 1./fps)[:len(dists)],dists)
plt.xlabel('Time')
plt.ylabel('différence')
plt.yticks(np.arange(0,1,0.1))
plt.xticks(np.arange(0,(len(dists)+1)/fps,5))
plt.grid(True)
plt.title("différence entre deux images adjacentes")
plt.show()
for view in view_changed[:0]:
    paused = True
    print("dist=",view["dist"])
    cv2.imshow("frame1",view["frame1"])
    cv2.imshow("frame",view["frame"])
    while paused:
            key = cv2.waitKey(25)
            if key == 32:
                paused = not paused