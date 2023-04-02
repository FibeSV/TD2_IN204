import cv2
import numpy as np
import matplotlib.pyplot as plt


cap = cv2.VideoCapture('Videos/Extrait1-Cosmos_Laundromat1(340p).m4v')
hist2 = None
dists = []
dists2 = []
dists3 = []
dists4 = []
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
        u_bins = 30
        v_bins = 30
        u_ranges = [0, 256]
        v_ranges = [0, 256]
        histSize = [u_bins, v_bins]
        ranges = u_ranges + v_ranges
        channels = [1, 2]
        # calculer l'histogramme 2D
         
        hist1 = hist2
        hist2 = cv2.calcHist([yuv], channels, None, histSize, ranges, accumulate=False)
        
        
        ax.clear()
        ax.imshow(np.log(hist2+1))
        ax.set_title("Histogram")
        plt.draw()
        plt.pause(0.00001)

        cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.imshow('Image', frame)

        # compare histograms
        if hist1 is None:
           continue
        else: 
            dist = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
            dist2 = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR )
            dist3 = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL )
            dist4 = cv2.compareHist(hist1, hist2, cv2.HISTCMP_INTERSECT )
            dists.append(dist)
            dists2.append(dist2)
            dists3.append(dist3)
            dists4.append(dist4)
            print(dist)
            if dist > 0.5:
                view_changed.append({"dist":dist,"frame1":frame1,"frame":frame})

        # ax.clear()
        # ax.plot(dists)
        # plt.draw()
        # plt.pause(0.00001)
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
dists = np.array(dists)
dists2 = np.array(dists2)
dists3 = np.array(dists3)
dists4 = np.array(dists4)
dists = (dists-dists.min())/(dists.max()-dists.min())
dists2 = (dists2-dists2.min())/(dists2.max()-dists2.min())
dists3 = 1-(dists3-dists3.min())/(dists3.max()-dists3.min())
dists4 = (dists4-dists4.min())/(dists4.max()-dists4.min())
plt.subplot(511)
plt.plot(np.arange(0,len(dists)/fps, 1./fps)[:len(dists)],dists, label='Bhattacharyya ')
plt.plot(np.arange(0,len(dists)/fps, 1./fps)[:len(dists)],dists2, label='Chi-Square ')
plt.plot(np.arange(0,len(dists)/fps, 1./fps)[:len(dists)],dists3, label='Correlation ')
plt.plot(np.arange(0,len(dists)/fps, 1./fps)[:len(dists)],dists4, label='Intersection ')
plt.xlabel('Time')
plt.ylabel('différence')
plt.yticks(np.arange(0,1,0.1))
plt.xticks(np.arange(0,(len(dists)+1)/fps,5))
plt.grid(True)
plt.legend()
plt.title("différence entre deux images adjacentes")
plt.subplot(512)
plt.plot(np.arange(0,len(dists)/fps, 1./fps)[:len(dists)],dists, label='Bhattacharyya ')
plt.xlabel('Time')
plt.ylabel('différence')
plt.yticks(np.arange(0,1,0.1))
plt.xticks(np.arange(0,(len(dists)+1)/fps,5))
plt.grid(True)
plt.legend()
plt.subplot(513)
plt.plot(np.arange(0,len(dists)/fps, 1./fps)[:len(dists)],dists2, label='Chi-Square ')
plt.xlabel('Time')
plt.ylabel('différence')
plt.yticks(np.arange(0,1,0.1))
plt.xticks(np.arange(0,(len(dists)+1)/fps,5))
plt.grid(True)
plt.legend()
plt.subplot(514)
plt.plot(np.arange(0,len(dists)/fps, 1./fps)[:len(dists)],dists3, label='Correlation ')
plt.xlabel('Time')
plt.ylabel('différence')
plt.yticks(np.arange(0,1,0.1))
plt.xticks(np.arange(0,(len(dists)+1)/fps,5))
plt.grid(True)
plt.legend()
plt.subplot(515)
plt.plot(np.arange(0,len(dists)/fps, 1./fps)[:len(dists)],dists4, label='Intersection ')
plt.xlabel('Time')
plt.ylabel('différence')
plt.yticks(np.arange(0,1,0.1))
plt.xticks(np.arange(0,(len(dists)+1)/fps,5))
plt.grid(True)
plt.legend()
plt.show()


for view in view_changed[:]:
    paused = True
    print("dist=",view["dist"])
    cv2.imshow("frame1",view["frame1"])
    cv2.imshow("frame",view["frame"])
    while paused:
            key = cv2.waitKey(25)
            if key == 32:
                paused = not paused