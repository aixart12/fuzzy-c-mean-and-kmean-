import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")
df = pd.read_csv("Iris.csv")   # reading the data
df.head()    # first 5 rows
df.drop(["Id"],axis=1,inplace=True)    # dropped


#---------------------------------------------------------------------------------------------------------------


#fuzzy c mean clustering begain here
def FuzzyClustering():
    
    import skfuzzy as fuzz
    X = df["PetalWidthCm"].values
    y = df["PetalLengthCm"].values
    
    
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/3,random_state=42)
    
    X_test= X_test[:,None]
    y_test= y_test[:,None]
    
    
    
    colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'k', 'Brown', 'ForestGreen']
    # Set up the loop and plot
    fig1, axes1 = plt.subplots(3, 3, figsize=(8, 8))
    alldata = np.vstack((X_train, y_train))
    fpcs = []
    
    for ncenters, ax in enumerate(axes1.reshape(-1), 2):
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            alldata, ncenters, 2, error=0.005, maxiter=1000, init=None)
    
        # Store fpc values for later
        fpcs.append(fpc)
    
        # Plot assigned clusters, for each data point in training set
        cluster_membership = np.argmax(u, axis=0)
        for j in range(ncenters):
            ax.plot(X_train[cluster_membership == j],
                    y_train[cluster_membership == j], '.', color=colors[j])
    
        # Mark the center of each fuzzy cluster
        for pt in cntr:
            ax.plot(pt[0], pt[1], 'rs')
    
    
    
    fig1.tight_layout()
    
    fig2, ax2 = plt.subplots()
    ax2.plot(np.r_[2:11], fpcs)
    ax2.set_xlabel("Number of centers")
    ax2.set_ylabel("Fuzzy partition coefficient")
    
    
    cntr, u_orig, _, _, _, _, _ = fuzz.cluster.cmeans(
        alldata, 3, 2, error=0.005, maxiter=1000)
    
    # Show 3-cluster model
    fig2, ax2 = plt.subplots()
    ax2.set_title('Trained model')
    for j in range(3):
        ax2.plot(alldata[0, u_orig.argmax(axis=0) == j],
                 alldata[1, u_orig.argmax(axis=0) == j], 'o',
                 label='series ' + str(j))
    ax2.legend()
    
    
    # Generate uniformly sampled data spread across the range [0, 10] in x and y
    
    
    newdata = np.concatenate((X_test,y_test),axis=1)
    
    # Predict new cluster membership with `cmeans_predict` as well as
    # `cntr` from the 3-cluster model
    u, u0, d, jm, p, fpc = fuzz.cluster.cmeans_predict(
        newdata.T, cntr, 2, error=0.005, maxiter=1000)
    
    # Plot the classified uniform data. Note for visualization the maximum
    # membership value has been taken at each point (i.e. these are hardened,
    # not fuzzy results visualized) but the full fuzzy result is the output
    # from cmeans_predict.
    cluster_membership = np.argmax(u, axis=0)  # Hardening for visualization
    
    fig3, ax3 = plt.subplots()
    ax3.set_title('Random points classifed according to known centers')
    for j in range(3):
        ax3.plot(newdata[cluster_membership == j, 0],
                 newdata[cluster_membership == j, 1], 'o',
                 label='series ' + str(j))
    ax3.legend()
    
    plt.show()







#-----------------------------------------------------------------------------------------------------------
def kmeanClustering():
    sns.pairplot(data=df,hue="Species",palette="Set2")
    plt.show()
    features = df.loc[:,["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]]
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=99)  
    
    
    
    #kmeans 
    from sklearn.cluster import KMeans
    wcss = []
    
    for k in range(1,15):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(features)
        wcss.append(kmeans.inertia_)
    
    
    plt.figure(figsize=(20,8))
    plt.title("WCSS / K Chart", fontsize=18)
    plt.plot(range(1,15),wcss,"-o")
    plt.grid(True)
    plt.xlabel("Amount of Clusters",fontsize=14)
    plt.ylabel("Inertia",fontsize=14)
    plt.xticks(range(1,20))
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(24,4))
    
    plt.suptitle("K Means Clustering",fontsize=20)
    
    
    
    # I drop labels since we only want to use features.
    #features.drop(["labels"],axis=1,inplace=True)
    
    plt.subplot(1,5,4)
    plt.title("K = 3",fontsize=16)
    plt.xlabel("PetalLengthCm")
    kmeans = KMeans(n_clusters=3)
    features["labels"] = kmeans.fit_predict(features)
    plt.scatter(features.PetalLengthCm[features.labels == 0],features.PetalWidthCm[features.labels == 0])
    plt.scatter(features.PetalLengthCm[features.labels == 1],features.PetalWidthCm[features.labels == 1])
    plt.scatter(features.PetalLengthCm[features.labels == 2],features.PetalWidthCm[features.labels == 2])
    

    
    # I drop labels since we only want to use features.
    #features.drop(["labels"],axis=1,inplace=True)
    
    plt.subplot(1,5,5)
    plt.title("Original Labels",fontsize=16)
    plt.xlabel("PetalLengthCm")
    plt.scatter(df.PetalLengthCm[df.Species == "Iris-setosa"],df.PetalWidthCm[df.Species == "Iris-setosa"])
    plt.scatter(df.PetalLengthCm[df.Species == "Iris-versicolor"],df.PetalWidthCm[df.Species == "Iris-versicolor"])
    plt.scatter(df.PetalLengthCm[df.Species == "Iris-virginica"],df.PetalWidthCm[df.Species == "Iris-virginica"])
    
    plt.subplots_adjust(top=0.8)
    plt.show()
    
    # I drop labels since we only want to use features.
    #features.drop(["labels"],axis=1,inplace=True)
    
    # kmeans
    kmeans = KMeans(n_clusters=3)
    kmeans_predict = kmeans.fit_predict(features)
    
    # cross tabulation table for kmeans
    df1 = pd.DataFrame({'labels':kmeans_predict,"Species":df['Species']})
    ct1 = pd.crosstab(df1['labels'],df1['Species'])
    
    
FuzzyClustering()
kmeanClustering()

    


