import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")
df = pd.read_csv("Iris.csv")   # reading the data
df.head()    # first 5 rows
df.drop(["Id"],axis=1,inplace=True)    # dropped


def fuzzy():
    import skfuzzy as fuzz
    X = df["PetalWidthCm"].values
    y = df["PetalLengthCm"].values
    colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'k', 'Brown', 'ForestGreen']
    
    # Set up the loop and plot
    
    alldata = np.vstack((X, y))
    cntr, u_orig, _, _, _, _, _ = fuzz.cluster.cmeans(
        alldata, 3, 2, error=0.005, maxiter=1000)
    
    # Show 3-cluster model
    fig2, ax2 = plt.subplots()
    ax2.set_title('Trained model')
    plt.xlabel("PetalLengthCm")
    plt.ylabel("PetalWidthCm")
    for j in range(3):
        ax2.plot(alldata[0, u_orig.argmax(axis=0) == j],
                 alldata[1, u_orig.argmax(axis=0) == j], 'o',
                 color=colors[j])
    ax2.legend()
    
def kmean():
    features = df.loc[:,["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]]
    from sklearn.cluster import KMeans
    plt.figure(figsize=(24,4))

    plt.suptitle("K Means Clustering",fontsize=20)
    
    
    
    
    # I drop labels since we only want to use features.
    #features.drop(["labels"],axis=1,inplace=True)
    
    plt.subplot(1,5,4)
    plt.title("K = 3",fontsize=16)
    plt.xlabel("PetalLengthCm")
    plt.ylabel("PetalWidthCm")
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
    
    
kmean()
fuzzy()
    