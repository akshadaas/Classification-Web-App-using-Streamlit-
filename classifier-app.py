# -*- coding: utf-8 -*-


import streamlit as st
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_precision_recall_curve,plot_confusion_matrix,plot_roc_curve
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
st.set_option('deprecation.showPyplotGlobalUse', False)
def get_dataset(dataset_name):
    if dataset_name == "IRIS":
        data = datasets.load_iris()
    elif dataset_name == "Breast cancer":
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()
    X = data.data
    y = data.target
    
    return X, y, data

def get_classifier(clf):
    if clf == "Random Forest":
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators = st.sidebar.slider("n_estimators",100,300)
        max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 20, step=1, key='n_estimators')
        bootstrap = st.sidebar.radio("Bootstrap samples when building trees", ('True', 'False'), key='bootstrap')
       
        clf = RandomForestClassifier(n_estimators = n_estimators,max_depth =max_depth, bootstrap = bootstrap, random_state=0)
    elif clf == "Support Vector Machines(SVM)":
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C_SVM')
        kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key='kernel')
        gamma = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"), key='gamma')
        clf = SVC(C=C)
    elif clf=="K-Nearest Neighbors(KNN)":
        st.sidebar.subheader("Model Hyperparameters")
        k = st.sidebar.slider('KNN', 1, 10)
        clf = KNeighborsClassifier(n_neighbors=k)
    return clf

#Plot
def plot_metrics(metrics_list,model,X,y,X_test,y_test):
        if 'Confusion Matrix' in metrics_list:
            fig = plt.figure()
            st.subheader("Confusion Matrix")
            plot_confusion_matrix(model,X_test,y_test,display_labels=model.classes_)   
            #plt.colorbar()                    
            st.pyplot()
        
        if 'Data-visualization' in metrics_list:
            fig,ax = plt.subplots()
            st.subheader('Data-visualization')
            pca = PCA(2)
            X_proj = pca.fit_transform(X)
            
            x1 = X_proj[:,0]
            x2 = X_proj[:,1]
            
            
            plt.scatter(x1,x2,c=y,cmap="viridis")
            plt.xlabel("Pricipal component 1")
            plt.ylabel("Pricipal component 2")
            plt.colorbar()
            st.pyplot(fig)
        
        
       

def main():
    st.title("Try out different classifiers on different datasets!!")
    st.sidebar.title("Welcome to classification Web App")
    st.sidebar.subheader("Choose your dataset")
    dataset_name = st.sidebar.selectbox("Select Dataset",("IRIS","Breast cancer", "Wine"))
    st.sidebar.subheader("Which classifier is the best?")
    classifier_name = st.sidebar.selectbox("Select classifier",("Random Forest","Support Vector Machines(SVM)","K-Nearest Neighbors(KNN)"))
    
    X,y,data = get_dataset(dataset_name)
    st.write("Shape of the dataset=",X.shape)
    st.write("No of classes=", len(np.unique(y)))
    #params=add_parameter_ui(classifier_name)
    
    clf = get_classifier(classifier_name)
    metrics_list = st.sidebar.multiselect("Choose a metric to plot",('Confusion Matrix','Data-visualization'))
    if st.sidebar.button("Classify!", key='classify'):
        X_train, X_test,y_train, y_test =train_test_split(X,y,test_size=0.3,random_state=0)
        #st.write("Shape of the dataset",X_train.shape)
        #st.write("Shape of the dataset",y_train.shape)
        clf.fit(X_train,y_train)
        
        y_pred = clf.predict(X_test)
        
        acc = accuracy_score(y_test,y_pred)
        #plot_metrics(metrics_list)
        st.write(f"Classifier name: {classifier_name}")
        st.write(f"Accuracy: {np.round(acc*100)}")
       
        plot_metrics(metrics_list,clf,X,y,X_test,y_test)
        
    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Classification on different UCI datasets")
        if dataset_name == "IRIS":
            st.subheader("IRIS-Dataset Information")
            st.markdown("This [data set](https://archive.ics.uci.edu/ml/datasets/iris) perhaps the best known database to be found in the pattern recognition literature. Fisher's paper is a classic in the field and is referenced frequently to this day. (See Duda & Hart, for example.) The data set contains 3 classes of 50 instances each, where each class refers to a type of iris plant. One class is linearly separable from the other 2; the latter are NOT linearly separable from each other.")
            st.subheader("Attribute Information")
            st.markdown("1. sepal length in cm \n2. sepal width in cm \n3. petal length in cm\n4. petal width in cm\n5. Predicted classes-: Iris Setosa, Iris Versicolour, Iris Virginica")
            
        elif dataset_name == "Wine":
            st.subheader("Wine-Dataset Information")
            st.markdown("In this [data set](https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic)), features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image. A few of the images can be found at [Web Link]")
            st.subheader("Attribute Information")
            st.markdown("1) Alcohol\n2) Malic acid\n3) Ash\n4) Alcalinity of ash\n5) Magnesium\n6) Total phenols\n7) Flavanoids\n8) Nonflavanoid phenols\n9) Proanthocyanins\n10)Color intensity\n11)Hue\n12)OD280/OD315 of diluted wines\n13)Proline")
            
        elif dataset_name == "Breast cancer":
            st.subheader("Breast cancer-Dataset Information")
            st.markdown("This [data set](https://archive.ics.uci.edu/ml/datasets/wine) is the results of a chemical analysis of wines grown in the same region in Italy but derived from three different cultivars. The analysis determined the quantities of 13 constituents found in each of the three types of wines.")
            st.subheader("Attribute Information")
            st.markdown("1) ID number \n2) Diagnosis (M = malignant, B = benign)3-32)")
            st.markdown("Ten real-valued features are computed for each cell nucleus:\na) radius (mean of distances from center to points on the perimeter)\nb) texture (standard deviation of gray-scale values)\nc) perimeter\nd) area\ne) smoothness (local variation in radius lengths)\nf) compactness (perimeter^2 / area - 1.0)\ng) concavity (severity of concave portions of the contour)\nh) concave points (number of concave portions of the contour)\ni) symmetry \nj) fractal dimension ('coastline approximation' - 1)")
            
        else:
            st.markdown("No information provided")

        
    return 0

if __name__ == '__main__':
    main()
