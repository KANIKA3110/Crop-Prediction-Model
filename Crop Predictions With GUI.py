#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[2]:


from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor,     GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import LinearSVC, SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, r2_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


# In[32]:


data=pd.read_csv("Crop_prediction.csv")


# In[33]:


data.head()


# * (Data set taken from Indian chamber of food and agriculture)
# **Data fields**
# * N - ratio of Nitrogen content in soil
# * P - ratio of Phosphorous content in soil
# * K - ratio of Potassium content in soil
# * temperature - temperature in degree Celsius
# * humidity - relative humidity in %
# * ph - ph value of the soil
# * rainfall - rainfall in mm

# In[34]:


data.tail()


# In[35]:


data.info()


# In[36]:


data.describe()


# In[37]:


data.isnull().sum()


# In[38]:


data.nunique()


# In[39]:


data.columns


# In[40]:


#Visualization
plt.figure(figsize=(8,8))
plt.title("Correlation between features")
corr=data.corr()
sns.heatmap(corr,annot=True)


# In[41]:


data['label'].unique()


# In[42]:


plt.figure(figsize=(6,8))
plt.title("Temperature relation with crops")
sns.barplot(y="label", x="temperature", data=data,palette="hot")
plt.ylabel("crops")

#Temperature has very effect with blackgram


# In[43]:


plt.figure(figsize=(6,8))
plt.title("Humidity relation with crops")
sns.barplot(y="label", x="humidity", data=data,palette='brg')
plt.ylabel("crops")

#humidity has very high relation with rice


# In[44]:


plt.figure(figsize=(6,8))
plt.title("pH relation with crops")
sns.barplot(y="label", x="ph", data=data,palette='hot')
plt.ylabel("crops")

#ph has a very high relationship with crops


# In[45]:


plt.figure(figsize=(6,8))
plt.title("Rainfall relation with crops")
sns.barplot(y="label", x="rainfall", data=data,palette='brg')
plt.ylabel("crops")

#Rice needs a lots of rainfall
#lentil needs a very less rainfall


# In[46]:


plt.figure(figsize=(8,6))
plt.title("Temperature and pH effect values for crops")
sns.scatterplot(data=data, x="temperature", y="label", hue="ph",palette='brg')
plt.ylabel("Crops")


# In[47]:


plt.figure(figsize=(8,6))
plt.title("Temperature and humidity effect values for crops")
sns.scatterplot(data=data, x="temperature", y="label", hue="humidity",palette='brg')
plt.ylabel("Crops")


# In[48]:


plt.figure(figsize=(8,6))
plt.title("Temperature and Rainfall effect values for crops")
sns.scatterplot(data=data, x="temperature", y="label", hue="rainfall",palette='brg')
plt.ylabel("Crops")


# In[49]:


#from pandas_profiling import ProfileReport


# In[50]:


#Predictions
encoder=LabelEncoder()
data.label=encoder.fit_transform(data.label)


# In[51]:


features=data.drop("label",axis=1)
target=data.label


# In[52]:


features


# In[53]:


X_train, X_test, y_train, y_test = train_test_split(features, target, random_state=42)


# In[54]:


#Linear Regression
lr = LinearRegression().fit(X_train, y_train)
lr_pred= lr.score(X_test, y_test)

print("Training score: {:.3f}".format(lr.score(X_train, y_train)))
print("Test score: {:.3f}".format(lr.score(X_test, y_test)))


# In[55]:


#Decision Tree Classifier
tree = DecisionTreeClassifier(max_depth=15,random_state=0).fit(X_train, y_train)
tree_pred= tree.score(X_test, y_test)

print("Training score: {:.3f}".format(tree.score(X_train, y_train)))
print("Test score: {:.3f}".format(tree.score(X_test, y_test)))


# In[56]:


#Random Forests
rf = RandomForestClassifier(n_estimators=10, max_features=3, random_state=0).fit(X_train, y_train)
rf_pred= rf.score(X_test, y_test)

print("Training score: {:.3f}".format(rf.score(X_train, y_train)))
print("Test score: {:.3f}".format(rf.score(X_test, y_test)))


# In[57]:


#GradientBoostingClassifier
gbr = GradientBoostingClassifier(n_estimators=20, max_depth=4, max_features=2, random_state=0).fit(X_train, y_train)
gbr_pred= gbr.score(X_test, y_test)

print("Training score: {:.3f}".format(gbr.score(X_train, y_train)))
print("Test score: {:.3f}".format(gbr.score(X_test, y_test)))


# In[58]:


#Support Vector Classifier
svm = SVC(C=100, gamma=0.001).fit(X_train, y_train)
svm_pred= svm.score(X_test, y_test)

print("Training score: {:.3f}".format(svm.score(X_train, y_train)))
print("Test score: {:.3f}".format(svm.score(X_test, y_test)))


# In[59]:


#Logistic regression
log_reg = LogisticRegression(C=0.1, max_iter=100000).fit(X_train, y_train)
log_reg_pred= log_reg.score(X_test, y_test)

print("Training score: {:.3f}".format(log_reg.score(X_train, y_train)))
print("Test score: {:.3f}".format(log_reg.score(X_test, y_test)))


# In[60]:


predictions_acc = { "Model": ['Decision Tree', 'Random Forest', 'Gradient Boosting', 'SVC', 'Logistic Regression'],
"Accuracy": [tree_pred, rf_pred, gbr_pred, svm_pred, log_reg_pred]}


# In[61]:


model_acc = pd.DataFrame(predictions_acc, columns=["Model", "Accuracy"])


# In[62]:


model_acc


# In[3]:


import tkinter as tk
from tkinter.font import BOLD
from tkinter import messagebox
from tkinter import scrolledtext
from tkinter.constants import RIGHT, Y
from tkinter import filedialog
from tkinter import *


# In[8]:


def mainscreen():
    
    global window
    window = tk.Tk()
    window.geometry("1530x795+0+0")
    window.configure(bg="#FFE4B5")
    window.title("Prediction model")
    
    
    head = tk.Label(window, text="\nEnter Details\n", font=("rockwell extra bold",45),fg="dark blue",bg="#FFE4B5").pack()
    
    
    def back3() :
        window.destroy()

    def values():
        
        n=n_tk.get()
        p=p_tk.get()
        k=k_tk.get()
        temp=temp_tk.get()
        humidity=humidity_tk.get()
        ph=ph_tk.get()
        rainfall=rainfall_tk.get()
        
        def predictfunc(n,p,k,temp,humidity,ph,rainfall):
            #Predicting Model
            data=pd.read_csv("Crop_prediction.csv")

            x=data.loc[:,"N":"rainfall"]
            y=data.loc[:,'label']

            Knn=KNeighborsClassifier()
            Knn.fit(x,y)

            test_data=[[n,p,k,temp,humidity,ph,rainfall]]
            predict=Knn.predict(test_data)
            #print(predict[0])
            output1 = tk.Label(window, text="The prediction is: ",font=("Arial", 20),bg="#FFE4B5").place(x=600, y=570)
            output2 = tk.Label(window, text=predict, font=("Arial", 20),bg="#FFE4B5").place(x=820, y=570)
        

        predictfunc(n,p,k,temp,humidity,ph,rainfall)
        


    n1 = tk.Label(window, text="Ratio of Nitrogen content in soil: ",font=("Arial", 20),bg="#FFE4B5").place(x=320, y=200)

    n_tk = tk.Entry(window, fg='blue', bg='white',borderwidth=5,font=("Arial", 18), width=30)
    n_tk.place(x=800, y=200)

    p2 = tk.Label(window, text="Ratio of Phosphorous content in soil: ",font=("Arial", 20),bg="#FFE4B5").place(x=320, y=250)

    p_tk = tk.Entry(window, fg='blue', bg='white',borderwidth=5,font=("Arial", 18), width=30)
    p_tk.place(x=800, y=250)

    k3 = tk.Label(window, text="Ratio of Potassium content in soil: ",font=("Arial", 20),bg="#FFE4B5").place(x=320, y=300)

    k_tk = tk.Entry(window, fg='blue', bg='white',borderwidth=5,font=("Arial", 18), width=30)
    k_tk.place(x=800, y=300)
    
    temp4= tk.Label(window, text="Temperature in degree Celsius: ",font=("Arial", 20),bg="#FFE4B5").place(x=320, y=350)

    temp_tk = tk.Entry(window, fg='blue', bg='white',borderwidth=5,font=("Arial", 18), width=30)
    temp_tk.place(x=800, y=350)
    
    humidity5= tk.Label(window, text="Relative humidity in %: ",font=("Arial", 20),bg="#FFE4B5").place(x=320, y=400)

    humidity_tk = tk.Entry(window, fg='blue', bg='white',borderwidth=5,font=("Arial", 18), width=30)
    humidity_tk.place(x=800, y=400)
    
    ph6= tk.Label(window, text="pH value of the soil: ",font=("Arial", 20),bg="#FFE4B5").place(x=320, y=450)

    ph_tk = tk.Entry(window, fg='blue', bg='white',borderwidth=5,font=("Arial", 18), width=30)
    ph_tk.place(x=800, y=450)
    
    rainfall7= tk.Label(window, text="Rainfall in mm: ",font=("Arial", 20),bg="#FFE4B5").place(x=320, y=500)

    rainfall_tk = tk.Entry(window, fg='blue', bg='white',borderwidth=5,font=("Arial", 18), width=30)
    rainfall_tk.place(x=800, y=500)
        

    back3_button = tk.Button(text="Exit", bg="blue", fg="white", height=1, width=10, borderwidth=8, cursor="hand2",font=("Arial", 12), command=back3)
    back3_button.place(x=530,y=680)

    submit_button = tk.Button(text="Submit", bg="green", fg="white", height=1, width=10, borderwidth=8, cursor="hand2",font=("Arial", 12), command=values)
    submit_button.place(x=830,y=680)

    # start the GUI
    window.mainloop()

mainscreen()


# In[ ]:




