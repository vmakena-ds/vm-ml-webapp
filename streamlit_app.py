import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title('This is ML App')
st.write('Classification Model on IRIS Dataset')

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

url = "https://raw.githubusercontent.com/vmakena-ds/vm-ml-webapp/refs/heads/master/Iris.csv"
df = pd.read_csv(url)
X = df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']].values
y = df['Species'].values


st.write("Sample Dataset:")
st.write(df.head(5))

st.header("Iris: Sepal Length vs Sepal Width")
st.scatter_chart(
    data=df,
    x="SepalLengthCm",
    y="SepalWidthCm",
    color="Species",     # category-based color encoding :contentReference[oaicite:1]{index=1}
    size=40,             # fixed circle size
    use_container_width=True
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

st.write("Accuracy:", accuracy_score(y_test, y_pred))
