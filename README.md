# Classification-Web-App-using-Streamlit-
I have tried to create a web app based on binary/multiclass classification using streamlit and the app is deployed through Streamlit, Azure and Heroku.

# Link for app
The deployed web app is live at streamlit : https://akshadaas-classification-web-app-using-st-classifier-app-t7w8fp.streamlitapp.com/, Azure : https://classifier-app.azurewebsites.net/, Heroku : https://mlclassifier-web-app.herokuapp.com/. 

Please refer links for deployment streamlit : https://docs.streamlit.io/streamlit-cloud/get-started/deploy-an-app , Azure : https://www.youtube.com/watch?v=2toRzAYT8yo&t=631s,  Heroku: https://www.youtube.com/watch?v=zK4Ch6e1zq8&t=307s.

This app perform classification on UCI datasets (IRIS, Breast Cancer, Wine) using different machine learning algorithms (Random Forest, Support vector machine (SVM), K-Nearest Neighbors (KNN)). You can also visualize these datasets. 

# Streamlit
Streamlit’s open-source app framework is the easiest way for data scientists and machine learning engineers to create beautiful, performant apps in only a few hours! All in pure Python. All for free. Streamlit allows rapid webapp development without the knowledge of HTML and CSS. And no need to write backend or handle HTTP requests. 

For additional information on streamlit, please refer https://docs.streamlit.io/

# Running on local machine

pip install -r requirements.txt

streamlit run classifier-app.py

