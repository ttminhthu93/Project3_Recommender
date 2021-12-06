# import libraries
# pip install pyqt5
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
import pandas as pd
import seaborn as sns
import re
import difflib

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt

import pandas_profiling
from streamlit_pandas_profiling import st_profile_report


# Source Code
# products = pd.read_csv('data/ProductRaw.csv')
# reviews = pd.read_csv('data/ReviewRaw.csv')
products = pd.read_csv('https://drive.google.com/file/d/1iaCzr-TIKuxphG8Ke1hwvAlzyFiy1oa1/view?usp=sharing')
reviews = pd.read_csv('https://drive.google.com/file/d/1vgg5VRp24MUnCxvnk6Obzn9kmCqu3PVG/view?usp=sharing')

pr_products = products.profile_report()
pr_reviews = reviews.profile_report()

# GUI
st.title("Data Science Project")
st.write("## Topic 3 - Recommender System")
# # Upload file
# uploaded_file = st.file_uploader("Choose a file", type=['csv'])
# if uploaded_file is not None:
#     data = pd.read_csv(uploaded_file)
#     data.to_csv("avocado_new.csv", index = False)

# GUI
menu = ["Business Objective", "Data Preprocessing", "Show Prediction"]
choice = st.sidebar.selectbox('Menu', menu)
if choice == 'Business Objective':    
    st.image("logo.jpg")

    st.subheader("Business Objective")
    st.write("""
    - Tiki là một hệ sinh thái thương mại “all in one”, trong đó có tiki.vn, là một website thương mại điện tử đứng top 2 của Việt Nam, top 6 khu vực Đông Nam Á.
    - Trên trang này đã triển khai nhiều tiện ích hỗ trợ nâng cao trải nghiệm người dùng và họ muốn xây dựng nhiều tiện ích hơn nữa.
    - Giả sử công ty này chưa triển khai Recommender System và bạn được yêu cầu triển khai hệ thống này, bạn sẽ làm gì?
    """)  

    st.write("""
    **Mục tiêu/ Vấn đề:** Xây dựng Recommendation System cho một hoặc một số nhóm hàng hóa trên tiki.vn giúp đề xuất và gợi ý cho người dùng/ khách hàng. 
    ###### => Project này sẽ tập trung xây dựng các mô hình đề xuất dựa trên:
    - Content-based filtering
    - Collaborative filtering
    """)

elif choice == 'Data Preprocessing':
    st.subheader("Data Preprocessing")
    st.write("""
    ##### Raw data:
    **Products**
    """)
    st.dataframe(products.head(3))
    st.dataframe(products.tail(3))  
    st_profile_report(pr_products)
    
    st.write("""
    **Reviews**
    """)
    st.dataframe(reviews.head(3))
    st.dataframe(reviews.tail(3))  
    st_profile_report(pr_reviews)

# elif choice == 'Build Project':
#     st.subheader("Build Project")
#     st.write("""
#     ##### Some data:
#     """)
#     st.dataframe(df_ts.head(3))
#     st.dataframe(df_ts.tail(3))   
#     st.text("Mean of Organic Avocado AveragePrice in California: " + str(round(df_ts['y'].mean(),2)) + " USD")
#     st.write("""
#     ##### Build model ...
#     """)
#     st.write("""
#     ##### Calculate MAE/RMSE between expected and predicted values
#     """)
#     st.code("MAE: " + str(round(mae_p,2)))
#     st.code("RMSE: " + str(round(rmse_p,2)))
#     st.write("""This result shows that Prophet's RMSE and MAE are good enough to predict the organic avocado AveragePrice in California, MAE = 0.16 (about 10% of the AveragePrice), compared to the AveragePrice ~ 1.68.
#     """)
#     st.write("##### Visualization: AveragePrice vs AveragePrice Prediction")
#     # Visulaize the result
#     fig, ax = plt.subplots()    
#     ax.plot(y_test_value, label='AveragePrice')
#     ax.plot(y_pred_value, label='AveragePrice Prediction')    
#     ax.set_xticklabels(y_test_value.index.date, rotation=60)
#     ax.legend()    
#     st.pyplot(fig)   

# elif choice == 'Show Prediction':
#     st.subheader("Make new prediction for the future in California")
#     st.write("##### Next 12 months")
#     fig1 = model.plot(forecast) 
#     fig1.show()
#     a = add_changepoints_to_plot(fig1.gca(), model, forecast)
#     st.pyplot(fig1)

#     fig2 = model.plot_components(forecast)
#     st.pyplot(fig2)

#     # Next 12 months
#     df_new = forecast[["ds", "yhat"]].tail(12)
#     st.table(df_new)

#     st.write("##### Long-term prediction for the next 5 years => Consider whether to expand cultivation/production, and trading")
#     fig3 = m.plot(forecast_new)     
#     a = add_changepoints_to_plot(fig3.gca(), m, forecast_new)
#     st.pyplot(fig3)

#     fig4, ax = plt.subplots()    
#     ax.plot(df_ts['y'], label='AveragePrice')
#     ax.plot(forecast_new['yhat'], label='AveragePrice with next 60 months prediction', 
#          color='red')    
#     ax.legend()
#     st.pyplot(fig4)
#     st.markdown("Based on the above results, we can see that It's possible to expand the cultivation/production and trading of organic avocados in California.")
