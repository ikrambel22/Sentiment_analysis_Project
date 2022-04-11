import pickle
#from turtle import width
from matplotlib import colors
import streamlit as st
import numpy as np
from streamlit_option_menu import option_menu
import pandas as pd
import plotly.express as px
import matplotlib as plt
import seaborn as sns
from streamlit_echarts import st_echarts
import json




model=pickle.load(open('sentiment_analysis.pickle','rb'))
vectorizer=pickle.load(open('TF_IDF_vectorizer.pickle','rb'))
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(layout="wide")

@st.cache
def load_data(valid):
    #text_valid=valid.select_dtypes(['object'])
    idx=(valid.applymap(type) == str).all(0)
    df_new = valid[valid.columns[idx]]
    text_cols=df_new.columns
    return df_new ,text_cols


def calcul_average(l):
        i=0
        j=0
        k=0
        for x in l:
            if x=='negative':
                i=i+1
            elif x=='positive':
                j=j+1
            else:
                k=k+1
        return i,j,k  
        
def string_sentiment(x):        
        if x=='negative':
            s='Negative'
        elif x=='positive':
            s='Positive'
        else:
           s='Neutral'
           

        return s   

   

def main():
    #sidebar menu
    with st.sidebar:
        selected=option_menu(
            menu_title="Main Menu",
            options=['Home','Dataset','About'],
            icons=['textarea-t','table','pin-fill'],
            menu_icon="cast",
            default_index=0,
            #orientation='horizental'
        )

    if selected=='Home':
        col1,col2 =  st.columns([3,1])
        
        with col1:
            st.title('Sentiment Analysis')
        with col2:
            st.image('image/logo2.png')
        
        #input text
        text=st.text_area('Have you ever tried an airline service, Tell us your story! ')
        #prediction code
        if st.button('Detect Sentiment'):
            #vectorize the text
            test = vectorizer.transform([text])
            #var_test=toNumpyArray(test)
            l=model.predict(test)
            #output=round(l[0],2)
           

            st.success('The predicted Sentiment is: {}'.format(l[0]))
            pred_proba=model.predict_proba(test)
            pred_percentage_for_all=dict(zip(model.classes_,pred_proba[0]))
            val=np.max(pred_proba)*100

            s=string_sentiment(l[0])
                                       
            # if s=='Negative':
            #                 option = {
            #                             "tooltip": {
            #                                 "formatter": '{a} <br/>{b} : {c}%'
            #                             },
            #                             "series": 
            #                             [{
            #                                 "name": 'sentiment analysis',
            #                                 "type": 'gauge',
            #                                 "startAngle": 180,
            #                                 "endAngle": 0,
            #                                 "progress": {
            #                                     "show": "true"
            #                                 },
            #                                 "radius":'100%', 

            #                                 "itemStyle": {
            #                                     "color": '#58D9F9',
            #                                     "shadowColor": 'rgba(0,138,255,0.45)',
            #                                     "shadowBlur": 10,
            #                                     "shadowOffsetX": 2,
            #                                     "shadowOffsetY": 2,
            #                                     "radius": '55%',
            #                                 },
            #                                 "progress": {
            #                                     "show": "true",
            #                                     "roundCap": "true",
            #                                     "width": 15
            #                                 },
            #                                 "pointer": {
            #                                     "length": '60%',
            #                                     "width": 8,
            #                                     "offsetCenter": [0, '5%']
            #                                 },
            #                                 "detail": {
            #                                     "valueAnimation": "true",
            #                                     "formatter": '{value}%',
            #                                     "backgroundColor": '#58D9F9',
            #                                     "borderColor": '#999',
            #                                     "borderWidth": 4,
            #                                     "width": '60%',
            #                                     "lineHeight": 20,
            #                                     "height": 20,
            #                                     "borderRadius": 188,
            #                                     "offsetCenter": [0, '40%'],
            #                                     "valueAnimation": "true",
            #                                 },
            #                                 "data":
            #                                 [ 
            #                                     {
                                                
            #                                     "value": 20,
            #                                     "name": 'Negative'
                                                
            #                                 }
            #                                 ]
            #                             }]
            #                         }
            # elif s=='Neutral':
            #                           option = {
            #                             "tooltip": {
            #                                 "formatter": '{a} <br/>{b} : {c}%'
            #                             },
            #                             "series": 
            #                             [{
            #                                 "name": 'sentiment analysis',
            #                                 "type": 'gauge',
            #                                 "startAngle": 180,
            #                                 "endAngle": 0,
            #                                 "progress": {
            #                                     "show": "true"
            #                                 },
            #                                 "radius":'100%', 

            #                                 "itemStyle": {
            #                                     "color": '#58D9F9',
            #                                     "shadowColor": 'rgba(0,138,255,0.45)',
            #                                     "shadowBlur": 10,
            #                                     "shadowOffsetX": 2,
            #                                     "shadowOffsetY": 2,
            #                                     "radius": '55%',
            #                                 },
            #                                 "progress": {
            #                                     "show": "true",
            #                                     "roundCap": "true",
            #                                     "width": 15
            #                                 },
            #                                 "pointer": {
            #                                     "length": '60%',
            #                                     "width": 8,
            #                                     "offsetCenter": [0, '5%']
            #                                 },
            #                                 "detail": {
            #                                     "valueAnimation": "true",
            #                                     "formatter": '{value}%',
            #                                     "backgroundColor": '#58D9F9',
            #                                     "borderColor": '#999',
            #                                     "borderWidth": 4,
            #                                     "width": '60%',
            #                                     "lineHeight": 20,
            #                                     "height": 20,
            #                                     "borderRadius": 188,
            #                                     "offsetCenter": [0, '40%'],
            #                                     "valueAnimation": "true",
            #                                 },
            #                                 "data":
            #                                 [ 
            #                                     {
                                                
            #                                     "value": 50,
            #                                     "name": 'Neutral'
                                                
            #                                 }
            #                                 ]
            #                             }]
            #                         }
            # else:
            #      option = {
            #                             "tooltip": {
            #                                 "formatter": '{a} <br/>{b} : {c}%'
            #                             },
            #                             "series": 
            #                             [{
            #                                 "name": 'sentiment analysis',
            #                                 "type": 'gauge',
            #                                 "startAngle": 180,
            #                                 "endAngle": 0,
            #                                 "progress": {
            #                                     "show": "true"
            #                                 },
            #                                 "radius":'100%', 

            #                                 "itemStyle": {
            #                                     "color": '#58D9F9',
            #                                     "shadowColor": 'rgba(0,138,255,0.45)',
            #                                     "shadowBlur": 10,
            #                                     "shadowOffsetX": 2,
            #                                     "shadowOffsetY": 2,
            #                                     "radius": '55%',
            #                                 },
            #                                 "progress": {
            #                                     "show": "true",
            #                                     "roundCap": "true",
            #                                     "width": 15
            #                                 },
            #                                 "pointer": {
            #                                     "length": '60%',
            #                                     "width": 8,
            #                                     "offsetCenter": [0, '5%']
            #                                 },
            #                                 "detail": {
            #                                     "valueAnimation": "true",
            #                                     "formatter": '{value}%',
            #                                     "backgroundColor": '#58D9F9',
            #                                     "borderColor": '#999',
            #                                     "borderWidth": 4,
            #                                     "width": '60%',
            #                                     "lineHeight": 20,
            #                                     "height": 20,
            #                                     "borderRadius": 188,
            #                                     "offsetCenter": [0, '40%'],
            #                                     "valueAnimation": "true",
            #                                 },
            #                                 "data":
            #                                 [ 
            #                                     {
                                                
            #                                     "value": 90,
            #                                     "name": 'Positive'
                                                
            #                                 }
            #                                 ]
            #                             }]
            #                         }

            
            # st_echarts(options=option , key="1")
        
            


    elif selected=='Dataset':
        st.title('Sentiment Analysis')
        
        uploaded_file=st.file_uploader(label='Upload your CSV or Excel file!',type=['csv','xslx'])
        
       
        if uploaded_file is not None:
            try:
                valid=pd.read_csv(uploaded_file)
                
            except Exception as e :  
                #print(e)
                valid=pd.read_excel(uploaded_file)
                 
             

            
            st.success('File uploaded succesfully')
            check_box=st.checkbox(label='Display Dataset')
            #vectorize the text
            text_valid,text_cols=load_data(valid) 
            if load_data(valid)==False:
                st.info('Please Enter your data First !') 
            
            # print(text_valid)
            # print(text_cols)
            if check_box:
                    st.write(text_valid)
            
            
            # This portion is part of my test code
       
        if(st.button('Generate Visualization')):
            for x in text_cols:
                #if x =='text' or x=='Text' or x=='TEXT':
                    test =vectorizer.transform(text_valid[x])
                    l=model.predict(test)
                    i,j,k=calcul_average(l)
                    val=max(i,j,k)
                    val1=(val/len(text_valid))*100   
                    col4, col5= st.columns([3,1])
                    
                    with col4:
                            labels = ['Negative', 'Neutral','Positive']
                            sizes = [(i/len(text_valid))*100, (k/len(text_valid))*100, (j/len(text_valid))*100]
                            # Plot
                                # plt.pie(sizes, labels=labels, colors=['red','green','orange'],
                                #         autopct='%1.1f%%', shadow=True, startangle=140)
                            st.subheader('Sentiments Circle Percentage')    
                            fig=px.pie(data_frame=text_valid,names=labels,values=sizes,color=['orange','red','green'],labels=labels)        
                            # plt.axis('equal')
                            # plt.title('Sentiments Circle Percentage')
                            #st.pyplot(fig)
                            st.write(fig)
                            # st.title('Sentiments CountPlot')   
                            # fig1=sns.countplot(x=l,data=text_valid)
                            
                            # st.pyplot()  
                    with col5:
                        st.subheader('Service Review ')
                        if val ==i:
                            if val1==0:
                                  st.image('image/0star.png')
                            elif val1>0 and val<=50:
                                st.image('image/1star.png')
                            else:
                                st.image('image/1.5star.jpg')
                        elif val==  k: 
                            if val1==0:
                                  st.image('image/2star.png')
                            elif val1>0 and val<=50:
                                st.image('image/2.5star.png')
                            else:
                                st.image('image/3star.png')
                        else:  
                            if val1==0:
                                  st.image('image/3.5star.png')
                            elif val1>0 and val<=50:
                                #st.image('4star.png')
                                st.image(bytes('image/4star.png', "utf-8"))
                            else:
                                st.image(bytes('image/4star.png', "utf-8"))
                                st.image('5star.png')          
                        break
                # else:
                #     st.info('Please make sure that the column name is Text!')
                #     break
        
    
            #Visualization
            
        # else:
        #      st.error('Format Not Recognized, Please Enter Csv or Excel File!')    

        #st.title(f'you have selected {selected}')
        # text_valid,text_cols=load_data(valid) 
        # st.write(text_valid[text_cols[0]])
    else:
        #st.title(f'you have selected {selected}') 
        col1,col2 =  st.columns([1,5])
        
        with col1:
            st.image('image/images (1).jpg')
        with col2:
            st.title('About The app')
        
        st.subheader("Objective of App")
        st.markdown('The objective of this application is to predict the sentiments of customers towards an aireline service , to have an idea of ​​​​the quality of the services offered by this aireline in order to deduce whether it is recommended or not recommended.')  

        st.subheader("There are Two Sections in The App")
        st.markdown("First Section Home: allows the user to enter a text (comment, review,...), in order to predict the sentiment : positive, negative or neutral.")

        st.markdown("Second Section Dataset: allows to upload a dataset (contains a single column of customers reviews or comments ), in order to predict the quality of services offered by an airline.")

        st.write()
        st.title('Contact Us')

        coll,coll2,col3 =  st.columns([1,1,1])
        
        with coll:
           
            st.image('image/team.png',width=150)
            st.title('Team')

            st.subheader('Ikram Belgas')   
            st.subheader('Laila Boullous') 
            
        with coll2:
           
            st.write()
        with col3:
                st.image('image/linkdin.png',width=150)
                st.text("")
                st.write('\t')
                st.title('Linkdin ')
                st.markdown(" [Ikram Belgas](https://www.linkedin.com/in/ikram-b-863612183/)")
                st.markdown(" [Laila Boullous]()")
   
    
    st.text("")
    st.write('Check Our kaggle notebook : [click here](https://www.kaggle.com/code/ikrambelgas/sentiment-analysis/notebook)')
         
        
    
        
    



            
if __name__=="__main__":
        main()   
        # text_valid,text_cols=load_data(valid) 
        # print(text_valid[text_cols[0]])
        
