import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn as snv
from sklearn .model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder 
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import pickle 


df=pd.read_csv("heart.csv")
st.header(":green[Heart Disease Model Prediction]")

st.subheader(":violet[Information of the data]")




# Load the DataFrame
#df = pd.read_csv("heart.csv")

# Display the DataFrame using st.dataframe()
st.dataframe(df)

# Display additional information using st.write()

st.write(df.info())

st.subheader(":violet[Shape of the data frame]")
st.write(df.shape)

st.subheader(":violet[Describe of the data frame]")
st.write(df.describe())

st.subheader(":violet[Value counts of the HeartDisease]")
st.write(":blue[If value count is 1 then person has HeartDisease]")
st.write(":blue[If value count is 0 then person has not HeartDisease]")

st.write(df['HeartDisease'].value_counts())


# Create a Next button
next_button = st.button("Next")

# Check if the Next button is clicked
if next_button:
    # Display additional content when the Next button is clicked
    #st.write("Additional content goes here...")
    st.write("You can put any content you want to show when the button")

#def moveon(direction):

       # left,middle,right = st.columns(3)
       # middle.button("Reset",on_click=moveon,args=['reset'],key='reset_button')
       # right.button("Next",on_click=moveon,args=['next'],key='next_button')
    










train, test = train_test_split(df,test_size = 0.20 ,random_state = 112)





def stage1():
    st.header(":green[Here I am doing Graph Plotation]")
    
    
    
    
    
    
    
    
    
    
    selected_state = st.radio(":orange[choose for different plotation]",
                              options=["Bar plot of the count of numeric features",
                                      "Fraction of  HeartDisease patient effected by ChestPainType",
                                      "Average age of patient effected by chestpain",
                                      "Age histogram of the Heart Disease patient",
                                      "Heart Disease graph plot according to sex",
                                      "Heart Disease graph plot according to st_slope"],
                              )
    if selected_state== "Age histogram of the Heart Disease patient":

        # Create a new figure
        fig = plt.figure(figsize=(10, 6))

        # Create the histogram directly on the figure object using 'Age' column from 'train' DataFrame
        ax = fig.add_subplot(111)
        train['Age'].hist(bins=30, color='darkred', alpha=0.7, ax=ax)

        # Set the x-axis label, y-axis label, and title of the plot directly on the subplot
        ax.set_xlabel("Age of the people", fontsize=18)
        ax.set_ylabel("Count", fontsize=18)
        ax.set_title("Age histogram of the Heart Disease patient", fontsize=22)
        fig
        
    elif selected_state=="Fraction of  HeartDisease patient effected by ChestPainType":
        
        f_class_HeartDisease=train.groupby('ChestPainType')['HeartDisease'].mean()
        f_class_HeartDisease = pd.DataFrame(f_class_HeartDisease)
        f_class_HeartDisease
        fig = plt.figure(figsize=(10, 6))

        # Create the bar plot directly on the figure object using 'f_class_HeartDisease' DataFrame
        ax = fig.add_subplot(111)
        f_class_HeartDisease.plot.bar(y='HeartDisease', ax=ax)

        # Set the x-axis label, y-axis label, and title of the plot directly on the subplot
        ax.set_xlabel("Chest Pain Type", fontsize=16)
        ax.set_ylabel("Fraction of Heart Disease Patients", fontsize=16)
        fig
        
    elif selected_state=="Average age of patient effected by chestpain":
        f_class_Age=train.groupby('ChestPainType')['Age'].mean()

        fig = plt.figure(figsize=(10, 6))

        # Create the bar plot directly on the figure object using 'f_class_Age' DataFrame
        ax = fig.add_subplot(111)
        f_class_Age.plot.bar(y='Age', ax=ax)

        # Set the x-axis label, y-axis label, and title of the plot directly on the subplot
        ax.set_xlabel("Chest Pain Type", fontsize=16)
        ax.set_ylabel("Average Age (years)", fontsize=16)
        fig
        
    elif selected_state=="Bar plot of the count of numeric features":
        d=df.describe()
        dT=d.T
        fig = plt.figure(figsize=(10, 6))

        # Create the bar plot directly on the figure object using 'dT' DataFrame
        ax = fig.add_subplot(111)
        dT.plot.bar(y='count', ax=ax)

        # Set the x-axis label, y-axis label, and title of the plot directly on the subplot
        ax.set_xlabel("Numeric Features", fontsize=16)
        ax.set_ylabel("Count", fontsize=16)
        fig
        
    elif  selected_state=="Heart Disease graph plot according to sex":
        fig = plt.figure(figsize=(10, 6))

        # Set the seaborn style to 'whitegrid'
        sns.set_style('whitegrid')

        # Create the count plot directly on the figure object using 'train' DataFrame
        ax = fig.add_subplot(111)
        sns.countplot(x='Sex', hue='HeartDisease', data=train, palette='RdBu_r', ax=ax)

        # Set the x-axis label, y-axis label, and title of the plot directly on the subplot
        ax.set_xlabel("Sex", fontsize=16)
        ax.set_ylabel("Count", fontsize=16)
        fig
        
    else:
        fig = plt.figure(figsize=(10, 6))

        # Set the seaborn style to 'whitegrid'
        sns.set_style('whitegrid')

        # Create the count plot directly on the figure object using 'train' DataFrame
        ax = fig.add_subplot(111)
        sns.countplot(x='ST_Slope', hue='HeartDisease', data=train, palette='RdBu_r', ax=ax)

        # Set the x-axis label, y-axis label, and title of the plot directly on the subplot
        ax.set_xlabel("ST Slope", fontsize=16)
        ax.set_ylabel("Count", fontsize=16)
        fig



    
    
    
   
    
def stage2():
    st.header(":green[Percentage wise various score prediction]")
    st.subheader(":blue[Conversion of string part into integer]")
    
    X=df.drop(columns='HeartDisease', axis =1)
    Y=df['HeartDisease']
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,stratify=Y,random_state=3)
    print(X.shape, X_train.shape, X_test.shape)
    
    model=LogisticRegression()
    
    label_encoder = LabelEncoder()
    p = ['Sex','ChestPainType','ExerciseAngina','ST_Slope','RestingECG']
    for i in p:
        X_train[i] = label_encoder.fit_transform(X_train[i])
    X_train
    
    for i in p:
        X_test[i] = label_encoder.fit_transform(X_test[i])
    X_test
    
#for opening the file
    model.fit(X_train,Y_train)
    # st.session_state['model'] = model
    with open('HeartDisease_model.pkl', 'wb') as f:
        pickle.dump(model, f)
        print('model is saved')
    
    
    
    
  #accuracy of the training data and test data  
    X_train_prediction=model.predict(X_train)
    training_data_accuracy=accuracy_score(X_train_prediction, Y_train) 
    
    X_test_prediction=model.predict(X_test)
    test_data_accuracy=accuracy_score(X_test_prediction, Y_test)

#aprecision of training data and test data
    precision_train=precision_score(Y_train,X_train_prediction)

    precision_test=precision_score(Y_test,X_test_prediction)
    
    
#recall of training data and test data


    recall_train=recall_score(Y_train,X_train_prediction)
    
    recall_test=recall_score(Y_test,X_test_prediction)
    
#F1 score of training data and test data

    f1_score_train=f1_score(Y_train,X_train_prediction)
    
    f1_score_test=f1_score(Y_test,X_test_prediction)

    st.subheader(":blue[Table of score prediction]")
    
    data = {
        'Type of score':["Accuracy", "Precision", "Recall", "F1"],
        "Training Data": [training_data_accuracy,precision_train,recall_train,f1_score_train],
        "Testing Data": [test_data_accuracy,precision_test,recall_test,f1_score_test]
    }
    
    df1 = pd.DataFrame(data)
    st.table(df1)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

def prediction():
    
        # 
    HeartDisease_model = pickle.load(open('HeartDisease_model.pkl', 'rb'))
    # HeartDisease_model = st.session_state['model']
    
    st.header(":green[Heart Disease prediction using model]")
    col1, col2,col3=st.columns(3)
    
    with col1:
        Age=st.number_input(":blue[Age of the people]")
    
        
    with col2:
    
            Sex=st.selectbox(":blue[Gender]",options = ['Male','Female'])
        
            

            #Sex = st.selectbox("sellect",
                               # options=["sellect","Male", "Female"],)

           # st.write(f"You chose: {Sex}")

        #if __name__ == "__main__":
          #  main()
        
    with col3:
        ChestPainType=st.selectbox(":red[Type of the Chest pain]",options = ['ATA','ASY','NAP','TA'])
    with col1:    
        RestingBP=st.number_input(":blue[RestingBP value]")
    with col2:   
        Cholesterol=st.number_input(":violet[ Cholestrol value]")
    with col3 :  
        FastingBS=st.number_input(":red[BS value]")
    with col1:    
        RestingECG=st.selectbox(":blue[ RestingECG]",options = ['Normal','ST','LVS'])
    with col2:    
        MaxHR=st.number_input(":violet[MaxHR value]")
    with col3 :  
        ExerciseAngina=st.selectbox(":red[exercise angina]",options = ['N','Y'])
    with col1:    
        Oldpeak=st.number_input(":blue[Old peak type]")
    with col2:    
        St_Slope=st.selectbox(":violet[st slope category]",options = ['Up','Flat','Down'])
    #code for prediction    
    heart_dis=''
    
    #string to float convert
    if Sex == 'Male':
        Sex = 1
    elif Sex == 'Female':
        Sex = 0
        
    if ChestPainType == 'ATA':
        ChestPainType = 1
    elif ChestPainType == 'ASY':
        ChestPainType = 0
    elif ChestPainType == 'NAP':
        ChestPainType = 2
    elif ChestPainType == 'TA':
        ChestPainType = 3    
        
    if RestingECG == 'Normal':
        RestingECG = 1
    elif RestingECG == 'ST':
        RestingECG = 2
    elif RestingECG == 'LVH':
        RestingECG = 0
    
    if ExerciseAngina == 'N':
        ExerciseAngina = 0
    elif ExerciseAngina == 'Y':
        ExerciseAngina = 1    
        
    if St_Slope == 'Up':
        St_Slope = 2
    elif St_Slope == 'Flat':
        St_Slope = 1
    elif St_Slope == 'Down':
        St_Slope = 0  
        
        
        
        
        
     #creating a button for prediction   
        
    if st.button("Heart Disease Test Result"):
        # st.write(dir(HeartDisease_model))
        heart_prediction=HeartDisease_model.predict([[Age,Sex,ChestPainType,RestingBP,Cholesterol,FastingBS, RestingECG ,MaxHR,ExerciseAngina,Oldpeak,St_Slope]])
        
        if ( heart_prediction[0]==0):
            heart_dis=":green[The person has not Heart Disease]"
            
        else:
            
            heart_dis=":Red[The person have Heart Disease]"
    
    st.success(heart_dis)        




selected_state = st.radio(":green[Choose your state]",
                          options=["display plots",
                                   'showing results',
                                   'prediction'],
                          horizontal=True   )
if selected_state=="display plots":
    stage1()
elif selected_state=='prediction':
    prediction()
else:
    stage2()