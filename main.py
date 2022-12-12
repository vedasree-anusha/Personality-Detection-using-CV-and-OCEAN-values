import math
import os

import docx2txt
import numpy as np
import pandas as pd
import pdfplumber
import streamlit as st
from PyPDF2 import PdfFileReader
from resume_parser import resumeparse
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import csv


# to read the CV file
def read_pdf(file):
    pdfReader = PdfFileReader(file)
    count = pdfReader.numPages
    all_page_text = ""
    for i in range(count):
        page = pdfReader.getPage(i)
        all_page_text += page.extractText()

    return all_page_text


def read_pdf2(file):
    with pdfplumber.open(file) as pdf:
        page = pdf.pages[0]
        return page.extract_text()


save_path = './resumes'


def fileselector(folder_path='./resumes'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select Resume', filenames)
    return os.path.join(folder_path, selected_filename)


primaryColor = "#F63366"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

# -------------
# layout design
# -------------

st.title('Personality Predictor')

# ---------------------------------#
# Model building
# Job Description
st.subheader("Job Summary")
st.caption("Urgently Required PHP Developers Senior and Junior @ Spero Healthcare innovations Pvt. Ltd.")
st.caption("Technical Skills")
st.caption("1 year - 3 years of working experience on Codeigniter/Laravel")
st.caption("Hands on Experience in web applications PHP(Codeigniter,Laravel), [website], Angular Js/React Js")
st.caption("Knowledge with front-end technologies such as Javascript, Bootstrap, HTML5, CSS, jQuery,CSS3")
st.caption("Knowledge of working with GitHub.")
st.caption("Experience working on Apache HTTP or any other server.")
st.caption("Experience using MySQL, PostgreSQL, Mongodb")
st.caption("Knowledge of Web Services - Rest API")
st.caption("Solid experience in design, coding, unit testing and debugging.")
st.caption("Experience working in Agile development environment.")

skills_required = ["Problem Solving and Technical skills", "OOPS", "data structures", "java", "python", "algorithms",
                   "matlab"]

name = st.text_input('Name', '')
gender = st.selectbox('Gender', ('Female', 'Male'))
age = st.slider('Age', min_value=10, max_value=40, value=20, step=1)
docx_file = st.file_uploader("Upload File", type=['txt', 'docx', 'pdf'])

openness = st.slider('Openness - Enjoy new experience',
                     min_value=0, max_value=10, value=0, step=1)
neuroticism = st.slider(
    'Neuroticism - How often do you feel negativity', min_value=0,
    max_value=10, value=0, step=1)
conscientiousness = st.slider(
    "Conscientiousness - Wishing to do one's work well and thoroughly", min_value=0,
    max_value=10, value=0, step=1)
agreeableness = st.slider(
    'Agreeableness - How much do you like to work with your peers', min_value=0,
    max_value=10, value=0, step=1)
extraversion = st.slider('Extraversion - How much do you like outgoing and social interaction', min_value=0,
                         max_value=10,
                         value=0, step=1)

if gender == "Female":
    gender = 0
else:
    gender = 1

user_input = np.array([gender, age, openness, neuroticism, conscientiousness, agreeableness, extraversion]).reshape(1,
                                                                                                                    -1)
text = ""
if docx_file is not None:
    file_details = {"Filename": docx_file.name, "FileType": docx_file.type, "FileSize": docx_file.size}
    # st.write(file_details)
    # Check File Type
    if docx_file.type == "text/plain":
        # raw_text = docx_file.read() # read as bytes
        # st.write(raw_text)
        # st.text(raw_text) # fails
        st.text(str(docx_file.read(), "utf-8"))  # empty
        raw_text = str(docx_file.read(), "utf-8")  # works with st.text and st.write,used for further processing
        # st.text(raw_text) # Works
        text = raw_text
        # st.write(raw_text)  # works
    elif docx_file.type == "application/pdf":
        raw_text = read_pdf(docx_file)
        # st.write(raw_text)
        try:
            with pdfplumber.open(docx_file) as pdf:
                page = pdf.pages[0]
                # st.write(page.extract_text())
                text = page.extract_text()
        except:
            st.write("None")
    elif docx_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        # Use the right file processor ( Docx,Docx2Text,etc)
        raw_text = docx2txt.process(docx_file)  # Parse in the uploadFile Class
        # st.write(raw_text)
        text = raw_text

f = open(".txt", "w")
f.write(text)
f.close()
file = ".txt"
# open and read the file after the appending:
f = open(file, "r")
# st.write(f.read())
data = resumeparse.read_file('.txt')
# st.write(data)
skills = data["skills"]
# st.write(skills)
# user_input = np.array(
#     [name, country, year, status, adultMortality, infantDeaths, alcohol, perExpenditure, HepB, Measles, BMI,
#      underFiveDeaths, Polio, totalExpenditure, diphtheria, HIV, GDP, population, thinness1_19, thinness5_9, ISR,
#      schooling]).reshape(1, -1)
# comparing skills
match = 0
for i in range(0, len(skills_required)):
    if skills_required[i] in skills:
        match += 1
percentage = (match / len(skills_required)) * 100


result = ""
if percentage > 50:
    result = "Yes"
else:
    result = "No"
personality = ""
if st.button('Submit'):
    df = pd.read_csv("F:\\5th Semester\\E Artificial Intelligence CSE3013\\3 Projects\\dataset\\personalitydataset.csv")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.mean(), inplace=True)
    gender = df.Gender.unique()
    d = {}
    j = 0
    for i in gender:
        d[i] = j
        j = j + 1
    addC = []
    for i in df['Gender']:
        addC.append(d[i])
    df = df.drop(['Gender'], axis=1)
    df.insert(loc=0, column='Gender', value=addC)

    per = df.Personality.unique()
    s = {}
    j = 0
    for i in per:
        s[i] = j
        j = j + 1
    addS = []
    for i in df['Personality']:
        addS.append(s[i])
        # addC.append(county[i])
    df = df.drop(['Personality'], axis=1)
    df.insert(loc=2, column='Personality', value=addS)

    target = df['Personality'].values
    features = df.drop(['Personality'], axis=1).values

    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2)

    regressor = RandomForestRegressor(n_estimators=100, random_state=0)

    # fit the regressor with x and y data
    model = regressor.fit(features, target)

    y_pred = model.predict(x_test)
    for i in range(len(y_pred)):
        y_pred[i] = round(y_pred[i])
    accuracy = r2_score(y_test, y_pred)

    prediction = model.predict(user_input)
    value = math.floor(prediction[0])
    if value == 0:
        ans = "Dependable"
    elif value == 1:
        ans = "Serious"
    elif value == 2:
        ans = "Responsible"
    elif value == 3:
        ans = "Extraverted"
    elif value == 4:
        ans = "Lively"
    st.write('Thank you! Your submission has been recorded')
    # st.info(ans)
    personality = ans[:]
    row = [name, age, ans, result]
    filename = 'details.csv'
    with open(filename, 'a') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(row)


add = ""
if personality == "Serious":
    add = "Serious"
elif personality == "Dependable":
    add = "Dependable"
elif personality == "Responsible":
    add = "Responsible"
elif personality == "Lively":
    add = "Lively"
elif personality == "Extraverted":
    add = "Extraverted"


