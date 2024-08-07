from django.contrib import messages
from django.contrib.auth.models import User
from django.shortcuts import render, redirect
from .forms import NewUserForm
from django.contrib import messages
from django.contrib.auth import login, authenticate  # add thi
from django.contrib.auth.forms import AuthenticationForm  # add this
from sklearn.metrics import accuracy_score
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import HashingVectorizer

# Create your views here

# index is a home page
def index(request):
    return render(request, 'index.html')

def about(request):
    return render(request, 'about.html')

# Registration page
# for filling a Detail of Usernamem,mail,password,confpassword
def registration(request):
    if request.method == "POST":
        form = NewUserForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, "Registration successful.")
            return redirect("login")
        messages.error(
            request, "Unsuccessful registration. Invalid information.")
    form = NewUserForm()
    return render(request=request, template_name= 'registration.html', context={"register_form": form})



# login page
def login(request):
    if request.method == "POST":
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:

                messages.info(request, f"You are now logged in as {username}.")
                return redirect("userhome")
            else:
                messages.error(request, "Invalid username or password.")
        else:
            messages.error(request, "Invalid username or password.")
    form = AuthenticationForm()
    return render(request=request, template_name= 'login.html', context={"login_form": form})

def userhome(request):
    return render(request, 'userhome.html')

# load = upload the csv file


def load(request):
    if request.method == "POST":
        global df
        file = request.FILES['myfile']
        df = pd.read_csv(file)
        df = df.sample(frac=0.2)
        return render(request, 'load.html', {'res': "Data Uploaded Succesfully", })
    return render(request, 'load.html')




# view the uploaded data
# before preprocessing
def view(request):
    global df
    col = df.head(100).to_html
    return render(request, 'view.html', {'table': col})

# After preprocessing
def prepro(request):
    global df
    df["clean_text"].fillna( method ='ffill', inplace = True)
    df["category"].fillna( method ='ffill', inplace = True)
    x = df.drop('category',axis=1)
    y = df["category"]
    messages=x.copy()
    messages.reset_index(inplace=True)
    nltk.download('stopwords')
    messages['clean_text']=messages['clean_text'].apply(str)
    from nltk.stem.porter import PorterStemmer ##stemming purpose
    ps = PorterStemmer()
    corpus = []
    for i in range(0, len(messages)):
        review = re.sub('[^a-zA-Z]', ' ', messages['clean_text'][i])
        review = review.lower()
        review = review.split()
        
        review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
        review = ' '.join(review)
        corpus.append(review)
    df["clean"] = corpus
    col = df.head(100).to_html
    return render(request, 'prepro.html', {'table': col})


def modules(request):
    global df
    if request.method == "POST":

        x = df['clean']
        y = df['category']
        x_train, x_test, y_train, y_test = train_test_split(x,y, stratify=y, test_size=0.3, random_state=42)
        hvectorizer = HashingVectorizer(n_features=10000,norm=None,alternate_sign=False,stop_words='english') 
        x_train = hvectorizer.fit_transform(x_train).toarray()
        x_test = hvectorizer.transform(x_test).toarray()
        model = request.POST['algo']
        if model == "1":
            de = DecisionTreeClassifier()
            de.fit(x_train[:10000], y_train[:10000])
            de_pred = de.predict(x_test[:10000])
            de_ac = accuracy_score(y_test[:10000], de_pred)
            msg = "Accuresy of Decesion Tree is: "+str(de_ac)
            return render(request, 'modules.html', {'msg': msg})

        if model == "2":
            rn = RandomForestClassifier()
            rn.fit(x_train[:10000], y_train[:10000])
            sv_pred = rn.predict(x_test[:10000])
            rn_ac = accuracy_score(y_test[:10000], sv_pred)
            msg = "Accuresy of RandomForest Clasifier: "+str(rn_ac)
            return render(request, 'modules.html', {'msg': msg})

        if model == "3":
            nb = GaussianNB()
            nb.fit(x_train[:500], y_train[:500])
            nv_pred = nb.predict(x_test[:500])
            nb_ac = accuracy_score(y_test[:500], nv_pred)
            msg = "Accuresy of Naive Bias is: "+str(nb_ac)
            return render(request, 'modules.html', {'msg': msg})
    return render(request, "modules.html")

def prediction(request):
    global df
    if request.method == "POST":
        x = df['clean']
        y = df['category']
        x_train, x_test, y_train, y_test = train_test_split(x,y, stratify=y, test_size=0.3, random_state=72)
        hvectorizer = HashingVectorizer(n_features=10000,norm=None,alternate_sign=False,stop_words='english') 
        x_train = hvectorizer.fit_transform(x_train).toarray()
        x_test = hvectorizer.transform(x_test).toarray()
        ab = request.POST['input']
        print(ab)
        rn = RandomForestClassifier()
        rn.fit(x_train[:10000], y_train[:10000])        
        out = rn.predict(hvectorizer.transform([ab]))
        if out == 0:
            msg = "Positive"
        elif out == 1:
            msg = "nutral"
        else:
            msg= "negitive"
        return render(request, "prediction.html", {'msg': msg})
    return render(request, 'prediction.html')

def graph(request):
    return render(request, 'graph.html')