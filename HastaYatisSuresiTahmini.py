#Bir hastanın yatış süresini etkileyen başlıca fonksiyonlar
# Yaş,Hastalık Türü,Tedavi süreci ve komplikasyonlar,Genel sağlık durumu



#import libraries---------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder,LabelEncoder
from sklearn.metrics import mean_squared_error,classification_report,accuracy_score
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
#"Ankara,İstanbul,İzmir" Label Encoding yapılınca Ankara(1),İstanbul(2),İzmir(3)
#"iyi,orta,kötü,çok kötü" OrdinalEncoder(Ordinal:Sıralı) iyi(1),orta(2),kötü(3),çok kötü(4) olmalı.
#classification_report:classification problemleri sonucunda aldığımız rapor
#mean_squared_eror:regression problemlerinde değerlendirme ölçeği.

#load dataset-----------------------
df=pd.read_csv("Hospital_Inpatient_Discharges__SPARCS_De-Identified___2021_20231012.csv")
df_ =df.head(50)
df.info()
#dtypes: float64(2), int64(4), object(27)

describe=df.describe()

los = df["Length of Stay"]=df["Length of Stay"].replace("120 +",120)
#Replace metodu ile 120+olan değeri 120 ile değiştiriyoruz.
df["Length of Stay"] = pd.to_numeric(df["Length of Stay"],errors="coerce")
#Yatış süresi nümerik olmadığı için nümerik değere çevirdik.
los = df["Length of Stay"]


df.isna().sum()
#CCSR Procedure Code:576021
#CCSR Procedure Description:576021

for column in df.columns:
    unique_values=len(df[column].unique())
    print(f"Number of unique values in {column}:{unique_values}")
    
df = df[df["Patient Disposition"] != "Expired"]
#İptal edilmiş.Yani gereksiz veriden kurtulduk.

#EDA-----------------------------------------------
 
#Hasta yarış süresini etkileyen durumlar bunlar olabilir:
#hasta_yatis_suresi = Yaş,Type of Admination(Hastaneye Geliş Şekli),Payment Typology(Ödeme Şekilleri),

sns.boxplot(x = "Payment Typology 1", y = "Length of Stay", data = df)
plt.title ("Payment Typology 1 vs Length of Stay")
#Length of Stay nümerik değil.
#featurların adı uzun olduğu için iç içe geçiyor sorunu çözmek için
plt.xticks(rotation=60)
#Medicare sağlık sigortası tercih edilmiş.
#Yaşlı insanlar medicare sağlık sigortasını tercih etmiş olabilir.

sns.countplot(x = "Age Group",data=df[df["Payment Typology 1"] == "Medicare"],order = ["0 to 17","18 to 29","30 to 49","50 to 69","70 to Older"])
plt.title("Medicare Patients for Age Group")
#Yaşlıların Medicare sağlık sigortası var ve yatış süreleri uzun.

sns.boxplot(x = "Type of Admission", y = "Length of Stay",data = df)
plt.title("Type of Admission vs Length of Stay")
plt.xticks(rotation = 60)
#Hastaneye geliş şekli ile kalma süresi ilişkisi
#Travmalarda hastanede kalma süresi yüksek.

f,ax =plt.subplots()
sns.boxplot(x ="Age Group", y = "Length of Stay",data = df, order = ["0 to 17","18 to 29","30 to 49","50 to 69","70 to Older"])
plt.title("Age Group vs Length of Stay")
plt.xticks(rotation = 60)
ax.set(ylim=(0,25))
#sütuna zoom yapmak yerine limitleri belirledik.
#Yaş ilerledikçe evde kalma süresi artıyor.
#Her sütunda outlier var.

#feature encoding-selection:label encoding--------------------

#feature selection

df = df.drop(["Hospital Service Area","Hospital County","Operating Certificate Number",
              "Facility Name","Zip Code - 3 digits","Patient Disposition","Discharge Year",
              "CCSR Diagnosis Description","CCSR Procedure Description","APR DRG Description",
              "APR MDC Description","APR Severity of Illness Description",
              "Payment Typology 2","Payment Typology 3","Birth Weight","Total Charges","Total Costs"],axis=1,errors='ignore')

#feature-encoding
#eğer çok fazla değer yoksa el ile label encoding yapılabilir.
age_group_index= {"0 to 17":1,"18 to 29":2 ,"30 to 49":3,"50 to 69":4 ,"70 or Older":5}
gender_index= {"U":0 , "F" :1, "M":2}
risk_and_seeverity_index = {np.nan:0,"Minor":1,"Moderate":2,"Major":3,"Extreme":4}

df["Age Group"]=df["Age Group"].apply(lambda x : age_group_index[x])
df["Gender"]=df["Gender"].apply(lambda x : gender_index[x])
df["APR Risk of Mortality"]=df["APR Risk of Mortality"].apply(lambda x : risk_and_seeverity_index[x])

#Çok fazla değer varsa bunu otomatik yapmak için encoder yöntemi kullanmalıyız.
#Race:Irk
encoder = OrdinalEncoder()

df["Race"] = encoder.fit_transform(np.asarray(df["Race"]).reshape(-1, 1))
df["Ethnicity"] = encoder.fit_transform(np.asarray(df["Ethnicity"]).reshape(-1, 1))
df["Type of Admission"] = encoder.fit_transform(np.asarray(df["Type of Admission"]).reshape(-1, 1))
df["CCSR Diagnosis Code"] = encoder.fit_transform(np.asarray(df["CCSR Diagnosis Code"]).reshape(-1, 1))
df["CCSR Procedure Code"] = encoder.fit_transform(np.asarray(df["CCSR Procedure Code"]).reshape(-1, 1))
df["APR Medical Surgical Description"] = encoder.fit_transform(np.asarray(df["APR Medical Surgical Description"]).reshape(-1, 1))
df["Payment Typology 1"] = encoder.fit_transform(np.asarray(df["Payment Typology 1"]).reshape(-1, 1))
df["Emergency Department Indicator"] = encoder.fit_transform(np.asarray(df["Emergency Department Indicator"]).reshape(-1, 1))

# missing value checking---------------------------------------------


df.isna().sum()
df = df.drop("CCSR Procedure Code",axis=1,errors='ignore')
df = df.dropna(subset=["Permanent Facility Id","CCSR Diagnosis Code"])

#train test split
X = df.drop(["Length of Stay"],axis=1)
y = df["Length of Stay"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#regression:train ve test 

dtree = DecisionTreeRegressor(max_depth=10)
dtree.fit(X_train,y_train)
train_prediction = dtree.predict(X_train)
#X_train ve X_test i predict ederek tahmin durumunun olup olmadığını gözlemliyoruz.
test_prediction = dtree.predict(X_test)

print("RMSE:Train",np.sqrt(mean_squared_error(y_train, train_prediction)))
print("RMSE:test",np.sqrt(mean_squared_error(y_test, test_prediction)))
#RMSE:Train 2.84783327422551 
#RMSE:Train 8.005075621885288
"""
OVERFİTTİNG VAR.
RMSE:Train 2.84783327422551 ->7 ise 10-13 olarak tahmin edebiliyoruz.
RMSE:Test 8.005075621885288 -> 2 ise 10-18 olarak tahmin ediyormuşuz.
Bu iki değerin birbirine yakın olmasını tercih ederiz ancak şu an biri 2.9 diğeri 8
Bizim DecisionTree modelimiz train veri setini ezberliyor.
Bunu çözmek için max_depth değerini değiştirdik.
Değiştirmemiz sonucunda:
    RMSE:Train 6.088278470926022
    RMSE:test 6.2415702365490775
    
"""
#Hasta Yatış Süresini Kategorik hale getirme:solve classification problem:train and test


bins =[0,5,10,20,30,50,120]
labels=[5,10,20,30,50,120]
#0-5 aralığının etiketi 5, 5-10 aralığının etiketi 10 şeklinde.

df["los_bin"] =pd.cut(x=df["Length of Stay"],bins=bins)
df["los_label"]=pd.cut(x=df["Length of Stay"],bins=bins,labels=labels)
df_ = df.head(50)

df["los_bin"]=df["los_bin"].apply(lambda x: str(x).replace(",","-"))
df["los_bin"]=df["los_bin"].apply(lambda x: str(x).replace("120","120+"))

f,ax = plt.subplots()
sns.countplot(x="los_bin",data=df)

#Label etiketlendirdikten sonra sınıflandırma problemini çözelim:

new_X = df.drop(["Length of Stay","los_bin","los_label"],axis=1)
new_y = df["los_label"] #kategorik hasta yatış süresi

X_train,X_test,y_train,y_test = train_test_split(new_X, new_y,test_size=0.2,random_state=42)

dtree =DecisionTreeClassifier(max_depth=10)
dtree.fit(X_train,y_train)

#X_train almamızın sebebi ezber yapıyor mu ?
train_prediction = dtree.predict(X_train)
test_prediction = dtree.predict(X_test)

print("Train Accuracy" ,accuracy_score(y_train,train_prediction))
print("Test Accuracy" ,accuracy_score(y_test,test_prediction))
print("Classification Report",classification_report(y_test,test_prediction))

"""
Overfitting Var
Train Accuracy 0.9244704097809807
Test Accuracy 0.6852532503276864
Çözmek için yine max_depth=10 :
    Train Accuracy 0.7418194070663043
    Test Accuracy 0.741067307146401
    Bu max_Depth değerini deneyerek bulmalıyız.
"""












