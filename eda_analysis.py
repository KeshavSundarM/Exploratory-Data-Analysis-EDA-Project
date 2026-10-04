import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

SEED = 42
DATA_FILE = Path("employee_eda_dataset.csv")

def create_dataset(n=600):
    rng=np.random.default_rng(SEED)
    age=rng.integers(21,61,n)
    experience=np.clip(age-rng.integers(20,30,n),0,40)
    department=rng.choice(["IT","Finance","HR","Marketing","Operations"],n,p=[.30,.18,.12,.16,.24])
    education=rng.choice(["Bachelor","Master","PhD"],n,p=[.55,.38,.07])
    income=np.clip(18000+experience*2500+(education=="Master")*8000+(education=="PhD")*18000+rng.normal(0,9000,n),18000,180000).round(0)
    satisfaction=np.clip(3+(department=="IT")*.25+(department=="Finance")*.10+rng.normal(0,.8,n),1,5).round(1)
    hours=np.clip(40+(department=="IT")*2+(department=="Operations")*3+rng.normal(0,5,n),25,65).round(1)
    performance=np.clip(55+experience*.45+satisfaction*4-np.maximum(hours-50,0)*.6+(education=="Master")*2+(education=="PhD")*4+rng.normal(0,8,n),30,100).round(1)
    leave=np.clip(12+satisfaction*1.2+rng.normal(0,4,n),5,30).round(1)
    p=np.clip(.18+np.maximum(hours-45,0)*.025-(satisfaction-3)*.12-experience*.008,.03,.65)
    attrition=rng.binomial(1,p)
    data=pd.DataFrame({"Age":age,"Experience_Years":experience,"Department":department,"Education":education,"Monthly_Income":income,"Job_Satisfaction":satisfaction,"Work_Hours_Per_Week":hours,"Performance_Score":performance,"Annual_Leave_Days":leave,"Attrition":np.where(attrition==1,"Yes","No")})
    for col,count in [("Monthly_Income",10),("Job_Satisfaction",8),("Work_Hours_Per_Week",7)]:
        idx=rng.choice(data.index,count,replace=False)
        data.loc[idx,col]=np.nan
    return pd.concat([data,data.iloc[:5]],ignore_index=True)

if DATA_FILE.exists():
    df=pd.read_csv(DATA_FILE)
else:
    df=create_dataset()
    df.to_csv(DATA_FILE,index=False)

print("Original shape:",df.shape)
print("\nMissing values:")
print(df.isnull().sum())
print("\nDuplicate rows:",df.duplicated().sum())

clean=df.drop_duplicates().copy()
numeric=["Age","Experience_Years","Monthly_Income","Job_Satisfaction","Work_Hours_Per_Week","Performance_Score","Annual_Leave_Days"]
for col in numeric:
    clean[col]=clean[col].fillna(clean[col].median())
clean["Attrition_Flag"]=clean["Attrition"].map({"No":0,"Yes":1})

clean[numeric].describe().T.round(2).to_csv("statistical_summary.csv")
corr=clean[numeric+["Attrition_Flag"]].corr().round(3)
corr.to_csv("correlation_matrix.csv")

dept=clean.groupby("Department").agg(Employees=("Department","size"),Avg_Income=("Monthly_Income","mean"),Avg_Performance=("Performance_Score","mean"),Avg_Satisfaction=("Job_Satisfaction","mean"),Attrition_Rate=("Attrition_Flag","mean")).round(2)
dept["Attrition_Rate"]=(dept["Attrition_Rate"]*100).round(2)
dept.to_csv("department_analysis.csv")

edu=clean.groupby("Education").agg(Employees=("Education","size"),Avg_Income=("Monthly_Income","mean"),Avg_Performance=("Performance_Score","mean")).round(2)
edu.to_csv("education_analysis.csv")

plt.figure(figsize=(8,5)); plt.hist(clean["Monthly_Income"],bins=25); plt.title("Distribution of Monthly Income"); plt.xlabel("Monthly Income"); plt.ylabel("Number of Employees"); plt.tight_layout(); plt.savefig("01_income_distribution.png",dpi=180); plt.close()
plt.figure(figsize=(8,5)); clean.groupby("Department")["Performance_Score"].mean().sort_values().plot(kind="bar"); plt.title("Average Performance Score by Department"); plt.xlabel("Department"); plt.ylabel("Average Performance Score"); plt.tight_layout(); plt.savefig("02_department_performance.png",dpi=180); plt.close()
plt.figure(figsize=(8,5)); clean.groupby("Education")["Monthly_Income"].mean().sort_values().plot(kind="bar"); plt.title("Average Monthly Income by Education"); plt.xlabel("Education"); plt.ylabel("Average Monthly Income"); plt.tight_layout(); plt.savefig("03_education_income.png",dpi=180); plt.close()
plt.figure(figsize=(8,5)); clean.groupby("Department")["Attrition_Flag"].mean().mul(100).sort_values().plot(kind="bar"); plt.title("Attrition Rate by Department"); plt.xlabel("Department"); plt.ylabel("Attrition Rate (%)"); plt.tight_layout(); plt.savefig("04_department_attrition.png",dpi=180); plt.close()
plt.figure(figsize=(8,5)); plt.scatter(clean["Job_Satisfaction"],clean["Performance_Score"],alpha=.5); plt.title("Job Satisfaction vs Performance Score"); plt.xlabel("Job Satisfaction"); plt.ylabel("Performance Score"); plt.tight_layout(); plt.savefig("05_satisfaction_performance.png",dpi=180); plt.close()
plt.figure(figsize=(9,7)); plt.imshow(corr,aspect="auto"); plt.xticks(range(len(corr.columns)),corr.columns,rotation=45,ha="right"); plt.yticks(range(len(corr.index)),corr.index); plt.title("Correlation Matrix")
for i in range(len(corr.index)):
    for j in range(len(corr.columns)): plt.text(j,i,f"{corr.iloc[i,j]:.2f}",ha="center",va="center",fontsize=7)
plt.colorbar(label="Correlation"); plt.tight_layout(); plt.savefig("06_correlation_matrix.png",dpi=180); plt.close()
print("\nEDA completed successfully.")
