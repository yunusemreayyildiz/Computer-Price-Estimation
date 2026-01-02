import math
from pathlib import Path
from category_encoders import CatBoostEncoder
from lightgbm import LGBMRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from catboost import CatBoostRegressor
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
import pickle
from feature_extraction import *
import json

### CODE SETTINGS ###
EXPORT_SETS = False
EXPORT_BENCHMARK_PLOTS = False
RANDOM_STATE = 42

#Project Root
BASE_DIR = Path(__file__).resolve().parent.parent.parent

#Filepaths
dataset_filename = "computer_prices_all.csv"
DATASET_PATH = BASE_DIR / "data" / dataset_filename

#Used for printing importance of features in dataset
def printFeatureImportances(cols, features):
    featureImportance = {}
    for i in range(len(features)):
        featureImportance[cols[i]] = features[i]

    #Print features by importance order
    for value in sorted(featureImportance.values(),reverse=True):
        print(list(featureImportance.keys())[list(featureImportance.values()).index(value)], value)

#This function is used for printing model metrics
def printMetrics(yTrue, yPred, modelName):
    rmse = np.sqrt(mean_squared_error(yTrue, yPred))
    mae = mean_absolute_error(yTrue, yPred)
    r2 = r2_score(yTrue, yPred)
    
    print("\n--- {} ---".format(modelName))
    print("R-squared (R²): {:.4f}".format(r2))
    print("MAE (Mean Absolute Error): ${:.2f}".format(mae))
    print("RMSE (Root Mean Squared Error): ${:.2f}".format(rmse))
    return mae, rmse, r2

#Fixes Nvidia and AMD gpu name formatting
def fixGPUNames(brand, model):
    if brand == "NVIDIA":
        model = model.split()
        return "{} {}{} {}".format(*model, "").strip()
    elif brand == "AMD":
        model = model.split()
        model[1] = model[1][:1]
        return "{} {}{}0 {}".format(*model, "").strip()
    else:
        return model
    
#This function creates CPU and GPU benchmark plots and shows
def scorePlots(df:pd.DataFrame):
    cpu = (
        df.sample(50).groupby("cpu_model", as_index=False)["cpu_score"]
        .mean()
        .sort_values("cpu_score", ascending=True)
    )
    plt.figure(figsize=(1920/100, 1080/100), dpi=100)
    plt.bar(cpu["cpu_model"],cpu["cpu_score"],)
    plt.xticks(rotation=45, ha='right')
    plt.margins(y=0.15)
    plt.title("CPU Benchmark Plot")
    plt.ylabel("CPU Score")
    plt.savefig('cpu_scores.png', dpi=100, bbox_inches='tight')
    plt.show()
    gpu = (
        df.groupby("gpu_model", as_index=False)["gpu_score"]
        .mean()
        .sort_values("gpu_score", ascending=True)
    )
    plt.figure(figsize=(1920/100, 1080/100), dpi=100)
    plt.bar(gpu["gpu_model"],gpu["gpu_score"],)
    plt.xticks(rotation=45, ha='right')
    plt.margins(y=0.15)
    plt.title("GPU Benchmark Plot")
    plt.ylabel("GPU Score")
    plt.savefig('gpu_scores.png', dpi=100, bbox_inches='tight')
    plt.show()

#Assign a random CPU to row by price of computer
def assignCPU(row):
    tier = row["tierGroup"]
    brand = row["brand"]

    #Check if CPU brand is Apple then select a CPU table by this condition
    if brand == "Apple":
        choices = tierCPUMap_apple[tier]
    else:
        choices = tierCPUMap_nonapple[tier]

    if len(choices) == 0:
        choices = cpuTable_apple if brand == "Apple" else cpuTable_nonapple

    return choices.sample(1).iloc[0]

#Assign a random GPU to row by price of computer
def assignGPU(row):
    tier = row["tierGroup"]
    brand = row["brand"]

    #Check if CPU brand is Apple then select a CPU table by this condition
    if brand == "Apple":
        choices = tierGPUMap_apple[tier]
    else:
        choices = tierGPUMap_nonapple[tier]

    if len(choices) == 0:
        choices = gpuTable_apple if brand == "Apple" else gpuTable_nonapple

    return choices.sample(1).iloc[0]

### LOAD THE DATASET ###

df = pd.read_csv(DATASET_PATH) #Read the computer prices dataset
print("Reading dataset successful")

#Below are our columns to include for process and target column is the value we are predicting in model.
features_to_keep = ['brand', 'release_year', 'os', 'device_type', 'cpu_brand', 'cpu_model', 'gpu_brand', "gpu_model", 'vram_gb', 
                    'ram_gb', 'storage_type', 'storage_gb', 'display_type', 'display_size_in', 'resolution', 'refresh_hz']
target = "price"

#Check if there is any missing columns in dataset
missingCols = [col for col in features_to_keep + [target] if col not in df.columns]
if missingCols:
    print("Error: The following columns are missing from the dataset: {}".format(missingCols)) #Give error and exit if there are any missing columns
    exit()
else:
    df = df[features_to_keep + [target]]

df = df.dropna() #Drop the rows with missing values if there are any

#Calculate iqr value then calculate lower and upper boundary to keep data.
q1 = df[target].quantile(0.25)
q3 = df[target].quantile(0.75)
iqr = q3 - q1

lower_bound = q1 - 2.5 * iqr
upper_bound = q3 + 2.5 * iqr
df = df[(df[target] >= lower_bound) & (df[target] <= upper_bound)]
print("Removed outlier rows")

#Fix the gpu_model columns (Nvidia and AMD cards have wrong formatting)
df["gpu_model"] = df.apply(lambda row: fixGPUNames(row["gpu_brand"],row["gpu_model"]), axis=1)

#Applying log on price will fix the skew and help models perform better
df[target] = np.log(df[target])

### FEATURE EXTRACTION FROM DATASET ###
#Convert resolution string into screen width and height and drop the old column.
df[["screen_width", "screen_height"]] = df["resolution"].str.split("x", expand=True).astype(int)
df = df.drop('resolution', axis=1)

#Extract pixel per inch from width and height values
df["diagonal_px"] = np.sqrt(df["screen_width"]**2 + df["screen_height"]**2)
df["ppi"] = df["diagonal_px"] / df["display_size_in"]
df = df.drop("diagonal_px", axis=1)

#Extract cpu features from cpu model
pattern = r"(i[3579]|Ryzen\s[3579]|Apple M)" #Pattern to extract cpu_family string
df["cpu_family"] = df["cpu_model"].str.extract(pattern)
df["cpu_gen"] = df.apply(lambda row: extract_cpu_gen(row["cpu_model"]), axis=1)
df["cpu_tier"] = df.apply(lambda row: extract_cpu_tier(row["cpu_model"]), axis=1)

#Extract gpu features from gpu model
df["gpu_gen"] = df.apply(lambda row: extract_gpu_gen(row["gpu_model"]), axis=1)
df["gpu_tier"] = df.apply(lambda row: extract_gpu_tier(row["gpu_model"]), axis=1)
df["gpu_premium"] = df.apply(lambda row: extract_gpu_premium(row["gpu_model"]), axis=1)

#Calculate cpu and gpu scores by features we extracted
df["cpu_score"] = df.apply(lambda row: calculate_cpu_performance(row["cpu_model"],row["cpu_family"],row["cpu_gen"],row["cpu_tier"]), axis=1)
df["gpu_score"] = df.apply(lambda row: calculate_gpu_performance(row["gpu_model"],row["gpu_gen"],row["gpu_tier"],row["cpu_gen"]), axis=1)

### CPU & GPU Model fix ###
cpuTable = df[["cpu_model","cpu_score"]].sort_values("cpu_score")
gpuTable = df[["gpu_model","gpu_score"]].sort_values("gpu_score")

#Group CPU models into Apple and nonApple models
cpuTable_apple = cpuTable[cpuTable["cpu_model"].str.contains("Apple", case=False)]
cpuTable_nonapple = cpuTable[~cpuTable["cpu_model"].str.contains("Apple", case=False)]

#Do the grouping same like CPU models
gpuTable_apple = gpuTable[gpuTable["gpu_model"].str.contains("Apple", case=False)]
gpuTable_nonapple = gpuTable[~gpuTable["gpu_model"].str.contains("Apple", case=False)]

#Define price boundaries and their labels
price_bins = [0, 600, 900, 1200, 1600, 2200, 3000, 99999]
labels = ["low", "mid-low", "mid", "mid-high", "high", "premium", "ultra"]

#Assign a tier group for each row by their prices
df["tierGroup"] = pd.cut(np.expm1(df["price"]), bins=price_bins, labels=labels)
df["tierGroup"] = df["tierGroup"].astype(str)

#Tier boundaries of CPUs based on performance scores
tierCPUMap_apple = {
    "low": cpuTable_apple[cpuTable_apple.cpu_score < 540],
    "mid-low": cpuTable_apple[cpuTable_apple.cpu_score.between(540, 750, inclusive="left")],
    "mid": cpuTable_apple[cpuTable_apple.cpu_score.between(750, 950, inclusive="left")],
    "mid-high": cpuTable_apple[cpuTable_apple.cpu_score.between(950, 1150, inclusive="left")],
    "high": cpuTable_apple[cpuTable_apple.cpu_score.between(1150, 1350, inclusive="left")],
    "premium": cpuTable_apple[cpuTable_apple.cpu_score.between(1350, 1500, inclusive="left")],
    "ultra": cpuTable_apple[cpuTable_apple.cpu_score > 1500],
}
tierCPUMap_nonapple = {
    "low": cpuTable_nonapple[cpuTable_nonapple.cpu_score < 540],
    "mid-low": cpuTable_nonapple[cpuTable_nonapple.cpu_score.between(540, 750, inclusive="left")],
    "mid": cpuTable_nonapple[cpuTable_nonapple.cpu_score.between(750, 950, inclusive="left")],
    "mid-high": cpuTable_nonapple[cpuTable_nonapple.cpu_score.between(950, 1150, inclusive="left")],
    "high": cpuTable_nonapple[cpuTable_nonapple.cpu_score.between(1150, 1350, inclusive="left")],
    "premium": cpuTable_nonapple[cpuTable_nonapple.cpu_score.between(1350, 1500, inclusive="left")],
    "ultra": cpuTable_nonapple[cpuTable_nonapple.cpu_score > 1500],
}

#Tier boundaries of GPUs based on performance scores
tierGPUMap_apple = {
    "low": gpuTable_apple[gpuTable_apple.gpu_score < 450],
    "mid-low": gpuTable_apple[gpuTable_apple.gpu_score.between(450, 850, inclusive="left")],
    "mid": gpuTable_apple[gpuTable_apple.gpu_score.between(850, 1350, inclusive="left")],
    "mid-high": gpuTable_apple[gpuTable_apple.gpu_score.between(1350, 2000, inclusive="left")],
    "high": gpuTable_apple[gpuTable_apple.gpu_score.between(2000, 2700, inclusive="left")],
    "premium": gpuTable_apple[gpuTable_apple.gpu_score.between(2700, 3400, inclusive="left")],
    "ultra": gpuTable_apple[gpuTable_apple.gpu_score > 3400],
}
tierGPUMap_nonapple = {
    "low": gpuTable_nonapple[gpuTable_nonapple.gpu_score < 450],
    "mid-low": gpuTable_nonapple[gpuTable_nonapple.gpu_score.between(450, 850, inclusive="left")],
    "mid": gpuTable_nonapple[gpuTable_nonapple.gpu_score.between(850, 1350, inclusive="left")],
    "mid-high": gpuTable_nonapple[gpuTable_nonapple.gpu_score.between(1350, 2000, inclusive="left")],
    "high": gpuTable_nonapple[gpuTable_nonapple.gpu_score.between(2000, 2700, inclusive="left")],
    "premium": gpuTable_nonapple[gpuTable_nonapple.gpu_score.between(2700, 3400, inclusive="left")],
    "ultra": gpuTable_nonapple[gpuTable_nonapple.gpu_score > 3400],
}

#For each row assign a CPU and GPU model and their corresponding scores
df[["cpu_model", "cpu_score"]] = df.apply(assignCPU, axis=1).apply(pd.Series)
df[["gpu_model", "gpu_score"]] = df.apply(assignGPU, axis=1).apply(pd.Series)

#Export benchmark plots if export setting is set to True
if EXPORT_BENCHMARK_PLOTS:
    scorePlots(df)
    
#Drop these features because we don't need these since we have score columns.
df = df.drop(["cpu_family","cpu_model","cpu_tier","gpu_model","gpu_tier","gpu_premium","tierGroup"], axis=1)
print("Feature extraction is done")

#Apply OneHotEncoding to make categorical features into 0 and 1
categorical_cols = df.select_dtypes(include = ["object"]).columns
df = pd.get_dummies(df, columns=categorical_cols, dtype=int)
print("One hot encoding is complete")

#Define x and y frames
x = df.drop([target], axis=1)
y = df[target]

#Split the dataset into train and test sets then split the columns into categorical and numeric columns
xTrain, xTest, yTrain, yTest = train_test_split(x,y,test_size=0.2, shuffle=True, random_state=RANDOM_STATE)
categorical = x.select_dtypes(include=["object"]).columns.tolist()
numeric = x.select_dtypes(exclude=["object"]).columns.tolist()

#Export train, test and whole table as .xlsx for data analysis purposes when we need
if EXPORT_SETS:
    trainTable = pd.concat([xTrain, yTrain], axis=1)
    trainTable[target] = yTrain
    testTable= pd.concat([xTest, yTest], axis=1)
    testTable[target] = yTest
    df.to_excel("dataframe.xlsx") #The table without splitting
    trainTable.to_excel("train_table.xlsx")
    testTable.to_excel("test_table.xlsx")
    print("Exported train and test sets into excel file")

#Save last state of dataframe to .csv file before giving to model.
df.to_csv("df_final.csv")

models = []
modelScores = []
yTestActual = np.exp(yTest)

#Training Model 1 - RandomForestRegressor and GridSearchCV
rf = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
param_grid = {
    "n_estimators": [100],
    "max_depth": [20, None],
    "min_samples_leaf": [1,2]
}
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3,
                        n_jobs=-1, scoring="neg_mean_squared_error", verbose = 2)
grid_search.fit(xTrain, yTrain) #Train the Grid Search
best_rf = grid_search.best_estimator_ #Use best estimator for best results
yPredLogRf = best_rf.predict(xTest) #Make predictions on test set
yPredRf = np.exp(yPredLogRf) #Convert log price to actual price
models.append({"name":"Random Forest", "model":rf, "pred":yPredRf}) #Append the model and properties into list

#Training Model 2 - XGBRegressor
xgb = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=8,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
xgb.fit(xTrain, yTrain) #Train the XGBoost model
yPredXGBLog = xgb.predict(xTest) #Make predictions on test set
yPredXGB = np.exp(yPredXGBLog) #Convert log price to actual price
models.append({"name":"XGBoost", "model":xgb, "pred":yPredXGB}) #Append the model and properties into list

#Training Model 3 - CatBoostRegressor
cat = CatBoostRegressor(iterations=2000, depth=8, learning_rate=0.05, loss_function="RMSE")
cat.fit(xTrain, yTrain)
yPredCatLog = cat.predict(xTest)
yPredCat = np.exp(yPredCatLog)
models.append({"name":"CatBoost", "model":cat, "pred":yPredCat}) #Append the model and properties into list

#Training Model 4 - Stacking Regressor (Combined version of others with some addition)
preprocess = ColumnTransformer(
    transformers=[ #Pass categorical columns into CatBoostEncoder
        ("cat_enc", CatBoostEncoder(cols=categorical, random_state=RANDOM_STATE), categorical),
        ("num", "passthrough", numeric),
    ],
    remainder="drop"
)
#Define four models for StackingRegressor
rf = RandomForestRegressor(n_estimators=300,random_state=64, n_jobs=-1)
lgbm = LGBMRegressor(random_state=RANDOM_STATE)
cat = CatBoostRegressor(random_state=RANDOM_STATE)
xgb = XGBRegressor(
    n_estimators=1500,
    learning_rate=0.03,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state = RANDOM_STATE
)
#Created pipeline for preprocessing data first and then training base model
stack = Pipeline([
    ("prep", preprocess),
    ("stack", StackingRegressor(
        estimators=[
            ("rf", rf),
            ("lgbm", lgbm),
            ("cat", cat),
            ("xgb",xgb)
        ],
        final_estimator=LinearRegression(),
        n_jobs=-1
    ))
])
stack.fit(xTrain, yTrain)
yPredStack = np.expm1(stack.predict(xTest))
models.append({"name":"Stack", "model":stack, "pred":yPredStack}) #Append the model and properties into list

### RESULTS AND VISUALIZATION ###

#Calculate row and column amount based on trained model count
figColumns = math.ceil(math.sqrt(len(models)))
figRows = math.ceil(len(models)/figColumns)
fig, axs = plt.subplots(figRows, figColumns) #Create subplots for putting multiple plots into one
space = np.linspace(0,10000,num=2) #For plotting line

figX = 0
figY = 0
metrics = []
for i, model in enumerate(models): #For each trained model print metrics and define plot
    mae, rmse, r2 = printMetrics(yTestActual, model["pred"], model["name"])
    metrics.append({
        "model": models[i]["name"],
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    })
    modelScores.append(r2_score(yTestActual, model["pred"]))

    #Create scatter plot and line with r2_score on title
    axs[figX,figY].scatter(yTestActual, model["pred"], alpha=0.6)
    axs[figX,figY].set_xlabel('Actual Price')
    axs[figX,figY].set_ylabel('Predicted Price')
    axs[figX,figY].set_title('{} (r2_score: {:.5f})'.format(model["name"],r2_score(yTestActual, model["pred"])))
    axs[figX,figY].plot(space, space, 'r--')

    figX += 1
    if figX == figRows:
        figX = 0
        figY += 1
plt.tight_layout()
plt.savefig('actual_vs_predicted.png') #Save the comparison plot
plt.show()

#Save the model with the best score
bestModelIndex = modelScores.index(max(modelScores))
for i in range(len(models)):
    metrics[i]["isUsed"] = True if i == bestModelIndex else False
pickle.dump(models[bestModelIndex]["model"],open('model.obj', 'wb'))
print("Saved model {}".format(models[bestModelIndex]["name"]))

#Save the Mean Absolute Error for model evaluation later on in interface.
with open("PricePrediction_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)