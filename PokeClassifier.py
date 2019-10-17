import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import tree
from sklearn.metrics import confusion_matrix

clf = tree.DecisionTreeClassifier()

pokemon = pd.read_csv("C:\\Users\\malac\\Downloads\\pokemon\\Pokemon.csv", index_col = '#')
pokemon = pokemon.sort_values(by = "Legendary")
pokemon.insert(1, "Rel Index", [i for i in range(len(pokemon))])

print(pokemon.tail(67))

types = pokemon['Type 1']
print(types.value_counts())
water = pokemon.loc[pokemon['Type 1'] == 'Water']

means = {"Mean HP": []}
for each_type in types.unique():
    means["Mean HP"].append(pokemon.loc[pokemon['Type 1'] == each_type]['HP'].mean())
    
means = pd.DataFrame(means)
means.insert(0, "Type", [each_type for each_type in types.unique()])
print(means)
    
plt.figure(figsize=(16,6))
ax = sns.barplot(data = means.sort_values('Mean HP', ascending = False), x = "Type", y = "Mean HP")
print(pokemon.head())

leg_pok = pokemon.loc[pokemon["Legendary"] == True].reset_index()
print(leg_pok.head())
plt.figure(figsize=(16,8))
ax = sns.scatterplot(x="Attack", y = "HP", hue ='Legendary', data = pokemon)

reg_pok = pokemon.loc[pokemon["Legendary"] == False].reset_index()
x_set = reg_pok.loc[reg_pok["Rel Index"] < 600].append(leg_pok.loc[leg_pok['Rel Index'] < 780])

x_train=x_set[["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]]
y_train=x_set["Legendary"]

clf.fit(x_train, y_train)

x_t_set = reg_pok.loc[reg_pok["Rel Index"] >= 600].append(leg_pok.loc[leg_pok['Rel Index'] >= 780])
x_test = x_t_set[["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]]
y_test = x_t_set["Legendary"]


prediction = clf.predict(x_t_set[["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]])
y_test = y_test.values

accuracy = 0
correct = 0
total = 0

for i in range(len(prediction)):
    print("Prediction: ", prediction[i], " Actual:", y_test[i])
    if(prediction[i] == y_test[i]):
        correct += 1
    total += 1
accuracy = correct/total
print(correct, " correct out of", total)
print(accuracy)

