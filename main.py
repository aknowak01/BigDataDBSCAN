# Import bibliotek
import pandas as pd
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# Wczytanie danych z pliku csv
data = pd.read_csv('drinks.csv')

# Zdefiniowanie cech
data['weighted_average'] = (data['beer_servings'] * 0.5 + data['spirit_servings'] * 0.05 + data['wine_servings'] * 0.15) / (0.5 + 0.05 + 0.15)
data['weighted_average'] = data['weighted_average'].round(2)
X = data[['weighted_average', 'total_litres_of_pure_alcohol']]
# Utworzenie modelu DBSCAN z parametrami eps=0.5 i min_samples=5
dbscan = DBSCAN(eps=6, min_samples=2)

# Fitting modelu do danych
dbscan.fit(X)

#
data['cluster'] = dbscan.labels_

data.sort_values(by='cluster', inplace=True)


plt.scatter(data['total_litres_of_pure_alcohol'], data['weighted_average'], c=data['cluster'])
plt.xlabel('Total litres of pure alcohol consumed per person')
plt.ylabel('Weighted average of alcohol consumption in liters (beer, wine, spirits)')
plt.show()

# Wyświetlenie wyniku
pd.DataFrame.to_csv(data, 'drinks_clustered.csv')
print(data)
