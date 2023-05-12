#Importing Libraries
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import matplotlib.pyplot as plt
import cluster_tools as ct
import matplotlib.patches as mpatches
import scipy.optimize as opt
import numpy as np
import errors as err

#Reading the world bank data into a dataframe
df_GDP_capita = pd.read_csv('GDP_percapita.csv', skiprows=4)
df_GDP_capita = df_GDP_capita.dropna().drop(['Indicator Code', 'Country Code', 'Indicator Name'], axis=1)
df_GDP_capita.set_index('Country Name', drop=True, inplace=True)
print(df_GDP_capita)


# countries are randomly picked from a list
selected_countries = ['Chad', 'Netherlands', 'Congo, Rep.', 'Austria', 'Japan', 'Rwanda',
       'Benin', 'Italy', 'Philippines', 'Fiji', 'Nigeria', 'Kenya', 'Gabon',
       'Bermuda', 'Spain', 'Ireland', 'St. Vincent and the Grenadines',
       'Congo, Dem. Rep.', 'Ghana', 'Luxembourg', 'Niger', 'Singapore', 'Togo',
       'Trinidad and Tobago', 'Colombia', 'Honduras', 'Senegal', 'Thailand',
       'Morocco', 'Papua New Guinea', "Cote d'Ivoire", 'Australia',
       'St. Kitts and Nevis', 'Lesotho', 'Nicaragua', 'Finland',
       'Central African Republic', 'Pakistan', 'Turkiye', 'Botswana', 'Greece',
       'China', 'Sri Lanka', 'Ecuador', 'United States', 'Panama']
df_selected = df_GDP_capita.loc[selected_countries]
df_GDP_capita1 = df_selected[["1960", "1970", "1980", "1990", "2000", "2010", "2020", "2021"]]
print(df_GDP_capita1)

# columns that are not needed are dropped
df_GDP_capita1 = df_GDP_capita1.drop(["1960", "1970"], axis=1)
print(df_GDP_capita1)

#plotting a scatter matrix to show the least correlated years
pd.plotting.scatter_matrix(df_GDP_capita1, figsize=(12, 12), s=5, alpha=0.8)

plt.show()

# extract columns for fitting. 
# .copy() prevents changes in df_fit to affect df_fish.
GDP_capita1_fit = df_GDP_capita1[["1980", "2020"]].copy()

# normalise dataframe and inspect result
# normalisation is done only on the extract columns. .copy() prevents
# changes in df_fit to affect df_fish. This make the plots with the 
# original measurements
GDP_capita1_fit, df_min, df_max = ct.scaler(GDP_capita1_fit)
print(GDP_capita1_fit.describe())
print()

print("n   score")
# loop over trial numbers of clusters calculating the silhouette
for ic in range(2, 7):
    # set up kmeans and fit
    kmeans = cluster.KMeans(n_clusters=ic)
    kmeans.fit(GDP_capita1_fit)     

    # extract labels and calculate silhoutte score
    labels = kmeans.labels_
    print (ic, skmet.silhouette_score(GDP_capita1_fit, labels))
    
# Fit k-means with 2 clusters
kmeans = cluster.KMeans(n_clusters=2)
kmeans.fit(GDP_capita1_fit)

# Add cluster label column to the original dataframe
df_GDP_capita1["cluster_label"] = kmeans.labels_

# Group countries by cluster label
grouped = df_GDP_capita1.groupby("cluster_label")

# Print countries in each cluster
for label, group in grouped:
    print("Cluster", label)
    print(group.index.tolist())
    print()

# Plot clusters with labels
plt.scatter(GDP_capita1_fit["1980"], GDP_capita1_fit["2020"], c=kmeans.labels_, cmap="Set1")
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c="b", marker="d", s=80)
plt.xlabel("1980")
plt.ylabel("2020")
plt.title("2 clusters")

plt.show()

"""
Reading the csv file of the dataset to be used for curve fit and error range 
for comparison for countries in each cluster
"""
Urban_pop = pd.read_csv("Urban_popgrowth.csv", skiprows=4)
Urban_pop = Urban_pop.dropna(how='all')
Urban_pop = Urban_pop.drop(['Indicator Code', 'Country Code', 'Indicator Name','Unnamed: 67'], axis=1)
Urban_pop.set_index('Country Name', drop=True, inplace=True)
print(Urban_pop)

# Picking two countries for comparison
countries = ['United States', 'China']
Urban_pop_countries = Urban_pop.loc[countries]
Urban_pop_countries = Urban_pop_countries.transpose()
Urban_pop_countries = Urban_pop_countries.rename_axis('Year')
Urban_pop_countries = Urban_pop_countries.applymap(lambda x: pd.to_numeric(x, errors='coerce'))
Urban_pop_countries = Urban_pop_countries.dropna()
print(Urban_pop_countries)

#plotting a trial fit using exponential growth
def exp_growth(t, scale, growth):
    """ Computes exponential function with scale and growth as free parameters
    """
    f = scale * np.exp(growth * t)
    return f

# Convert index to numeric
Urban_pop_countries.index = pd.to_numeric(Urban_pop_countries.index, errors='coerce')

# fit exponential growth
popt, pcorr = opt.curve_fit(exp_growth, Urban_pop_countries.index, Urban_pop_countries["United States"],
p0=[4e8, 0.03])
# much better
print("Fit parameter", popt)
Urban_pop_countries["pop_exp"] = exp_growth(Urban_pop_countries.index, *popt)
plt.figure()
plt.plot(Urban_pop_countries.index, Urban_pop_countries["United States"], label="data")
plt.plot(Urban_pop_countries.index, Urban_pop_countries["pop_exp"], label="fit")
plt.legend()
plt.title("Trial fit exponential growth")
plt.show()
print()
print("Population in")
print("2030:", exp_growth(2030, *popt) / 1.0e6, "Mill.")
print("2040:", exp_growth(2040, *popt) / 1.0e6, "Mill.")
print("2050:", exp_growth(2050, *popt) / 1.0e6, "Mill.")

# result not fitting

#Plotting a fit using logistics
def poly(t, c0, c1, c2, c3):
    """ Computes a polynomial c0 + c1*t + c2*t^2 + c3*t^3
    """
    t = t - 1950
    f = c0 + c1*t + c2*t**2 + c3*t**3
    return f

def err_ranges(x, func, popt, sigmas):
    lower = func(x, *(popt - sigmas))
    upper = func(x, *(popt + sigmas))
    return lower, upper

# data loading and preprocessing here

fig, axs = plt.subplots(2, 1, figsize=(10, 10))

for i, country in enumerate(['United States', 'China']):
    popt, pcorr = opt.curve_fit(poly, pd.to_numeric(Urban_pop_countries.index), Urban_pop_countries[country])
    print("Fit parameter for " + country, popt)
    # extract variances and calculate sigmas
    sigmas = np.sqrt(np.diag(pcorr))
    # create extended year range
    years = np.arange(1950, 2051)
    lower, upper = err_ranges(years, poly, popt, sigmas)
    Urban_pop_countries["poly"] = poly(pd.to_numeric(Urban_pop_countries.index), *popt)
    axs[i].plot(Urban_pop_countries.index, Urban_pop_countries[country], label="data")
    axs[i].plot(Urban_pop_countries.index, Urban_pop_countries["poly"], label="fit")
    # plot error ranges with transparency
    axs[i].fill_between(years, lower, upper, alpha=0.5)
    axs[i].legend(loc="upper left")
    axs[i].set_title("Polynomial fit for " + country)

plt.show()