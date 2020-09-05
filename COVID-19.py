#!/usr/bin/env python
# coding: utf-8

# # Part 1

# ## Import modules

# In[1]:


# import modules
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

sns.set_style("darkgrid")

get_ipython().run_line_magic('matplotlib', 'inline')


# ## Reading the data

# The data has been gathered from this link - "https://github.com/CSSEGISandData/COVID-19"(John Hopkins university)

# In[2]:


COVID_CONFIRMED_URL = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
covid_confirmed = pd.read_csv(COVID_CONFIRMED_URL) # The Confirmed DataFrame
covid_confirmed.head()


# In[3]:


COVID_RECOVERED_URL = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv'
covid_recovered = pd.read_csv(COVID_RECOVERED_URL) # The Recovered DataFrame
covid_recovered.head()


# In[4]:


COVID_DEATHS_URL = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
covid_deaths = pd.read_csv(COVID_DEATHS_URL) # The Deaths DataFrame
covid_deaths.head()


# ## Data Cleaning

# Most of the data here is clean and I shall mostly perform replacement of few country names('Mainland China' with 'China') and fill in the blanks.

# In[5]:


# Replace the name 'Mainland China' with just 'China' is all the three DataFrames
covid_confirmed['Country/Region'].replace('Mainland China', 'China', inplace=True)
covid_deaths['Country/Region'].replace('Mainland China', 'China', inplace=True)
covid_recovered['Country/Region'].replace('Mainland China', 'China', inplace=True)


# In[6]:


# Fill empty strings if there are no Province/State
covid_confirmed[['Province/State']] = covid_confirmed[['Province/State']].fillna('')
covid_confirmed.fillna(0, inplace=True)

covid_deaths[['Province/State']] = covid_deaths[['Province/State']].fillna('')
covid_deaths.fillna(0, inplace=True)

covid_recovered[['Province/State']] = covid_recovered[['Province/State']].fillna('')
covid_recovered.fillna(0, inplace=True)


# Perform some checks:

# In[7]:


covid_confirmed.isna().sum().sum()


# In[8]:


covid_recovered.isna().sum().sum()


# In[9]:


covid_deaths.isna().sum().sum()


# ## Data Analysis and Data Wrangling

# The analysis starts with the worldwide impact of the coronavirus. (Need to create new columns, create intermediate DataFrames
# ,  and re-shape of the data. The process which is known as Data Wrangling)

# In[10]:


# total count of the confirmed cases
covid_confirmed_count = covid_confirmed.iloc[:, 4:].sum().max()

covid_confirmed_count


# In[11]:


# total count of the death cases
covid_deaths_count = covid_deaths.iloc[:, 4:].sum().max()

covid_deaths_count


# In[12]:


# total count of the recovered cases
covid_recovered_count = covid_recovered.iloc[:, 4:].sum().max()

covid_recovered_count


# Therefore, the Active case count = Confirmed - ( Deaths + Recovered )

# In[13]:


world_df = pd.DataFrame({
    'confirmed': [covid_confirmed_count],
    'deaths': [covid_deaths_count],
    'recovered': [covid_recovered_count],
    'active': [covid_confirmed_count - covid_deaths_count - covid_recovered_count]
})

world_df


# In[14]:


world_long_df = world_df.melt(value_vars=['active', 'deaths', 'recovered'], var_name="status", value_name="count")
world_long_df['upper'] = 'confirmed'
world_long_df


# In[15]:


# Plot the world_df DataFrame
world_df_plot = px.treemap(world_long_df, path=["upper", "status"], values="count",
                 color_discrete_sequence=['#fbff2b', '#59ff7a', '#ff4a26'],
                 template='plotly_dark')
world_df_plot.show()


# ### Worldwide over the time evolution analysis

# In[16]:


# Daily confirmed cases
covid_worldwide_confirmed = covid_confirmed.iloc[:, 4:].sum(axis=0)
covid_worldwide_confirmed.head()


# In[17]:


# Daily death cases
covid_worldwide_deaths = covid_deaths.iloc[:, 4:].sum(axis=0)
covid_worldwide_deaths.head()


# In[18]:


# Daily recovered cases
covid_worldwide_recovered = covid_recovered.iloc[:, 4:].sum(axis=0)
covid_worldwide_recovered.head()


# In[19]:


# Daily Active Cases
covid_worldwide_active = covid_worldwide_confirmed - covid_worldwide_deaths - covid_worldwide_recovered


# In[20]:


fig, ax = plt.subplots(figsize=(20, 10))
sns.lineplot(x=covid_worldwide_confirmed.index, y=covid_worldwide_confirmed, sort=False, linewidth=2) # Confirmed
sns.lineplot(x=covid_worldwide_deaths.index, y=covid_worldwide_deaths, sort=False, linewidth=2) # Deaths
sns.lineplot(x=covid_worldwide_recovered.index, y=covid_worldwide_recovered, sort=False, linewidth=2) # Recovered
sns.lineplot(x=covid_worldwide_active.index, y=covid_worldwide_active, sort=False, linewidth=3) # Active
ax.lines[0].set_linestyle("--") # Dashed lines for the confirmed cases
plt.suptitle("COVID-19 Time Series Evolution", fontsize=25, fontweight='bold', color='white')
plt.xticks(rotation=45)
plt.ylabel('Number of cases')
ax.legend(['Confirmed', 'Deaths', 'Recovered', 'Active'])
plt.show()


# In[21]:


# Logarithmic Scale
fig, ax = plt.subplots(figsize=(20, 10))
ax.set(yscale="log")
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
sns.lineplot(x=covid_worldwide_confirmed.index, y=covid_worldwide_confirmed, sort=False, linewidth=2)
sns.lineplot(x=covid_worldwide_deaths.index, y=covid_worldwide_deaths, sort=False, linewidth=2)
sns.lineplot(x=covid_worldwide_recovered.index, y=covid_worldwide_recovered, sort=False, linewidth=2)
sns.lineplot(x=covid_worldwide_active.index, y=covid_worldwide_active, sort=False, linewidth=3)
ax.lines[0].set_linestyle("--") # Dashed lines for the confirmed cases
plt.suptitle("COVID-19 worldwide cases over the time", fontsize=25, fontweight='bold', color='white')
plt.title("(logarithmic scale)", color='white')
plt.xticks(rotation=45)
plt.ylabel('Number of cases')
ax.legend(['Confirmed', 'Deaths', 'Recovered', 'Active'])
plt.show()


# ### Recovery and mortality rate over time

# In[22]:


# Rate
world_rate_df = pd.DataFrame({
    'confirmed': covid_worldwide_confirmed,
    'deaths': covid_worldwide_deaths,
    'recovered': covid_worldwide_recovered,
    'active': covid_worldwide_active
}, index=covid_worldwide_confirmed.index)

world_rate_df.tail()


# In[23]:


world_rate_df['recovered / 100 confirmed'] = world_rate_df['recovered'] / world_rate_df['confirmed'] * 100 # Patients recovered to confirmedx100 ratio
world_rate_df['deaths / 100 confirmed'] = world_rate_df['deaths'] / world_rate_df['confirmed'] * 100 # Patients deaths to confirmedx100 ratio
world_rate_df['date'] = world_rate_df.index
world_rate_df.tail()


# In[24]:


world_rate_long_df = world_rate_df.melt(id_vars="date",
                                        value_vars=['recovered / 100 confirmed', 'deaths / 100 confirmed'],
                                        var_name="status",
                                        value_name="ratio")
world_rate_long_df


# In[25]:


fig = px.line(world_rate_long_df, x="date", y="ratio", color='status', log_y=True, 
              title='Recovery and Mortality rate over the time',
              color_discrete_sequence=['#00FF5E', '#FF0000'],
              template='plotly_dark')

fig.show()


# ### Visualizing worldwide COVID-19 cases in a map

# In[26]:


covid_confirmed_agg = covid_confirmed.groupby('Country/Region').sum().reset_index()


# In[27]:


covid_confirmed_agg.loc[:, ['Lat', 'Long']] = covid_confirmed.groupby('Country/Region').mean().reset_index().loc[:, ['Lat', 'Long']]
covid_confirmed_agg


# In[28]:


MIN_CASES = 100 # involve only the cases where number is more than 100
covid_confirmed_agg = covid_confirmed_agg[covid_confirmed_agg.iloc[:, 3:].max(axis=1) > MIN_CASES]


# Convert data from wide to long format:

# In[29]:


print(covid_confirmed_agg.shape)
covid_confirmed_agg.head()


# In[30]:


covid_confirmed_agg_long = pd.melt(covid_confirmed_agg,
                                   id_vars=covid_confirmed_agg.iloc[:, :3],
                                   var_name='date',
                                   value_vars=covid_confirmed_agg.iloc[:, 3:],
                                   value_name='date_confirmed_cases')
print(covid_confirmed_agg_long.shape)
covid_confirmed_agg_long.head()


# In[31]:


fig = px.scatter_geo(covid_confirmed_agg_long,
                     lat="Lat", lon="Long", color="Country/Region",
                     hover_name="Country/Region", size="date_confirmed_cases",
                     size_max=50, animation_frame="date",
                     template='plotly_dark', projection="natural earth",
                     title="COVID-19 worldwide confirmed cases over time")

fig.show()


# # Part 2

# Convert data to long format:

# In[32]:


covid_confirmed_long = pd.melt(covid_confirmed,
                               id_vars=covid_confirmed.iloc[:, :4],
                               var_name='date',
                               value_name='confirmed')

covid_deaths_long = pd.melt(covid_deaths,
                               id_vars=covid_deaths.iloc[:, :4],
                               var_name='date',
                               value_name='deaths')

covid_recovered_long = pd.melt(covid_recovered,
                               id_vars=covid_recovered.iloc[:, :4],
                               var_name='date',
                               value_name='recovered')

covid_confirmed_long.shape


# In[33]:


covid_confirmed_long.head()


# Merge the DataFrames:

# In[34]:


covid_df = covid_confirmed_long
covid_df['deaths'] = covid_deaths_long['deaths']
covid_df['recovered'] = covid_recovered_long['recovered']

print(covid_df.shape)
covid_df.head()


# Calculate active column:

# In[35]:


covid_df['active'] = covid_df['confirmed'] - covid_df['deaths'] - covid_df['recovered']


# In[36]:


print(covid_df.shape)
covid_df.head()


# In[37]:


# Data Cleaning
covid_df['Country/Region'].replace('Mainland China', 'China', inplace=True)
covid_df[['Province/State']] = covid_df[['Province/State']].fillna('')
covid_df.fillna(0, inplace=True)


# In[38]:


covid_df.isna().sum().sum()


# ## Save DataFrame to CSV

# In[39]:


covid_df.to_csv('covid_df.csv', index=None) # Temporary DataFrame


# In[40]:


pd.read_csv('covid_df.csv')


# In[41]:


covid_df.head()


# Aggregate the Data:

# In[42]:


covid_countries_df = covid_df.groupby(['Country/Region', 'Province/State']).max().reset_index()
covid_countries_df


# In[43]:


covid_countries_df = covid_countries_df.groupby('Country/Region').sum().reset_index()
covid_countries_df


# Remove unused 'Lat' and 'Long':

# In[44]:


covid_countries_df.drop(['Lat', 'Long'], axis=1, inplace=True)
covid_countries_df


# In[45]:


# Top 10 countries with confirmed cases:
top_10_confirmed = covid_countries_df.sort_values(by='confirmed', ascending=False).head(10)
top_10_confirmed


# In[46]:


fig = px.bar(top_10_confirmed.sort_values(by='confirmed', ascending=True),
             x="confirmed", y="Country/Region",
             title='Confirmed Cases', 
             text='confirmed',
             template='plotly_dark', 
             orientation='h')

fig.update_traces(marker_color='#F3FF7F', textposition='outside')
fig.show()


# In[47]:


# Top 10 countries with recovered cases:
top_10_recovered = covid_countries_df.sort_values(by='recovered', ascending=False).head(10)
top_10_recovered


# In[48]:


fig = px.bar(top_10_recovered.sort_values(by='recovered', ascending=True),
             x="recovered", y="Country/Region",
             title='Recovered Cases', text='recovered',
             template='plotly_dark', orientation='h')

fig.update_traces(marker_color='#43FF82', textposition='outside')

fig.show()


# In[49]:


# Top 10 countries with recovered cases:
top_10_deaths = covid_countries_df.sort_values(by='deaths', ascending=False).head(10)
top_10_deaths


# In[50]:


fig = px.bar(top_10_confirmed.sort_values(by='deaths', ascending=True),
             x="deaths", y="Country/Region",
             title='Death Cases', text='deaths',
             template='plotly_dark', orientation='h')

fig.update_traces(marker_color='#FF1A00', textposition='outside')

fig.show()


# In[51]:


# Countries with high mortatlity rate ( deaths / confirmed cases ):
covid_countries_df['mortality_rate'] = round(covid_countries_df['deaths'] / covid_countries_df['confirmed'] * 100, 2)
temp = covid_countries_df[covid_countries_df['confirmed'] > 100]
top_20_mortality_rate = temp.sort_values(by='mortality_rate', ascending=False).head(20)
top_20_mortality_rate


# In[52]:


fig = px.bar(top_20_mortality_rate.sort_values(by='mortality_rate', ascending=True),
             x="mortality_rate", y="Country/Region",
             title='Mortality rate', text='mortality_rate',
             template='plotly_dark', orientation='h',
             width=700, height=600)

fig.update_traces(marker_color='#FF6150', textposition='outside')
fig.show()


# In[53]:


# country wise confirmed cases over time
covid_countries_date_df = covid_df.groupby(['Country/Region', 'date'], sort=False).sum().reset_index()
covid_countries_date_df


# In[54]:


# Analyzing the data of India
covid_India = covid_countries_date_df[covid_countries_date_df['Country/Region'] == 'India']
covid_India


# In[55]:


# Other countries where COVID-19 has made a huge impact
covid_US = covid_countries_date_df[covid_countries_date_df['Country/Region'] == 'US']
covid_China = covid_countries_date_df[covid_countries_date_df['Country/Region'] == 'China']
covid_Italy = covid_countries_date_df[covid_countries_date_df['Country/Region'] == 'Italy']
covid_Germany = covid_countries_date_df[covid_countries_date_df['Country/Region'] == 'Germany']
covid_Spain = covid_countries_date_df[covid_countries_date_df['Country/Region'] == 'Spain']
covid_Argentina = covid_countries_date_df[covid_countries_date_df['Country/Region'] == 'Argentina']


# In[56]:


# World except China
covid_no_China = covid_countries_date_df[covid_countries_date_df['Country/Region'] != 'China']
covid_no_China = covid_no_China.groupby('date', sort=False).sum().reset_index()


# In[57]:


fig, ax = plt.subplots(figsize=(16, 6))

sns.lineplot(x=covid_India['date'], y=covid_India['confirmed'], sort=False, linewidth=2)
sns.lineplot(x=covid_US['date'], y=covid_US['confirmed'], sort=False, linewidth=2)
sns.lineplot(x=covid_China['date'], y=covid_China['confirmed'], sort=False, linewidth=2)
sns.lineplot(x=covid_Italy['date'], y=covid_Italy['confirmed'], sort=False, linewidth=2)
sns.lineplot(x=covid_Germany['date'], y=covid_Germany['confirmed'], sort=False, linewidth=2)
sns.lineplot(x=covid_Spain['date'], y=covid_Spain['confirmed'], sort=False, linewidth=2)
sns.lineplot(x=covid_no_China['date'], y=covid_no_China['confirmed'], sort=False, linewidth=2)

plt.suptitle("COVID-19 per country cases over the time", fontsize=16, fontweight='bold', color='white')

plt.xticks(rotation=45)
plt.ylabel('Confirmed cases')
ax.legend(['India', 'US', 'China', 'Italy', 'Germany', 'Spain', 'World except China'])

plt.show()


# In[58]:


fig, ax = plt.subplots(figsize=(16, 6))
ax.set(yscale="log")
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))

sns.lineplot(x=covid_India['date'], y=covid_India['confirmed'], sort=False, linewidth=2)
sns.lineplot(x=covid_US['date'], y=covid_US['confirmed'], sort=False, linewidth=2)
sns.lineplot(x=covid_China['date'], y=covid_China['confirmed'], sort=False, linewidth=2)
sns.lineplot(x=covid_Italy['date'], y=covid_Italy['confirmed'], sort=False, linewidth=2)
sns.lineplot(x=covid_Germany['date'], y=covid_Germany['confirmed'], sort=False, linewidth=2)
sns.lineplot(x=covid_Spain['date'], y=covid_Spain['confirmed'], sort=False, linewidth=2)
sns.lineplot(x=covid_no_China['date'], y=covid_no_China['confirmed'], sort=False, linewidth=2)

plt.suptitle("COVID-19 per country cases over the time", fontsize=16, fontweight='bold', color='white')
plt.title("(logarithmic scale)", color='white')

plt.xticks(rotation=45)
plt.ylabel('Confirmed cases')
ax.legend(['India', 'US', 'China', 'Italy', 'Germany', 'Spain', 'World except China'])

plt.show()


# ## Analyzing select countries

# In[59]:


def plot_country_global_info(country):# Get the no of confirmed, recovered and death cases in a country using treemap
    country_info = covid_countries_df[covid_countries_df['Country/Region'] == country]
    
    country_info_long = country_info.melt(value_vars=['active', 'deaths', 'recovered'],
                                          var_name="status",
                                          value_name="count")

    country_info_long['upper'] = 'Confirmed cases'
    
    fig = px.treemap(country_info_long, path=["upper", "status"], values="count",
                     title=f"Total COVID-19 confirmed cases in {country}",
                     color_discrete_sequence=['#EFFF00', '#11FF00', '#FF0022'],
                     template='plotly_dark')

    fig.data[0].textinfo = 'label+text+value'

    fig.show()

def plot_country_cases_over_time(country, log): # Time series analysis of the country
    country_date_info = covid_countries_date_df[covid_countries_date_df['Country/Region'] == country]
    
    fig, ax = plt.subplots(figsize=(16, 6))

    if log:
        ax.set(yscale="log")
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
        plt.title("(logarithmic scale)", color='white')

    sns.lineplot(x=country_date_info['date'], y=country_date_info['confirmed'], sort=False, linewidth=2)
    sns.lineplot(x=country_date_info['date'], y=country_date_info['deaths'], sort=False, linewidth=2)
    sns.lineplot(x=country_date_info['date'], y=country_date_info['recovered'], sort=False, linewidth=2)
    sns.lineplot(x=country_date_info['date'], y=country_date_info['active'], sort=False, linewidth=2)
                
    ax.lines[0].set_linestyle("--")

    plt.suptitle(f"COVID-19 cases in {country} over the time", fontsize=16, fontweight='bold', color='white')

    plt.xticks(rotation=45)
    plt.ylabel('Number of cases')

    ax.legend(['Confirmed', 'Deaths', 'Recovered', 'Active'])

    plt.show()

def plot_province_cases(country): # Display province distribution using treemap
    covid_provinces_df = covid_df.groupby(['Province/State', 'Country/Region']).max().reset_index()
    
    country_provinces_info = covid_provinces_df[covid_provinces_df['Country/Region'] == country]
    
    has_provinces = country_provinces_info.shape[0] > 1
    
    if (has_provinces):
        country_info_long = country_provinces_info.melt(id_vars=['Province/State'],
                                                        value_vars=['active', 'deaths', 'recovered'],
                                                        var_name="status",
                                                        value_name="count")

        country_info_long['upper'] = 'Confirmed cases'

        fig = px.treemap(country_info_long, path=['upper', "Province/State", "status"],
                         values="count",
                         title=f"Number of COVID-19 confirmed cases per Province/State in {country}",
                         template='plotly_dark')
        
        fig.data[0].textinfo = 'label+text+value'

        fig.show()

def get_country_covid_info(country, log=False):
    plot_country_global_info(country)
    
    plot_country_cases_over_time(country, log)
    
    plot_province_cases(country)


# In[60]:


get_country_covid_info('India')


# In[61]:


get_country_covid_info('US')


# In[62]:


get_country_covid_info('Italy')


# In[63]:


get_country_covid_info('China')


# In[64]:


get_country_covid_info('Spain')


# In[65]:


get_country_covid_info('India', log=True)


# In[66]:


get_country_covid_info('US', log=True)


# In[67]:


get_country_covid_info('Italy', log=True)


# In[68]:


get_country_covid_info('China', log=True)


# In[69]:


get_country_covid_info('Spain', log=True)


# # Part 3

# Convert data to long format:

# In[70]:


covid_confirmed_long = pd.melt(covid_confirmed,
                               id_vars=covid_confirmed.iloc[:, :4],
                               var_name='date',
                               value_name='confirmed')
covid_confirmed_long.shape


# In[71]:


covid_confirmed_long.head()


# In[72]:


covid_confirmed_long['Country/Region'].replace('Mainland China', 'China', inplace=True)
covid_confirmed_long[['Province/State']] = covid_confirmed_long[['Province/State']].fillna('')
covid_confirmed_long.fillna(0, inplace=True)
covid_confirmed_long.isna().sum().sum()


# ## Country wise analysis over time

# In[73]:


covid_countries_date_df = covid_confirmed_long.groupby(['Country/Region', 'date'], sort=False).sum().reset_index()
covid_countries_date_df.drop(['Lat', 'Long'], axis=1, inplace=True)
covid_countries_date_df


# In[74]:


covid_country = covid_countries_date_df[covid_countries_date_df['Country/Region'] == 'India']
covid_country.head()


# In[75]:


COUNTRY='US'
covid_country = covid_countries_date_df[covid_countries_date_df['Country/Region'] == COUNTRY]
covid_country.head()


# In[76]:


days = np.array([i for i in range(len(covid_country['date']))])
days


# In[77]:


fig, ax = plt.subplots(figsize=(16, 6))

sns.lineplot(x=days, y=covid_country['confirmed'],
             markeredgecolor="#3498db", markerfacecolor="#3498db", markersize=8, marker="o",
             sort=False, linewidth=1, color="#3498db")

plt.suptitle(f"COVID-19 confirmed cases in {COUNTRY} over the time", fontsize=16, fontweight='bold', color='white')

plt.ylabel('Confirmed cases')
plt.xlabel('Days since 1/22')
plt.show()


# In[78]:


fig, ax = plt.subplots(figsize=(16, 6))

ax.set(yscale="log")
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
plt.title("(logarithmic scale)", color='white')

sns.lineplot(x=days, y=covid_country['confirmed'],
             markeredgecolor="#3498db", markerfacecolor="#3498db", markersize=8, marker="o",
             sort=False, linewidth=1, color="#3498db")

plt.suptitle(f"COVID-19 confirmed cases in {COUNTRY} over the time", fontsize=16, fontweight='bold', color='white')

plt.ylabel('Confirmed cases')
plt.xlabel('Days since 1/22')

plt.show()


# In[79]:


# Skip the first four weeks
SKIP_DAYS = 30
covid_country_confirmed_sm = list(covid_country['confirmed'][SKIP_DAYS:])
covid_country_confirmed_sm[:15]


# In[80]:


X = days[SKIP_DAYS:].reshape(-1, 1)

X


# In[81]:


y = list(np.log(covid_country_confirmed_sm))

y


# In[82]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)


# ## Builiding a Machine Learning Model

# In[83]:


linear_model = LinearRegression(fit_intercept=True)

linear_model.fit(X_train, y_train)


# In[84]:


y_pred = linear_model.predict(X_test)

y_pred


# In[85]:


print('MAE:', mean_absolute_error(y_pred, y_test))
print('MSE:',mean_squared_error(y_pred, y_test))


# ### Forecasting the next two weeks

# In[86]:


# y = ax + b
a = linear_model.coef_
b = linear_model.intercept_


# In[87]:


X_fore = list(np.arange(len(days), len(days) + 14))
y_fore = [(a*x+b)[0] for x in X_fore]

X_fore, y_fore


# In[88]:


y_train_l = list(np.exp(y_train))
y_test_l = list(np.exp(y_test))
y_pred_l = list(np.exp(y_pred))
y_fore_l = list(np.exp(y_fore))


# In[91]:


fig, ax = plt.subplots(figsize=(16, 6))

ax.set(yscale="log")
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
plt.title("(logarithmic scale)", color='red')

sns.lineplot(x=days, y=covid_country['confirmed'],
             markeredgecolor="#0A26B2", markerfacecolor="#0A26B2", markersize=8, marker="o",
             sort=False, linewidth=1, color="#0A26B2")

sns.lineplot(x=X_train.reshape(-1), y=y_train_l,
             markeredgecolor="#3498db", markerfacecolor="#3498db", markersize=8, marker="o",
             sort=False, linewidth=1, color="#3498db")

sns.lineplot(x=X_test.reshape(-1), y=y_test_l,
             markeredgecolor="#e67e22", markerfacecolor="#e67e22", markersize=8, marker="o",
             sort=False, linewidth=1, color="#e67e22")

sns.lineplot(x=X_test.reshape(-1), y=y_pred_l,
             markeredgecolor="#f1c40f", markerfacecolor="#f1c40f", markersize=8, marker="o",
             sort=False, linewidth=1, color="#f1c40f")

sns.lineplot(x=X_fore, y=y_fore_l,
             markeredgecolor="#2ecc71", markerfacecolor="#2ecc71", markersize=8, marker="o",
             sort=False, linewidth=1, color="#2ecc71")

plt.suptitle(f"COVID-19 confirmed cases and forecasting in {COUNTRY} over the time", fontsize=16, fontweight='bold', color='white')

plt.ylabel('Confirmed cases')
plt.xlabel('Days since 1/22')

plt.legend(['Unused train data', 'Train data', 'Test data', 'Predictions', 'Forecast'])

plt.show()


# In[90]:


fig, ax = plt.subplots(figsize=(16, 6))

sns.lineplot(x=days, y=covid_country['confirmed'],
             markeredgecolor="#0A26B2", markerfacecolor="#0A26B2", markersize=8, marker="o",
             sort=False, linewidth=1, color="#0A26B2")

sns.lineplot(x=X_train.reshape(-1), y=y_train_l,
             markeredgecolor="#3498db", markerfacecolor="#3498db", markersize=8, marker="o",
             sort=False, linewidth=1, color="#3498db")

sns.lineplot(x=X_test.reshape(-1), y=y_test_l,
             markeredgecolor="#e67e22", markerfacecolor="#e67e22", markersize=8, marker="o",
             sort=False, linewidth=1, color="#e67e22")

sns.lineplot(x=X_test.reshape(-1), y=y_pred_l,
             markeredgecolor="#f1c40f", markerfacecolor="#f1c40f", markersize=8, marker="o",
             sort=False, linewidth=1, color="#f1c40f")

sns.lineplot(x=X_fore, y=y_fore_l,
             markeredgecolor="#2ecc71", markerfacecolor="#2ecc71", markersize=8, marker="o",
             sort=False, linewidth=1, color="#2ecc71")

plt.suptitle(f"COVID-19 confirmed cases and forecasting in {COUNTRY} over the time", fontsize=16, fontweight='bold', color='white')

plt.ylabel('Confirmed cases')
plt.xlabel('Days since 1/22')

plt.legend(['Unused train data', 'Train data', 'Test data', 'Predictions', 'Forecast'])
plt.savefig('reg.svg', format='svg', dpi=1200)
plt.show()


# In[ ]:




