import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ipaddress

fraud = pd.read_csv('Fraud_Data.csv')
ip_country = pd.read_csv('IpAddress_to_Country.csv')
credit = pd.read_csv('creditcard.csv')

# Remove duplicates
fraud = fraud.drop_duplicates()
ip_country = ip_country.drop_duplicates()
credit = credit.drop_duplicates()

# Convert date columns to datetime
fraud['signup_time'] = pd.to_datetime(fraud['signup_time'])
fraud['purchase_time'] = pd.to_datetime(fraud['purchase_time'])

print(fraud.info())
print(fraud.head())
print(ip_country.info())
print(ip_country.head())
print(credit.info())
print(credit.head())
print(fraud.isnull().sum())
print(ip_country.isnull().sum())
print(credit.isnull().sum())

# Convert IP addresses to integer
def safe_ip_to_int(x):
    try:
        return int(ipaddress.IPv4Address(x))
    except Exception:
        return None

fraud['ip_int'] = fraud['ip_address'].apply(safe_ip_to_int)
ip_country['lower_int'] = ip_country['lower_bound_ip_address'].apply(lambda x: int(float(x)))
ip_country['upper_int'] = ip_country['upper_bound_ip_address'].apply(lambda x: int(float(x)))

# Map each transaction to a country
def find_country(ip):
    row = ip_country[(ip_country['lower_int'] <= ip) & (ip_country['upper_int'] >= ip)]
    if not row.empty:
        return row.iloc[0]['country']
    return 'Unknown'

fraud['country'] = fraud['ip_int'].apply(find_country)

# Transaction frequency per user
fraud['user_txn_count'] = fraud.groupby('user_id')['user_id'].transform('count')

# Transaction frequency per device
fraud['device_txn_count'] = fraud.groupby('device_id')['device_id'].transform('count')

# Time-based features
fraud['hour_of_day'] = fraud['purchase_time'].dt.hour
fraud['day_of_week'] = fraud['purchase_time'].dt.dayofweek
fraud['time_since_signup'] = (fraud['purchase_time'] - fraud['signup_time']).dt.total_seconds() / 3600  # in hours

# Univariate
fraud['purchase_value'].hist()
plt.title('Purchase Value Distribution')
plt.show()

fraud['age'].hist()
plt.title('Age Distribution')
plt.show()

fraud['class'].value_counts().plot(kind='bar')
plt.title('Fraud Class Distribution')
plt.show()

print("Fraud class distribution:")
print(fraud['class'].value_counts(normalize=True))
print("Credit class distribution:")
print(credit['Class'].value_counts(normalize=True))

# Bivariate
sns.boxplot(x='class', y='purchase_value', data=fraud)
plt.title('Purchase Value by Fraud Class')
plt.show()

sns.countplot(x='age', hue='class', data=fraud)
plt.title('Fraud Count by Age')
plt.show()