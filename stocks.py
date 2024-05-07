import wrds
import pandas as pd
from datetime import date

# Establish a connection to the WRDS database
# You need to replace 'username' and 'password' with your WRDS credentials
db = wrds.Connection(wrds_username='vivek12', wrds_password='Powerplay@ec3')

df = db.get_table(library="crsp", table="stocknames", rows=10)
stock = df[df['ticker'] == 'AMGN']
print(stock)