import pandas as pd
import numpy as np
import requests
import joblib
from cluster import Clustering

''' Real Code'''
class Supply_Prediction():
    def __init__(self, store_category, store_segment, canton, item_price, pickup_length):
        self.store_category = store_category
        self.store_segment = store_segment
        self.canton = canton
        self.item_price = item_price
        self.pickup_length = pickup_length

    ## Build DF
    def predict(self):
        ## Call Clustering function
        self.cluster = Clustering().predict()
        self.region_language = self.get_language()
        self.region_type = self.get_region_type()
        df = pd.DataFrame({'Store Category': [self.store_category],
                           'Store Segment': [self.store_segment],
                           "Store Region" : [self.canton],
                           'Region Language': [self.region_language],
                           "Region Type" : [self.region_type],
                           "Cluster" : [self.cluster],
                           "Item Price": [self.item_price],
                           "Pickup Length": [self.pickup_length]
                           })

        ## Extend Cluster to initial DF
        pipe_duration = joblib.load('../linear_regression_supply.joblib')
        return pipe_duration.predict(df)[0]

    ## GetLanguage
    def get_language(self):
        region_language = {"Vaud": "french", "Zürich": "german", "St. Gallen": "german", "Valais": "french", "Bern": "german", "Ticino": "italien", "Genève": "french",
            "Aargau": "german", "Basel-Stadt": "german", "Thurgau": "german", "Luzern": "german", "Obwalden": "german", "Solothurn": "german", "Graubünden": "german",
            "Basel-Landschaft": "german", "Freiburg": "french", "Neuchâtel": "french", "Zug": "german", "Schwyz": "german", "Schaffhausen": "german",
            "Appenzell Ausserrhoden": "german", "Appenzell Innerrhoden": "german", "Jura": "french", "Uri": "german", "Glarus": "german", "Nidwalden": "german"}

        return region_language[self.canton]

    def get_region_type(self):
        region_type = {"Vaud": "urban", "Zürich": "urban", "St. Gallen": "urban", "Valais": "rural", "Bern": "urban", "Ticino": "urban", "Genève": "urban",
                       "Aargau": "rural", "Basel-Stadt": "urban", "Thurgau": "rural", "Luzern": "urban", "Obwalden": "rural", "Solothurn": "rural",
                       "Graubünden": "rural", "Basel-Landschaft": "rural", "Freiburg": "urban", "Neuchâtel": "urban", "Zug": "urban", "Schwyz": "rural", "Schaffhausen": "rural",
                       "Appenzell Ausserrhoden": "rural", "Appenzell Innerrhoden": "rural", "Jura": "rural", "Uri": "rural", "Glarus": "rural", "Nidwalden": "rural"}
        return region_type[self.canton]

if __name__ == "__main__":
    print(Supply_Prediction("Key Account", 'Sushi', 'Zürich', "11.90", 20).predict())
    new_bakery = Supply_Prediction("Key Account", 'Sushi', 'Zürich', "11.90", 20)
    print(f'The predicted supply is: {new_bakery.predict()}')
