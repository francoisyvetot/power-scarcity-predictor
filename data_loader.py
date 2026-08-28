import pandas as pd


class PowerDataLoader:
    def __init__(self, base_path: str, bidding_zone: list, weather_scenario: list, target_year: list):
        self.base_path = base_path
        self.bidding_zone = bidding_zone
        self.weather_scenario = weather_scenario
        self.target_year = target_year

    def load_system_data(self) -> pd.DataFrame:
        """Load demand and ENS"""
        df = pd.read_csv(f"{self.base_path}/SystemData_Hourly.csv")
        df = df[(df['Bidding Zone'].isin(self.bidding_zone)) & 
                (df['WS'].isin(self.weather_scenario)) & 
                (df['TY'].isin(self.target_year))].copy()
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        return df.set_index('Datetime').sort_index()

    def load_gen_data(self) -> pd.DataFrame:
        """Load the gen and pivot it to get 1 column by tech type"""
        df = pd.read_csv(f"{self.base_path}/Generation_By_Tech_Hourly.csv")

        df = df[(df['Bidding Zone'].isin(self.bidding_zone)) & 
                (df['WS'].isin(self.weather_scenario)) & 
                (df['TY'].isin(self.target_year))].copy()
        
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        
        df_pivot = df.pivot_table(
            index=['Datetime', 'Bidding Zone', 'WS', 'TY'], 
            columns='Technology_group', 
            values='Generation_MW', 
            aggfunc='sum'
        ).fillna(0).reset_index()
        
        return df_pivot.set_index('Datetime').sort_index()